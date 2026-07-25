from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI


PROJECT_DIRECTORY = Path(__file__).resolve().parent
MCP_SERVER_SCRIPT = PROJECT_DIRECTORY / "approved_sources_mcp_server.py"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_DOCUMENT_CHARACTERS = 16_000
DEFAULT_DOCUMENT_SOURCE_ID = "primary_document"


async def _get_document_from_mcp() -> tuple[str, int]:
    if not MCP_SERVER_SCRIPT.is_file():
        raise RuntimeError("The local approved-sources MCP server file is missing.")

    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_SERVER_SCRIPT)],
    )

    async with stdio_client(server_parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(
                "read_approved_source",
                arguments={
                    "source_id": os.getenv(
                        "DOCUMENT_SOURCE_ID", DEFAULT_DOCUMENT_SOURCE_ID
                    ),
                    "maximum_characters": MAX_DOCUMENT_CHARACTERS,
                },
            )

    structured_content = getattr(result, "structuredContent", None)
    if isinstance(structured_content, dict):
        text = structured_content.get("text")
        page_count = structured_content.get("page_count", 0)
        if isinstance(text, str) and text.strip():
            return text, int(page_count)

    text_parts: list[str] = []
    for content in getattr(result, "content", []):
        candidate = getattr(content, "text", None)
        if isinstance(candidate, str):
            text_parts.append(candidate)

    if not text_parts:
        raise RuntimeError("The document MCP server returned no readable content.")

    raw_payload = "\n".join(text_parts)
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError("The document MCP server returned an invalid response.") from exc

    text = payload.get("text")
    page_count = payload.get("page_count", 0)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("The document MCP server did not return document text.")

    return text, int(page_count)


def answer_document_question(question: str) -> tuple[str, int]:
    """Answer a question using text provided by the local MCP server only."""
    question = question.strip()
    if not question:
        raise ValueError("Enter a question about the approved document.")

    document_text, page_count = asyncio.run(_get_document_from_mcp())

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing from .env.")

    model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip()
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        timeout=60.0,
        max_retries=2,
    )

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        max_tokens=900,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise document assistant. Answer only from the supplied "
                    "document text. Do not invent facts or details. Clearly say when the "
                    "document does not provide an answer."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Approved document text:\n---\n{document_text}\n---\n\n"
                    f"Question: {question}"
                ),
            },
        ],
    )

    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        raise RuntimeError("Groq returned an empty document-assistant response.")

    return answer.strip(), page_count

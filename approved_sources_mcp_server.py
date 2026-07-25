from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from pypdf import PdfReader
from pypdf.errors import PdfReadError


PROJECT_DIRECTORY = Path(__file__).resolve().parent
ALLOWLIST_FILE = PROJECT_DIRECTORY / "approved_sources.json"
MAX_RETURNED_CHARACTERS = 20_000
MAX_DIRECTORY_ENTRIES = 250

mcp = FastMCP(
    "Approved Sources Reader",
    instructions=(
        "This local read-only server can access only files and folders explicitly "
        "listed in approved_sources.json. It cannot inspect any other location."
    ),
)


def _load_sources() -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(ALLOWLIST_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError("approved_sources.json is missing.") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("approved_sources.json is not valid JSON.") from exc

    sources = payload.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise RuntimeError("approved_sources.json must contain at least one source.")

    validated: dict[str, dict[str, Any]] = {}
    for source_id, source in sources.items():
        if not isinstance(source_id, str) or not source_id.strip() or not isinstance(source, dict):
            continue

        kind = source.get("kind")
        path_value = source.get("path")
        extensions = source.get("allowed_extensions")

        if kind not in {"file", "directory"} or not isinstance(path_value, str):
            continue
        if not isinstance(extensions, list) or not all(
            isinstance(item, str) and item.startswith(".") for item in extensions
        ):
            continue

        path = Path(path_value).expanduser().resolve()
        if path == Path(path.anchor):
            raise RuntimeError("Drive roots cannot be approved sources.")

        validated[source_id] = {
            "kind": kind,
            "path": path,
            "allowed_extensions": {item.lower() for item in extensions},
        }

    if not validated:
        raise RuntimeError("No valid approved sources were configured.")

    return validated


def _get_source(source_id: str) -> dict[str, Any]:
    sources = _load_sources()
    source = sources.get(source_id)
    if source is None:
        raise PermissionError("That source ID is not in the approved allowlist.")
    return source


def _validate_extension(path: Path, source: dict[str, Any]) -> None:
    if path.suffix.lower() not in source["allowed_extensions"]:
        raise PermissionError("This file extension is not approved for the source.")


def _resolve_file(source: dict[str, Any], relative_path: str) -> Path:
    source_path: Path = source["path"]

    if source["kind"] == "file":
        if relative_path.strip():
            raise PermissionError("An approved file source does not accept a relative path.")
        candidate = source_path
    else:
        requested = Path(relative_path)
        if not relative_path.strip() or requested.is_absolute() or ".." in requested.parts:
            raise PermissionError("Provide a relative file path inside the approved directory.")

        candidate = (source_path / requested).resolve()
        if not candidate.is_relative_to(source_path):
            raise PermissionError("The requested path is outside the approved directory.")

    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError("The approved file does not exist.")

    _validate_extension(candidate, source)
    return candidate


def _extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            reader = PdfReader(str(path))
        except PdfReadError as exc:
            raise RuntimeError("The approved PDF could not be read.") from exc

        if reader.is_encrypted:
            raise RuntimeError("The approved PDF is encrypted and cannot be extracted.")

        text = "\n\n".join((page.extract_text() or "").strip() for page in reader.pages)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")

    text = text.strip()
    if not text:
        raise RuntimeError("The approved file contains no readable text.")

    return text


@mcp.tool()
def list_approved_sources() -> list[dict[str, Any]]:
    """List source IDs and permissions from the local allowlist without exposing contents."""
    return [
        {
            "source_id": source_id,
            "kind": source["kind"],
            "allowed_extensions": sorted(source["allowed_extensions"]),
        }
        for source_id, source in _load_sources().items()
    ]


@mcp.tool()
def list_approved_directory(source_id: str) -> list[str]:
    """List approved files in one configured directory; it never lists outside that folder."""
    source = _get_source(source_id)
    if source["kind"] != "directory":
        raise ValueError("This source ID refers to a file, not a directory.")

    root: Path = source["path"]
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError("The approved directory does not exist.")

    results = [
        entry.name
        for entry in sorted(root.iterdir(), key=lambda item: item.name.lower())
        if entry.is_file() and entry.suffix.lower() in source["allowed_extensions"]
    ]
    return results[:MAX_DIRECTORY_ENTRIES]


@mcp.tool()
def read_approved_source(
    source_id: str,
    relative_path: str = "",
    maximum_characters: int = 12_000,
) -> dict[str, str | bool]:
    """Read one allowed file, or an approved relative file inside an allowed directory."""
    if not 1 <= maximum_characters <= MAX_RETURNED_CHARACTERS:
        raise ValueError(
            f"maximum_characters must be between 1 and {MAX_RETURNED_CHARACTERS}."
        )

    source = _get_source(source_id)
    file_path = _resolve_file(source, relative_path)
    text = _extract_text(file_path)

    return {
        "text": text[:maximum_characters],
        "truncated": len(text) > maximum_characters,
        "source_id": source_id,
        "file_name": file_path.name,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")

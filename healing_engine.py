
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
    

GROQ_API_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
# Keeps DOM-based healing requests below the Groq free-plan token envelope.
MAX_DOM_CHARS = 18_000


@dataclass(frozen=True)
class SelectorHealing:
    selector: str
    strategy: str
    confidence: float
    rationale: str


class HealingEngine:
    """Groq-powered planner and self-healing locator engine."""

    def __init__(self) -> None:
        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Add it to the .env file before running the framework."
            )

        self.model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip()
        if not self.model_name:
            raise RuntimeError("GROQ_MODEL cannot be empty when it is set in .env.")

        self.client = OpenAI(
            api_key=api_key,
            base_url=GROQ_API_BASE_URL,
            timeout=90.0,
            max_retries=2,
        )
        

    def create_plan(self, url: str, task: str) -> list[dict[str, str]]:
        """
        Converts an English testing instruction into executable Playwright actions.
        Navigation is handled by graph.py, so generated actions begin after page load.
        """
        prompt = f"""
You are an enterprise SDET workflow planner. Convert the following web-testing
instruction into a precise sequence of Playwright actions.

Target URL:
{url}

English task:
{task}

Return ONLY valid JSON in this exact format:
{{
  "actions": [
    {{
      "type": "fill|click|press|select_option|check|hover|wait|assert_visible|extract_text",
      "selector": "a valid Playwright CSS selector or xpath= selector",
      "value": "text, key, option value, or milliseconds where applicable",
      "description": "brief human-readable action description"
    }}
  ]
}}

Requirements:
- Use only CSS selectors or XPath selectors beginning with xpath=.
- Prefer stable selectors: data-testid, aria-label, name, id, role-like semantic attributes,
  visible text XPath, and stable form attributes.
- Never use JavaScript, Python, shell commands, browser devtools commands, or arbitrary code.
- For a search workflow, fill the search box, press Enter, then click the requested result.
- Use selector "body" for wait actions when no target element is needed.
- Use assert_visible to verify an important final outcome.
- Do not add navigation actions because the framework opens the URL itself.
- Keep the plan minimal and executable.
"""
        payload = self._generate_json(prompt)
        raw_actions = payload.get("actions")

        if not isinstance(raw_actions, list) or not raw_actions:
            return self._fallback_plan(task)

        actions: list[dict[str, str]] = []
        for raw_action in raw_actions:
            normalized = self._normalize_action(raw_action)
            if normalized is not None:
                actions.append(normalized)

        if not actions:
            return self._fallback_plan(task)

        return actions

    def heal_selector(
        self,
        *,
        original_selector: str,
        action_type: str,
        action_description: str,
        failure_message: str,
        dom_snapshot: str,
    ) -> SelectorHealing:
        """
        Inspects the current page HTML and returns one corrected CSS/XPath selector.
        """
        trimmed_dom = self._prepare_dom_context(dom_snapshot, action_description)

        prompt = f"""
You are a self-healing Playwright locator expert. A browser action failed because
its previous locator no longer matched the current page. Analyze the raw HTML DOM
snapshot and produce one corrected selector.

Failed action type:
{action_type}

Action intent:
{action_description}

Previous selector:
{original_selector}

Failure:
{failure_message}

Raw HTML DOM snapshot:
--- DOM START ---
{trimmed_dom}
--- DOM END ---

Return ONLY valid JSON:
{{
  "selector": "valid CSS selector or xpath= selector",
  "strategy": "short locator strategy name",
  "confidence": 0.0,
  "rationale": "brief explanation based only on the DOM"
}}

Rules:
- Return exactly one selector.
- The selector must target the element needed for the failed action.
- Prefer data-testid, id, name, aria-label, stable attributes, and semantic visible-text XPath.
- Use xpath= only for XPath; otherwise provide a CSS selector.
- Never return JavaScript, XPath functions that execute scripts, code blocks, or prose outside JSON.
- Do not invent attributes absent from the supplied DOM.
"""
        payload = self._generate_json(prompt)

        selector = self._sanitize_selector(str(payload.get("selector", "")))
        if not selector:
            raise RuntimeError("Groq returned an invalid healed selector.")

        strategy = str(payload.get("strategy", "DOM-guided locator"))
        rationale = str(payload.get("rationale", "Selector generated from the current DOM."))

        try:
            confidence = float(payload.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5

        confidence = max(0.0, min(confidence, 1.0))

        return SelectorHealing(
            selector=selector,
            strategy=strategy[:120],
            confidence=confidence,
            rationale=rationale[:500],
        )

    def _generate_json(self, prompt: str) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only a valid JSON object. Never use Markdown fences, "
                        "explanatory text, or extra keys."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=600,
        )

        response_text = response.choices[0].message.content
        if not response_text:
            raise RuntimeError("Groq returned an empty response.")

        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            parsed = json.loads(self._extract_json_object(response_text))

        if not isinstance(parsed, dict):
            raise RuntimeError("Groq returned JSON with an unexpected structure.")

        return parsed

    @staticmethod
    def _prepare_dom_context(dom_snapshot: str, action_description: str) -> str:
        """Remove high-volume non-UI markup while preserving likely target snippets."""
        compact_dom = re.sub(r"<!--.*?-->", " ", dom_snapshot, flags=re.DOTALL)
        compact_dom = re.sub(
            r"<(script|style|noscript|svg|template|iframe)\b[^>]*>.*?</\1>",
            " ",
            compact_dom,
            flags=re.IGNORECASE | re.DOTALL,
        )
        compact_dom = re.sub(r"\s+", " ", compact_dom).strip()

        if len(compact_dom) <= MAX_DOM_CHARS:
            return compact_dom

        excerpts = [compact_dom[:3_000]]
        terms = {
            term.lower()
            for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{3,}", action_description)
            if term.lower() not in {"click", "using", "with", "from", "result", "visible"}
        }

        for term in terms:
            match = re.search(re.escape(term), compact_dom, flags=re.IGNORECASE)
            if match is None:
                continue

            start = max(0, match.start() - 2_500)
            end = min(len(compact_dom), match.end() + 2_500)
            excerpts.append(compact_dom[start:end])

            if sum(len(excerpt) for excerpt in excerpts) >= MAX_DOM_CHARS - 3_000:
                break

        excerpts.append(compact_dom[-2_000:])
        return "\n...\n".join(excerpts)[:MAX_DOM_CHARS]

    @staticmethod
    def _extract_json_object(text: str) -> str:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start < 0 or end < start:
            raise RuntimeError("Could not extract JSON from Groq's response.")

        return cleaned[start : end + 1]

    @staticmethod
    def _sanitize_selector(selector: str) -> str:
        selector = selector.strip().strip("`").replace("\x00", "")

        if selector.startswith("//") or selector.startswith("("):
            selector = f"xpath={selector}"

        disallowed = (
            "javascript:",
            "<script",
            "eval(",
            "document.",
            "window.",
            "\n",
            "\r",
        )

        if (
            not selector
            or len(selector) > 1000
            or any(token in selector.lower() for token in disallowed)
        ):
            return ""

        if selector.startswith("xpath="):
            xpath_value = selector.removeprefix("xpath=").strip()
            if not xpath_value.startswith(("/", "(")):
                return ""
            return f"xpath={xpath_value}"

        return selector

    def _normalize_action(self, action: Any) -> dict[str, str] | None:
        if not isinstance(action, dict):
            return None

        action_type = str(action.get("type", "")).strip().lower()
        allowed_types = {
            "fill",
            "click",
            "press",
            "select_option",
            "check",
            "hover",
            "wait",
            "assert_visible",
            "extract_text",
        }

        if action_type not in allowed_types:
            return None

        selector = self._sanitize_selector(str(action.get("selector", "")))
        value = str(action.get("value", ""))
        description = str(action.get("description", "")).strip()

        if action_type != "wait" and not selector:
            return None

        if action_type == "wait" and not selector:
            selector = "body"

        if not description:
            description = f"{action_type} using {selector}"

        return {
            "type": action_type,
            "selector": selector,
            "value": value,
            "description": description[:300],
        }

    def _fallback_plan(self, task: str) -> list[dict[str, str]]:
        """
        Provides safe execution for common test phrases if an LLM planning response
        is unavailable or malformed.
        """
        normalized_task = " ".join(task.split())
        lowered = normalized_task.lower()
        actions: list[dict[str, str]] = []

        search_match = re.search(
            r"\bsearch\s+(?:for\s+)?['\"]?(.+?)['\"]?(?:\s+(?:and|then)\s+|\s*$)",
            normalized_task,
            flags=re.IGNORECASE,
        )
        if search_match:
            query = search_match.group(1).strip(" .")
            actions.extend(
                [
                    {
                        "type": "fill",
                        "selector": (
                            "input[type='search'], input[name*='search' i], "
                            "input[placeholder*='search' i], input[aria-label*='search' i]"
                        ),
                        "value": query,
                        "description": f"Enter search query: {query}",
                    },
                    {
                        "type": "press",
                        "selector": (
                            "input[type='search'], input[name*='search' i], "
                            "input[placeholder*='search' i], input[aria-label*='search' i]"
                        ),
                        "value": "Enter",
                        "description": "Submit the search",
                    },
                ]
            )

        if "first link" in lowered or "first result" in lowered:
            actions.append(
                {
                    "type": "click",
                    "selector": "xpath=(//a[normalize-space(string())])[1]",
                    "value": "",
                    "description": "Click the first visible link result",
                }
            )

        click_match = re.search(
            r"\bclick\s+(?:on\s+)?['\"]([^'\"]+)['\"]",
            normalized_task,
            flags=re.IGNORECASE,
        )
        if click_match:
            label = click_match.group(1)
            actions.append(
                {
                    "type": "click",
                    "selector": (
                        f"xpath=(//*[self::button or self::a or @role='button']"
                        f"[normalize-space(string())={json.dumps(label)}])[1]"
                    ),
                    "value": "",
                    "description": f"Click element labelled '{label}'",
                }
            )

        if not actions:
            raise RuntimeError(
                "Groq could not generate an executable action plan for this task. "
                "Use a more specific instruction such as "
                "'Search for Automation Anywhere and click the first link'."
            )

        return actions
      
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, TypedDict

from langgraph.graph import END, START, StateGraph
from playwright.sync_api import Browser, BrowserContext, Error as PlaywrightError
from playwright.sync_api import Page, Playwright, TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from healing_engine import HealingEngine


LogSink = Callable[[str, str], None]


@dataclass
class BrowserSession:
    playwright: Playwright
    browser: Browser
    context: BrowserContext
    page: Page

    def close(self) -> None:
        try:
            self.context.close()
        finally:
            try:
                self.browser.close()
            finally:
                self.playwright.stop()


class AutomationState(TypedDict, total=False):
    url: str
    task: str
    actions: list[dict[str, str]]
    action_index: int
    status: str
    result: str
    last_error: str
    failed_action: dict[str, str]
    dom_snapshot: str
    healing_attempts: dict[int, int]
    max_healing_attempts: int
    session: BrowserSession
    engine: HealingEngine
    log_sink: LogSink


def _log(state: AutomationState, level: str, message: str) -> None:
    sink = state.get("log_sink")
    if sink:
        sink(level, message)


def _launch_browser(state: AutomationState) -> dict[str, Any]:
    _log(state, "INFO", "Launching isolated Chromium browser session.")

    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(
        viewport={"width": 1440, "height": 1000},
        ignore_https_errors=False,
    )
    page = context.new_page()
    page.set_default_timeout(12_000)
    page.set_default_navigation_timeout(30_000)

    return {
        "session": BrowserSession(
            playwright=playwright,
            browser=browser,
            context=context,
            page=page,
        ),
        "status": "browser_ready",
    }


def _open_target(state: AutomationState) -> dict[str, Any]:
    session = state["session"]
    url = state["url"]

    _log(state, "INFO", f"Navigating to {url}")
    session.page.goto(url, wait_until="domcontentloaded")

    # Dynamic commerce sites can keep analytics and recommendation requests open
    # indefinitely, so waiting for networkidle makes a successfully loaded page fail.
    try:
        session.page.wait_for_load_state("load", timeout=10_000)
    except PlaywrightTimeoutError:
        _log(
            state,
            "WARNING",
            "Page load event was delayed; continuing after DOM content became available.",
        )

    session.page.wait_for_timeout(800)

    _log(state, "SUCCESS", f"Page loaded: {session.page.title() or url}")
    return {"status": "page_ready"}


def _plan_actions(state: AutomationState) -> dict[str, Any]:
    _log(state, "INFO", "Generating execution plan from the English task.")

    actions = state["engine"].create_plan(state["url"], state["task"])

    for position, action in enumerate(actions, start=1):
        _log(
            state,
            "INFO",
            f"Plan {position}/{len(actions)}: {action['description']} "
            f"[{action['type']}]",
        )

    return {
        "actions": actions,
        "action_index": 0,
        "healing_attempts": {},
        "status": "executing",
    }


def _execute_action(state: AutomationState) -> dict[str, Any]:
    actions = state["actions"]
    index = state["action_index"]

    if index >= len(actions):
        _log(state, "SUCCESS", "All planned actions completed successfully.")
        return {
            "status": "completed",
            "result": "Workflow completed successfully.",
        }

    action = actions[index]
    session = state["session"]
    page = session.page

    _log(
        state,
        "INFO",
        f"Executing {index + 1}/{len(actions)}: {action['description']} "
        f"using selector: {action['selector']}",
    )

    try:
        _perform_action(page, action)

        _log(state, "SUCCESS", f"Action {index + 1} completed.")
        return {
            "action_index": index + 1,
            "status": "executing",
            "last_error": "",
        }

    except (PlaywrightTimeoutError, PlaywrightError, ValueError) as exc:
        failure_message = str(exc)
        dom_snapshot = page.content()

        _log(
            state,
            "WARNING",
            f"Locator failed for action {index + 1}. Starting self-healing analysis.",
        )

        return {
            "status": "needs_healing",
            "last_error": failure_message,
            "failed_action": action,
            "dom_snapshot": dom_snapshot,
        }


def _perform_action(page: Page, action: dict[str, str]) -> None:
    action_type = action["type"]
    selector = action["selector"]
    value = action.get("value", "")

    if action_type == "wait":
        milliseconds = int(value) if value.isdigit() else 1000
        page.wait_for_timeout(max(0, min(milliseconds, 30_000)))
        return

    locator = page.locator(selector).first

    if action_type == "fill":
        locator.wait_for(state="visible")
        locator.fill(value)
    elif action_type == "click":
        locator.wait_for(state="visible")
        locator.click()
    elif action_type == "press":
        locator.wait_for(state="visible")
        locator.press(value or "Enter")
    elif action_type == "select_option":
        locator.wait_for(state="visible")
        locator.select_option(value)
    elif action_type == "check":
        locator.wait_for(state="visible")
        locator.check()
    elif action_type == "hover":
        locator.wait_for(state="visible")
        locator.hover()
    elif action_type == "assert_visible":
        locator.wait_for(state="visible")
    elif action_type == "extract_text":
        locator.wait_for(state="visible")
        text_content = locator.text_content()
        if text_content is None:
            raise ValueError(f"Failed to extract text from selector: {selector}")
        action["extracted_text"] = text_content

    else:
        raise ValueError(f"Unsupported action type: {action_type}")


def _heal_locator(state: AutomationState) -> dict[str, Any]:
    index = state["action_index"]
    attempt_counts = dict(state.get("healing_attempts", {}))
    attempt_number = attempt_counts.get(index, 0) + 1
    max_attempts = state.get("max_healing_attempts", 2)

    if attempt_number > max_attempts:
        message = (
            f"Self-healing exhausted after {max_attempts} attempts for "
            f"action {index + 1}: {state['failed_action']['description']}"
        )
        _log(state, "ERROR", message)
        return {"status": "failed", "result": message}

    attempt_counts[index] = attempt_number
    failed_action = state["failed_action"]

    _log(
        state,
        "INFO",
        f"Healing attempt {attempt_number}/{max_attempts}: analyzing current DOM snapshot.",
    )

    try:
        healed = state["engine"].heal_selector(
            original_selector=failed_action["selector"],
            action_type=failed_action["type"],
            action_description=failed_action["description"],
            failure_message=state["last_error"],
            dom_snapshot=state["dom_snapshot"],
        )

        updated_actions = [dict(action) for action in state["actions"]]
        updated_actions[index]["selector"] = healed.selector

        _log(
            state,
            "SUCCESS",
            f"Healed locator selected ({healed.strategy}, "
            f"confidence {healed.confidence:.0%}): {healed.selector}",
        )
        _log(state, "INFO", f"RCA: {healed.rationale}")

        return {
            "actions": updated_actions,
            "healing_attempts": attempt_counts,
            "status": "executing",
        }

    except Exception as exc:
        message = f"Locator healing failed: {exc}"
        _log(state, "ERROR", message)
        return {
            "healing_attempts": attempt_counts,
            "status": "failed",
            "result": message,
        }


def _after_execute(state: AutomationState) -> str:
    if state["status"] == "needs_healing":
        return "heal_locator"

    if state["status"] == "completed":
        return "end"

    return "execute_action"


def _after_healing(state: AutomationState) -> str:
    if state["status"] == "failed":
        return "end"

    return "execute_action"


def build_graph() -> Any:
    workflow = StateGraph(AutomationState)

    workflow.add_node("launch_browser", _launch_browser)
    workflow.add_node("open_target", _open_target)
    workflow.add_node("plan_actions", _plan_actions)
    workflow.add_node("execute_action", _execute_action)
    workflow.add_node("heal_locator", _heal_locator)

    workflow.add_edge(START, "launch_browser")
    workflow.add_edge("launch_browser", "open_target")
    workflow.add_edge("open_target", "plan_actions")
    workflow.add_edge("plan_actions", "execute_action")

    workflow.add_conditional_edges(
        "execute_action",
        _after_execute,
        {
            "execute_action": "execute_action",
            "heal_locator": "heal_locator",
            "end": END,
        },
    )

    workflow.add_conditional_edges(
        "heal_locator",
        _after_healing,
        {
            "execute_action": "execute_action",
            "end": END,
        },
    )

    return workflow.compile()


def run_workflow(
    *,
    url: str,
    task: str,
    log_sink: LogSink | None = None,
    max_healing_attempts: int = 2,
) -> Iterator[dict[str, Any]]:
    """
    Streams LangGraph node updates and always closes Playwright resources.
    """
    graph = build_graph()
    initial_state: AutomationState = {
        "url": url,
        "task": task,
        "engine": HealingEngine(),
        "log_sink": log_sink,
        "max_healing_attempts": max_healing_attempts,
        "status": "starting",
    }

    current_state: dict[str, Any] = dict(initial_state)
    session: BrowserSession | None = None

    try:
        for update in graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_update in update.items():
                if not isinstance(node_update, dict):
                    continue

                current_state.update(node_update)
                session = current_state.get("session", session)

                yield {
                    "node": node_name,
                    "status": current_state.get("status", "running"),
                    "result": current_state.get("result", ""),
                    "action_index": current_state.get("action_index", 0),
                    "action_count": len(current_state.get("actions", [])),
                }

    except Exception as exc:
        message = f"Workflow terminated unexpectedly: {exc}"
        _log(current_state, "ERROR", message)
        yield {
            "node": "framework_error",
            "status": "failed",
            "result": message,
            "action_index": current_state.get("action_index", 0),
            "action_count": len(current_state.get("actions", [])),
        }

    finally:
        if session is not None:
            session.close()
            _log(current_state, "INFO", "Browser session closed safely.")

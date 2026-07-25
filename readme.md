# Aegis TestOps

Aegis TestOps is an Autonomous Self-Healing Multi-Agent Testing Framework for browser testing. It combines Streamlit, Playwright, GroqCloud, and LangGraph to execute natural-language test workflows, repair broken locators, and expose live execution telemetry.

The framework is designed around enterprise SDET practices including Agentic Process Automation (APA), a Process Reasoning Engine (PRE), Self-Healing Locators, and Root-Cause Analysis (RCA).

## Enterprise Capabilities

- Agentic Process Automation (APA): Converts English test instructions into browser automation workflows.
- Process Reasoning Engine (PRE): Uses Groq-hosted Llama 3.3 to create executable Playwright action plans and repair failed selectors.
- Self-Healing Locators: Inspects the live DOM after a locator failure and dynamically replaces stale CSS or XPath selectors.
- Root-Cause Analysis (RCA): Logs action failures, repaired selectors, healing strategy, and Groq-generated rationale.
- LangGraph Orchestration: Coordinates browser launch, navigation, action execution, healing, retry handling, and clean termination.
- Secure Credentials: Loads the Groq API key with `os.getenv("GROQ_API_KEY")` from a local `.env` file.
- Live Observability: Displays execution events and healing activity directly in the Streamlit dashboard.
- Playwright Automation: Uses a real Chromium browser with visibility checks, browser timeouts, and safe teardown.

## Architecture

```text
Streamlit UI
    |
    v
LangGraph Workflow Orchestrator
    |
    +--> Execution Agent (Playwright)
    |        |
    |        +--> Action succeeds --> Next action
    |        |
    |        +--> Locator fails --> DOM snapshot
    |
    +--> Healing Agent (GroqCloud)
             |
             +--> Corrected CSS/XPath selector
             |
             +--> Retry failed action

## Document Assistant

The Streamlit dashboard also includes a Document Assistant. It launches the local
`approved_sources_mcp_server.py` over stdio and reads the `primary_document` source
from `approved_sources.json`. Add other explicitly approved file or directory
sources to that configuration when needed. The MCP server does not accept
arbitrary absolute paths and cannot read outside configured sources.

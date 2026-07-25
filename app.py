from __future__ import annotations

from datetime import datetime
from urllib.parse import urlparse

import streamlit as st

from graph import run_workflow
from mcp_assistant import answer_document_question


st.set_page_config(
    page_title="Aegis TestOps",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def validate_url(value: str) -> str:
    url = value.strip()

    if not url:
        raise ValueError("Enter a target URL.")

    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a valid HTTP or HTTPS URL.")

    return url


def render_log(logs: list[dict[str, str]], container: st.delta_generator.DeltaGenerator) -> None:
    colors = {
        "INFO": "#8ea0b8",
        "SUCCESS": "#43d19e",
        "WARNING": "#f6c453",
        "ERROR": "#ff6b7a",
    }

    rows = []
    for item in logs[-250:]:
        color = colors.get(item["level"], "#d6deeb")
        rows.append(
            f"<div style='padding: 5px 0; border-bottom: 1px solid #1e293b;'>"
            f"<span style='color:{color}; font-weight:700; margin-right:10px;'>"
            f"{item['level']}</span>"
            f"<span style='color:#7f8ea3; margin-right:10px;'>{item['time']}</span>"
            f"<span style='color:#e5edf8;'>{item['message']}</span>"
            f"</div>"
        )

    container.markdown(
        "<div style='background:#0b1220; border:1px solid #26364d; border-radius:12px; "
        "padding:12px 16px; height:430px; overflow-y:auto; font-family:ui-monospace, "
        "SFMono-Regular, Menlo, monospace; font-size:0.85rem;'>"
        + "".join(rows)
        + "</div>",
        unsafe_allow_html=True,
    )


st.markdown(
    """
<style>
    .stApp {
        background:
            radial-gradient(circle at 12% 4%, rgba(20, 184, 166, 0.18), transparent 28rem),
            radial-gradient(circle at 88% 0%, rgba(59, 130, 246, 0.16), transparent 26rem),
            #070b14;
    }
    .block-container {
        max-width: 1400px;
        padding-top: 2.4rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 1.7rem 2rem;
        border: 1px solid rgba(84, 125, 167, 0.35);
        border-radius: 18px;
        background: linear-gradient(120deg, rgba(11, 24, 42, 0.96), rgba(13, 33, 48, 0.82));
        margin-bottom: 1.3rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.45rem;
        color: #edf7ff;
        letter-spacing: -0.05em;
    }
    .hero p {
        margin: 0.55rem 0 0;
        color: #9eb2c8;
        font-size: 1.04rem;
    }
    .metric-card {
        background: rgba(14, 24, 40, 0.92);
        border: 1px solid #26364d;
        border-radius: 14px;
        padding: 1rem 1.1rem;
    }
    .metric-label {
        color: #8294ab;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .metric-value {
        color: #f2f7fc;
        font-size: 1.35rem;
        font-weight: 750;
        margin-top: 0.3rem;
    }
    div.stButton > button {
        width: 100%;
        min-height: 3rem;
        border: none;
        border-radius: 10px;
        background: linear-gradient(90deg, #0d9488, #2563eb);
        color: white;
        font-weight: 750;
    }
    div.stButton > button:hover {
        border: none;
        color: white;
        background: linear-gradient(90deg, #0f766e, #1d4ed8);
    }
</style>
""",
    unsafe_allow_html=True,
)

if "logs" not in st.session_state:
    st.session_state.logs = []

if "last_result" not in st.session_state:
    st.session_state.last_result = ""

if "last_status" not in st.session_state:
    st.session_state.last_status = "Ready"

st.markdown(
    """
<div class="hero">
    <h1>🛡️ Aegis TestOps</h1>
    <p>Autonomous self-healing Playwright execution powered by Groq and LangGraph.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Control Plane")
    st.markdown(
        "Run natural-language browser tests with live observability, "
        "self-healing locators, and root-cause analysis."
    )
    healing_attempts = st.slider(
        "Maximum healing attempts per action",
        min_value=1,
        max_value=5,
        value=2,
    )
    st.divider()
    st.caption("Security: the Groq API key is read only from `.env`.")

left, right = st.columns([1.05, 1.95], gap="large")

with left:
    st.markdown("### Test Request")

    url_input = st.text_input(
        "Target URL",
        placeholder="https://www.google.com",
        help="The browser opens this URL before running the task.",
    )

    task_input = st.text_area(
        "English task",
        placeholder="Search for Automation Anywhere and click the first link",
        height=160,
        help="Describe the task as a clear end-user workflow.",
    )

    run_clicked = st.button("Run Autonomous Test", type="primary")

    st.markdown("### Execution State")
    status_placeholder = st.empty()
    result_placeholder = st.empty()

with right:
    st.markdown("### Live Execution Log")
    log_placeholder = st.empty()

    st.markdown("### Telemetry")
    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.markdown(
        f"<div class='metric-card'><div class='metric-label'>Status</div>"
        f"<div class='metric-value'>{st.session_state.last_status}</div></div>",
        unsafe_allow_html=True,
    )
    metric_two.markdown(
        "<div class='metric-card'><div class='metric-label'>Browser</div>"
        "<div class='metric-value'>Chromium</div></div>",
        unsafe_allow_html=True,
    )
    metric_three.markdown(
        "<div class='metric-card'><div class='metric-label'>Reasoning</div>"
        "<div class='metric-value'>Groq Llama 3.3</div></div>",
        unsafe_allow_html=True,
    )

render_log(st.session_state.logs, log_placeholder)

st.divider()
st.markdown("### Document Assistant")
st.caption(
    "Answers use only the selected approved document through the local read-only MCP server."
)

document_question = st.text_area(
    "Ask about the approved document",
    placeholder="Summarize this document and list its key points.",
    height=110,
    key="document_question",
)

if st.button("Ask Document Assistant", type="secondary"):
    try:
        with st.spinner("Reading the approved document through MCP and preparing an answer..."):
            document_answer, document_page_count = answer_document_question(document_question)

        st.success(f"Answer generated from the approved {document_page_count}-page document.")
        st.markdown(document_answer)
    except Exception as exc:
        st.error(f"Document Assistant could not answer: {exc}")

if run_clicked:
    try:
        target_url = validate_url(url_input)

        if not task_input.strip():
            raise ValueError("Enter an English task for the automation agent.")

        st.session_state.logs = []
        st.session_state.last_result = ""
        st.session_state.last_status = "Running"

        def log_sink(level: str, message: str) -> None:
            st.session_state.logs.append(
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "level": level,
                    "message": message,
                }
            )
            render_log(st.session_state.logs, log_placeholder)

        status_placeholder.info("Initializing autonomous test run...")
        final_event: dict[str, str] | None = None

        for event in run_workflow(
            url=target_url,
            task=task_input.strip(),
            log_sink=log_sink,
            max_healing_attempts=healing_attempts,
        ):
            final_event = event
            current_status = event["status"].replace("_", " ").title()
            status_placeholder.info(
                f"Status: {current_status} · "
                f"Action {event['action_index']}/{event['action_count']}"
            )

        if final_event and final_event["status"] == "completed":
            st.session_state.last_status = "Passed"
            st.session_state.last_result = final_event["result"]
            status_placeholder.success("Test completed successfully.")
            result_placeholder.success(final_event["result"])
        else:
            failure_message = (
                final_event["result"]
                if final_event and final_event.get("result")
                else "The test did not complete."
            )
            st.session_state.last_status = "Failed"
            st.session_state.last_result = failure_message
            status_placeholder.error("Test execution failed.")
            result_placeholder.error(failure_message)

    except Exception as exc:
        st.session_state.last_status = "Configuration Error"
        st.session_state.last_result = str(exc)
        status_placeholder.error("Unable to start the test.")
        result_placeholder.error(str(exc))

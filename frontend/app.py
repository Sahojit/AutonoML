"""
Streamlit Dashboard — Multi-Agent AI System v2.1
─────────────────────────────────────────────────
Upgraded panels (matching 5 new capabilities):
  • Per-agent model/temperature display     (Step 4 — Multi-Model Strategy)
  • Hybrid eval panel: LLM + programmatic   (Step 1 — Hybrid Evaluation)
  • Programmatic / ML-sanity issue badges   (Step 3 — ML-Aware Evaluation)
  • Self-reflection memory indicator        (Step 2 — Self-Reflection Memory)
  • Auto-experiment leaderboard renderer    (Step 5 — Auto-Experimentation)
  • Planner task graph with dependency arrows
  • Agent execution timeline
  • Tool-usage display
  • Evaluation score meter
"""
from __future__ import annotations

import html as _html
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL = os.getenv("FRONTEND_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Multi-Agent AI System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* ── Global dark-theme overrides ── */
    .stApp, section[data-testid="stSidebar"] {
        background-color: #0e1117;
        color: #e6edf3;
    }
    .stChatMessage {
        background: transparent !important;
    }

    /* User bubble */
    .chat-user {
        background-color: #1f6feb;
        color: #ffffff;
        padding: 12px 16px;
        border-radius: 12px 12px 4px 12px;
        margin-bottom: 8px;
        font-size: 1rem;
        line-height: 1.6;
        word-break: break-word;
    }

    /* AI bubble */
    .chat-ai {
        background-color: #161b22;
        color: #e6edf3;
        border: 1px solid #30363d;
        padding: 14px 18px;
        border-radius: 4px 12px 12px 12px;
        margin-bottom: 8px;
        font-size: 1rem;
        line-height: 1.7;
        word-break: break-word;
    }

    /* Input boxes */
    .stTextInput input, .stTextArea textarea, .stChatInputContainer textarea {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
        border-color: #30363d !important;
        font-size: 1rem !important;
    }

    /* Markdown text */
    .stMarkdown, .stMarkdown p, .stMarkdown li {
        color: #e6edf3;
        font-size: 1rem;
        line-height: 1.7;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 2px 9px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .badge-research   { background:#1c3a5e; color:#58a6ff; }
    .badge-execution  { background:#1a3a2a; color:#3fb950; }
    .badge-evaluation { background:#3a2a00; color:#e3b341; }
    .badge-pass       { background:#1a3a2a; color:#3fb950; }
    .badge-fail       { background:#3a1a1a; color:#f85149; }
    .badge-tool       { background:#2a1a3a; color:#bc8cff; }
    .badge-prog-fail  { background:#3a1a1a; color:#f85149; }
    .badge-prog-pass  { background:#1a3a2a; color:#3fb950; }
    .badge-reflect    { background:#3a2800; color:#e3b341; }
    .badge-model      { background:#1c2a4a; color:#79c0ff; }
    .badge-route      { background:#21262d; color:#8b949e; }

    /* Step card output block */
    .output-block {
        font-size: 0.9rem;
        line-height: 1.6;
        color: #c9d1d9;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 12px 16px;
        margin-top: 6px;
        white-space: pre-wrap;
        word-break: break-word;
        font-family: "JetBrains Mono", "Fira Code", "Source Code Pro", monospace;
    }

    /* Final streaming answer */
    .answer-block {
        font-size: 1rem;
        line-height: 1.75;
        color: #e6edf3;
    }

    /* Section labels */
    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 10px;
        margin-bottom: 4px;
    }

    /* Leaderboard */
    .leaderboard-row { display:flex; align-items:center; padding:4px 0; font-size:0.85rem; }
    .leaderboard-best { background:#1a3a2a; border-left:3px solid #3fb950;
                        padding-left:6px; border-radius:3px; }

    /* Expanders */
    .streamlit-expanderHeader { color: #8b949e !important; font-size:0.85rem !important; }

    /* Dividers */
    hr { border-color: #21262d !important; }

    /* st.code() blocks — match dark theme */
    .stCode > div, pre {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
    }
    .stCode code {
        font-family: "JetBrains Mono", "Fira Code", "Cascadia Code", monospace !important;
        font-size: 0.88rem !important;
        color: #e6edf3 !important;
    }

    /* Buttons */
    .stButton button {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border-color: #30363d !important;
    }
    .stButton button:hover {
        background-color: #30363d !important;
        border-color: #58a6ff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "last_step_records" not in st.session_state:
    st.session_state["last_step_records"] = []

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AGENT_ICONS = {
    "planner":    "📋",
    "research":   "🔍",
    "execution":  "⚙️",
    "evaluation": "✅",
}

_TYPE_COLORS = {
    "research":   "#3b82f6",
    "execution":  "#22c55e",
    "evaluation": "#f59e0b",
}

_STATUS_COLORS = {
    "done":    "🟢",
    "failed":  "🔴",
    "running": "🟡",
    "pending": "⚪",
    "skipped": "⚫",
}

# Per-agent model profiles (mirrors llm_loader._AGENT_PROFILES)
_AGENT_TEMPS = {
    "planner":    ("temp=0.2", "Deterministic planning"),
    "research":   ("temp=0.4", "Creative synthesis"),
    "execution":  ("temp=0.1", "Precise code generation"),
    "evaluation": ("temp=0.0", "Strict judging"),
}


# Route badge colours
_ROUTE_STYLE = {
    "direct":   ("⚡", "#e0e7ff", "#3730a3"),
    "rag":      ("🧠", "#fef9c3", "#a16207"),
    "tool":     ("🌐", "#f3e8ff", "#6b21a8"),
    "pipeline": ("🔄", "#dcfce7", "#15803d"),
    "cache":    ("💾", "#f1f5f9", "#475569"),
}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def call_chat_smart(query: str, session_id: str) -> Optional[Dict[str, Any]]:
    """POST /chat/smart — adaptive routing, non-streaming."""
    try:
        resp = requests.post(
            f"{API_URL}/chat/smart",
            json={"query": query, "session_id": session_id},
            timeout=360,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the backend API. Is it running on port 8000?")
    except requests.exceptions.Timeout:
        st.error("Request timed out. The model may be busy — try again.")
    except Exception as exc:
        st.error(f"API error: {exc}")
    return None


def stream_chat(query: str, session_id: str):
    """
    Generator that yields decoded text chunks from POST /chat/stream.
    Strips SSE 'data: ' prefix and skips control messages (__ROUTE__, __STATUS__).
    Yields (chunk_text, route_str, status_str).
    """
    route  = "pipeline"
    status = ""
    try:
        with requests.post(
            f"{API_URL}/chat/stream",
            json={"query": query, "session_id": session_id},
            stream=True,
            timeout=360,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue
                text = line[6:]   # strip "data: "
                if text.startswith("__ROUTE__:"):
                    route = text.split(":", 1)[1].strip()
                    yield ("", route, status)
                elif text.startswith("__STATUS__:"):
                    status = text.split(":", 1)[1].strip()
                    yield ("", route, status)
                else:
                    yield (text, route, status)
    except Exception as exc:
        yield (f"\n[Stream error: {exc}]", route, status)


def call_chat(query: str, session_id: str) -> Optional[Dict[str, Any]]:
    """POST /chat — original full-pipeline endpoint (fallback)."""
    try:
        resp = requests.post(
            f"{API_URL}/chat",
            json={"query": query, "session_id": session_id},
            timeout=360,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the backend API. Is it running on port 8000?")
    except requests.exceptions.Timeout:
        st.error("Request timed out. The model may be busy — try again.")
    except Exception as exc:
        st.error(f"API error: {exc}")
    return None


def call_ingest(text: str, metadata: Dict, session_id: str) -> Optional[Dict]:
    try:
        resp = requests.post(
            f"{API_URL}/memory",
            json={"text": text, "metadata": metadata, "session_id": session_id},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"Ingestion error: {exc}")
    return None


def get_status() -> Optional[Dict]:
    try:
        return requests.get(f"{API_URL}/status", timeout=3).json()
    except Exception:
        return None


def get_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Auto-experiment leaderboard renderer  (Step 5)
# ---------------------------------------------------------------------------

def _try_parse_leaderboard(text: str) -> Optional[List[Dict]]:
    """
    Detect and parse an auto-experiment leaderboard from plain-text output.
    Returns list of {model, accuracy, f1_weighted} dicts or None.
    """
    if "leaderboard" not in text.lower() and "auto-experiment" not in text.lower():
        return None
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if "accuracy=" in line.lower() and "f1=" in line.lower():
            try:
                parts = line.split()
                model_name = None
                acc = f1 = None
                for part in parts:
                    if "accuracy=" in part.lower():
                        acc = float(part.split("=")[1])
                    elif "f1=" in part.lower():
                        f1 = float(part.split("=")[1])
                # grab model name: first token that isn't a number
                for part in parts:
                    cleaned = part.strip("0123456789. ")
                    if cleaned and "=" not in cleaned and len(cleaned) > 2:
                        model_name = cleaned
                        break
                if model_name and acc is not None:
                    rows.append({"model": model_name, "accuracy": acc, "f1_weighted": f1 or 0.0})
            except (ValueError, IndexError):
                pass
    return rows if rows else None


def render_auto_experiment_leaderboard(leaderboard: List[Dict]) -> None:
    """Render a colour-coded leaderboard table for auto-experiment results."""
    st.markdown("##### 🏆 Auto-Experiment Leaderboard")
    if not leaderboard:
        return
    best_acc = max(r["accuracy"] for r in leaderboard)

    cols = st.columns([3, 2, 2, 1])
    cols[0].markdown("**Model**")
    cols[1].markdown("**Accuracy**")
    cols[2].markdown("**F1 (weighted)**")
    cols[3].markdown("**Rank**")

    for rank, row in enumerate(leaderboard, 1):
        is_best = row["accuracy"] == best_acc
        bg = "#f0fdf4" if is_best else "white"
        border = "2px solid #22c55e" if is_best else "1px solid #e5e7eb"
        with st.container():
            c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
            label = f"{'🥇 ' if is_best else ''}{row['model']}"
            c1.markdown(
                f'<div style="background:{bg};border:{border};border-radius:4px;'
                f'padding:4px 8px;">{label}</div>',
                unsafe_allow_html=True,
            )
            c2.progress(min(float(row["accuracy"]), 1.0),
                        text=f"{row['accuracy']:.4f}")
            c3.progress(min(float(row.get("f1_weighted", 0)), 1.0),
                        text=f"{row.get('f1_weighted', 0):.4f}")
            c4.markdown(f"**#{rank}**")


# ---------------------------------------------------------------------------
# Planner task graph  (dependency-aware)
# ---------------------------------------------------------------------------

def render_task_graph(plan: Dict[str, Any]) -> None:
    tasks = plan.get("tasks") or plan.get("steps", [])
    if not tasks:
        return

    st.markdown("#### 📋 Task Graph")

    cols = st.columns(len(tasks))
    for i, task in enumerate(tasks):
        stype    = task.get("type", "execution")
        color    = _TYPE_COLORS.get(stype, "#9ca3af")
        dep_ids  = task.get("depends_on", [])
        dep_str  = f"← depends on {dep_ids}" if dep_ids else "⚡ independent"
        task_txt = task.get("task", "")

        with cols[i]:
            st.markdown(
                f"""
                <div style="border:2px solid {color}; border-radius:8px;
                            padding:10px; text-align:center; min-height:130px;">
                    <div style="font-size:1.5rem">{_AGENT_ICONS.get(stype,'🔹')}</div>
                    <div class="badge badge-{stype}">{stype.upper()}</div>
                    <div style="font-size:0.7rem; margin-top:5px; color:#374151;">
                        Task {task.get('id','?')}
                    </div>
                    <div style="font-size:0.68rem; color:#6b7280; margin-top:3px;">
                        {dep_str}
                    </div>
                    <div style="font-size:0.75rem; margin-top:6px; color:#1f2937;">
                        {task_txt[:70]}{'…' if len(task_txt) > 70 else ''}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        # Arrow between columns
        if i < len(tasks) - 1:
            pass  # columns are naturally side-by-side


# ---------------------------------------------------------------------------
# Execution timeline
# ---------------------------------------------------------------------------

def render_timeline(step_records: List[Dict[str, Any]]) -> None:
    timed = [
        (r["step_id"], r["step_type"], r.get("duration_s") or 0)
        for r in step_records
    ]
    if not any(d for _, _, d in timed):
        return

    st.markdown("#### ⏱ Execution Timeline")
    total_t = sum(d for _, _, d in timed) or 1
    for step_id, stype, dur in timed:
        pct   = max(int(dur / total_t * 100), 1)
        color = _TYPE_COLORS.get(stype, "#9ca3af")
        st.markdown(
            f'<div style="display:flex;align-items:center;margin-bottom:4px;">'
            f'<span style="width:100px;font-size:0.8rem;color:#374151;">'
            f'Step {step_id} [{stype[:4]}]</span>'
            f'<div style="background:{color};height:16px;width:{pct}%;'
            f'border-radius:3px;margin:0 8px;min-width:4px;"></div>'
            f'<span style="font-size:0.75rem;color:#6b7280;">{dur:.2f}s</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tool trace renderer
# ---------------------------------------------------------------------------

def render_tool_badges(text: str, stype: str) -> None:
    """Show tool-call badges detected in the output text."""
    if stype != "execution" or not text:
        return

    tools_found = []
    text_upper = text.upper()
    if "```PYTHON" in text_upper or "PYTHONEXECUTOR" in text_upper or "PYTHON_EXEC" in text_upper:
        tools_found.append("🐍 python_exec")
    if "WEB_SEARCH" in text_upper or "WEBSEARCH" in text_upper:
        tools_found.append("🌐 web_search")
    if "SQL_QUERY" in text_upper or "SQL QUERY" in text_upper:
        tools_found.append("🗄️ sql_query")
    if "DATASET_ANALYZE" in text_upper:
        tools_found.append("📊 dataset_analyze")
    if "AUTO_EXPERIMENT" in text_upper or "AUTO-EXPERIMENT" in text_upper or "LEADERBOARD" in text_upper:
        tools_found.append("🏆 auto_experiment")

    if tools_found:
        badges_html = " ".join(
            f'<span class="badge badge-tool">{t}</span>' for t in tools_found
        )
        st.markdown(f"**Tools used:** {badges_html}", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Hybrid evaluation panel  (Step 1 + Step 3)
# ---------------------------------------------------------------------------

def render_hybrid_eval_panel(rec: Dict[str, Any]) -> None:
    """
    Show the two-layer evaluation breakdown:
      (A) LLM verdict
      (B) Programmatic / ML-sanity issues
    """
    verdict      = rec.get("verdict")
    score        = rec.get("eval_score")
    issues       = rec.get("eval_issues", [])
    prog_issues  = rec.get("programmatic_issues", [])   # new field from upgrade
    llm_verdict  = rec.get("llm_verdict")               # new field from upgrade
    suggestions  = rec.get("suggestions", [])

    if verdict is None and score is None:
        return

    st.markdown("**Evaluation:**")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**(A) LLM Evaluation**")
        lv = llm_verdict or verdict or "—"
        lv_color = "#22c55e" if lv == "PASS" else "#ef4444"
        st.markdown(
            f'<span class="badge" style="background:{lv_color}20;color:{lv_color};">'
            f'LLM: {lv}</span>',
            unsafe_allow_html=True,
        )
        if score is not None:
            st.progress(float(score), text=f"Score: {score:.2f}")

        llm_only_issues = [i for i in issues if i not in (prog_issues or [])]
        if llm_only_issues:
            for iss in llm_only_issues:
                st.markdown(f"- ⚠️ {iss}")

    with col_b:
        st.markdown("**(B) Programmatic Checks**")
        if prog_issues:
            st.markdown(
                '<span class="badge badge-prog-fail">FAIL</span>',
                unsafe_allow_html=True,
            )
            for iss in prog_issues:
                st.markdown(f"- 🚫 {iss}")
        else:
            st.markdown(
                '<span class="badge badge-prog-pass">PASS</span>',
                unsafe_allow_html=True,
            )
            st.markdown("✓ No programmatic issues found")

    if suggestions:
        with st.expander("💡 Suggestions", expanded=False):
            for s in suggestions:
                st.markdown(f"- {s}")


# ---------------------------------------------------------------------------
# Self-reflection indicator  (Step 2)
# ---------------------------------------------------------------------------

def render_reflection_indicator(iterations: int, feedback_applied: bool) -> None:
    """Show whether self-reflection memory was used in this step."""
    if iterations > 1:
        st.markdown(
            f'<span class="badge badge-reflect">'
            f'🧠 Self-correction: {iterations} iterations</span>',
            unsafe_allow_html=True,
        )
    if feedback_applied:
        st.markdown(
            '<span class="badge badge-reflect">'
            '💾 Past reflections injected</span>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Response splitting — detect and separate code blocks from prose
# ---------------------------------------------------------------------------

# Matches ```lang\n...\n``` fenced blocks (lang is optional)
_CODE_FENCE_RE = re.compile(r"```([\w+-]*)\n(.*?)```", re.DOTALL)

# Part type: ("text", prose) | ("code", lang, code_str)
_Part = Union[Tuple[str, str], Tuple[str, str, str]]


def split_response(text: str) -> List[_Part]:
    """
    Split an LLM response into alternating text and code-block segments.

    Unescapes any HTML entities first (fixes &gt; &lt; &amp; in streamed output).

    Returns a list of tuples:
      ("text", prose_text)
      ("code", language, code_body)
    """
    # Unescape HTML entities that can appear in streamed LLM output
    text = _html.unescape(text)

    parts: List[_Part] = []
    last_end = 0

    for match in _CODE_FENCE_RE.finditer(text):
        start, end = match.span()

        before = text[last_end:start]
        if before.strip():
            parts.append(("text", before))

        lang = match.group(1).strip() or "python"
        code = match.group(2)
        parts.append(("code", lang, code))
        last_end = end

    tail = text[last_end:]
    if tail.strip():
        parts.append(("text", tail))

    if not parts:
        parts.append(("text", text))

    return parts


def render_response(text: str) -> None:
    """
    Render an LLM response with proper code-block formatting.

    - Prose segments → st.markdown()   (native Markdown, no HTML wrapper)
    - Code segments  → st.code()       (syntax-highlighted, copy button)
    """
    for part in split_response(text):
        if part[0] == "text":
            prose = part[1].strip()
            if prose:
                st.markdown(prose)
        else:  # "code"
            _, lang, code = part
            st.code(code.rstrip(), language=lang or "python")


# ---------------------------------------------------------------------------
# Step record card
# ---------------------------------------------------------------------------

def render_step_card(rec: Dict[str, Any], expanded: bool = False) -> None:
    stype    = rec.get("step_type", "execution")
    step_id  = rec.get("step_id", "?")
    task     = rec.get("task", "")
    status   = rec.get("status", "pending")
    dur      = rec.get("duration_s")
    iters    = rec.get("iterations", 0)
    verdict  = rec.get("verdict")
    output   = rec.get("output_full", "") or rec.get("output_preview", "")

    status_dot = _STATUS_COLORS.get(status, "⚪")
    icon       = _AGENT_ICONS.get(stype, "🔹")
    dur_str    = f"  ⏱ {dur:.2f}s" if dur is not None else ""
    iter_str   = f"  🔁 {iters} iter(s)" if iters > 1 else ""

    title = f"{status_dot} {icon} Step {step_id} · **{stype.upper()}**{dur_str}{iter_str}"
    if verdict == "PASS":
        title += "  ✅ PASS"
    elif verdict == "FAIL":
        title += "  ❌ FAIL"

    with st.expander(title, expanded=expanded or status == "failed"):
        st.markdown(f"**Task:** {task}")

        # ── Self-reflection indicators ─────────────────────────────────
        feedback_applied = rec.get("feedback_applied", False)
        render_reflection_indicator(iters, feedback_applied)

        # ── Tool badges ────────────────────────────────────────────────
        render_tool_badges(output, stype)

        # ── Auto-experiment leaderboard ────────────────────────────────
        if stype == "execution" and output:
            lb = _try_parse_leaderboard(output)
            if lb:
                render_auto_experiment_leaderboard(lb)

        # ── Hybrid evaluation panel ────────────────────────────────────
        render_hybrid_eval_panel(rec)

        # ── Output ────────────────────────────────────────────────────
        if output:
            st.markdown('<div class="section-label">Output</div>', unsafe_allow_html=True)
            truncated = output[:4000] + ("\n\n…*(truncated)*" if len(output) > 4000 else "")
            render_response(truncated)

        # ── Error ─────────────────────────────────────────────────────
        if rec.get("error"):
            st.error(f"**Error:** {rec['error']}")


# ---------------------------------------------------------------------------
# Chat message renderer
# ---------------------------------------------------------------------------

def render_route_badge(route: str, from_cache: bool = False) -> None:
    """Show which execution path was taken for this response."""
    key   = "cache" if from_cache else route
    icon, bg, fg = _ROUTE_STYLE.get(key, ("🔄", "#21262d", "#8b949e"))
    label = {
        "direct":   "Direct LLM",
        "rag":      "RAG · memory",
        "tool":     "Tool · web search",
        "pipeline": "Full pipeline",
        "cache":    "Cached",
    }.get(key, key)
    st.markdown(
        f'<span class="badge badge-route" style="background:{bg};color:{fg};">'
        f'{icon} {label}</span>',
        unsafe_allow_html=True,
    )


def render_chat_message(msg: Dict[str, Any]) -> None:
    role         = msg["role"]
    content      = msg["content"]
    step_records = msg.get("step_records", [])
    plan         = msg.get("plan", {})
    duration     = msg.get("duration_s")

    if role == "user":
        st.markdown(f"**You:** {_html.unescape(content)}")
        st.markdown("---")
        return

    # ── Assistant message ────────────────────────────────────────────
    render_route_badge(msg.get("route", "pipeline"), msg.get("from_cache", False))

    # Pipeline details (only when the full pipeline ran)
    if plan and (plan.get("tasks") or plan.get("steps")):
        with st.expander("📋 Task Graph", expanded=False):
            render_task_graph(plan)

    if step_records:
        done   = sum(1 for r in step_records if r.get("status") == "done")
        failed = sum(1 for r in step_records if r.get("status") == "failed")
        label  = f"🔍 Agent Steps ({done}/{len(step_records)} done"
        if failed:
            label += f", {failed} failed"
        label += ")"
        with st.expander(label, expanded=False):
            render_timeline(step_records)
            st.divider()
            for rec in step_records:
                render_step_card(rec)

    render_response(content)
    if duration:
        st.caption(f"⏱ {duration:.1f}s")
    st.markdown("---")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🤖 Multi-Agent AI")
    st.divider()

    if st.button("🔄 New Chat", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages   = []
        st.session_state.last_step_records = []
        st.rerun()

    # ── Document ingestion ────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📚 Add to Knowledge Base")
    doc_text   = st.text_area(
        "Paste document text:",
        height=100,
        placeholder="Paste content to remember across sessions…",
        label_visibility="collapsed",
    )
    doc_source = st.text_input("Source label (optional):", placeholder="e.g. my-docs")
    if st.button("➕ Add Document", use_container_width=True):
        if doc_text.strip():
            with st.spinner("Ingesting…"):
                result = call_ingest(
                    doc_text,
                    {"source": doc_source or "user-upload"},
                    st.session_state.session_id,
                )
            if result:
                st.success(result.get("message", "Added!"))
        else:
            st.warning("Paste some text first.")

    # ── Example queries ───────────────────────────────────────────────
    st.divider()
    st.markdown("#### 💡 Try an example")
    examples = [
        "What is a transformer model?",
        "Latest news on AI regulation",
        "Compare ML models on the wine dataset",
        "Analyse the Iris dataset with Random Forest",
        "Write a Python function to detect outliers using IQR",
    ]
    for ex in examples:
        label = ex[:52] + "…" if len(ex) > 52 else ex
        if st.button(label, use_container_width=True):
            st.session_state["_prefill"] = ex

    # ── API status (minimal) ──────────────────────────────────────────
    st.divider()
    healthy = get_health()
    if healthy:
        st.caption("🟢 API online")
    else:
        st.caption("🔴 API offline — start the backend on port 8000")


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("🤖 Multi-Agent AI")
st.caption("Direct · RAG · Web Search · Full Pipeline — streaming responses")

# Render conversation history
for msg in st.session_state.messages:
    render_chat_message(msg)

# Pre-fill from sidebar example buttons
_prefill = st.session_state.pop("_prefill", "")

# Chat input
user_input = st.chat_input(
    "Ask anything… e.g. 'Compare ML models on the wine dataset'",
) or _prefill

if user_input:
    user_msg: Dict[str, Any] = {"role": "user", "content": user_input}
    st.session_state.messages.append(user_msg)
    render_chat_message(user_msg)

    route_placeholder  = st.empty()
    status_placeholder = st.empty()
    stream_placeholder = st.empty()

    accumulated_text = ""
    detected_route   = "pipeline"

    for chunk, route, status in stream_chat(
        user_input, st.session_state.session_id
    ):
        if route and route != detected_route:
            detected_route = route
            route_placeholder.empty()
            with route_placeholder.container():
                render_route_badge(route, route == "cache")
        if status:
            status_placeholder.info(f"⚙️ {status}")
        if chunk:
            accumulated_text += chunk
            stream_placeholder.markdown(
                f'<div class="chat-ai">{accumulated_text}▌</div>',
                unsafe_allow_html=True,
            )

    # Final render — replace live-cursor placeholder with split text+code blocks
    status_placeholder.empty()
    stream_placeholder.empty()
    render_response(accumulated_text)
    st.markdown("---")

    assistant_msg: Dict[str, Any] = {
        "role":         "assistant",
        "content":      accumulated_text,
        "step_records": [],
        "plan":         {},
        "duration_s":   None,
        "route":        detected_route,
        "from_cache":   detected_route == "cache",
    }
    st.session_state.messages.append(assistant_msg)
    st.session_state.last_step_records = []

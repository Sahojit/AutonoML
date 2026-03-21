# Multi-Agent AI System — Complete Project Reference

## Project Identity

| Field | Value |
|---|---|
| Name | Multi-Agent AI System |
| Version | v2.2.0 |
| Python | 3.11 |
| Root path | `/Users/sahojitkarmakar/Documents/project/LLm Finetuning/Multi-Agent AI System project/multi_agent_ai` |
| Virtualenv | `.venv/` inside root |

---

## How to Run

```bash
# 1. Activate virtualenv
source .venv/bin/activate

# 2. Start backend (FastAPI)
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload

# 3. Start frontend (Streamlit) — separate terminal
streamlit run frontend/app.py --server.port 8501

# 4. Run tests
pytest tests/ -v
```

Backend: `http://localhost:8000`
Frontend: `http://localhost:8501`
API docs: `http://localhost:8000/docs`

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Ollama (`llama3.2:3b`) — default. Also supports vLLM and OpenAI |
| LLM framework | LangChain 0.2.x (`langchain`, `langchain-community`, `langchain-openai`) |
| Embeddings | `BAAI/bge-large-en-v1.5` via HuggingFace (runs on CPU by default) |
| Vector store | ChromaDB (persisted to `./data/chroma_db`) |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit 1.32+ |
| DB (SQL tool) | SQLite via SQLAlchemy (`./data/agent_data.db`) |
| Config | `pydantic-settings` reading from `.env` |

---

## Architecture Overview

```
User Query
    │
    ▼
QueryRouter  ──────────────────────────────────────────┐
    │  (DIRECT / RAG / TOOL / PIPELINE / cache)        │
    │                                                   │
    ├── DIRECT   → single LLM call + history           │
    ├── RAG      → ChromaDB retrieve + LLM             │
    ├── TOOL     → web_search + LLM summary            │
    └── PIPELINE → AgentOrchestrator ──────────────────┘
                        │
                        ▼
                  PlannerAgent
                  (JSON plan: goal + tasks[])
                        │
                        ▼
              Parallel ResearchAgent(s)
              (RAG + web search, asyncio.gather)
                        │
                        ▼
                 ExecutionAgent
                 (ReAct loop: up to MAX_TOOL_ROUNDS=3)
                        │
                        ▼
                EvaluationAgent
                (LLM verdict + programmatic checks)
                        │
               PASS? ───┴─── FAIL? → store reflection
                │                  → retry (up to MAX_AGENT_ITERATIONS=3)
                ▼
           Final Answer
           (synthesised by orchestrator)
                │
                ▼
         FastAPI /chat/stream  →  SSE streaming  →  Streamlit frontend
```

---

## Directory Structure

```
multi_agent_ai/
├── agents/
│   ├── base_agent.py          # AgentMessage envelope + BaseAgent base class
│   ├── planner_agent.py       # Hierarchical planner, JSON output
│   ├── research_agent.py      # RAG + web search
│   ├── execution_agent.py     # ReAct tool-calling loop
│   └── evaluation_agent.py    # Hybrid validator (LLM + programmatic)
├── backend/
│   └── api.py                 # FastAPI app, all endpoints
├── config/
│   └── settings.py            # Pydantic settings, all env vars
├── core/
│   └── query_router.py        # Smart routing + TTL cache + streaming
├── frontend/
│   └── app.py                 # Streamlit UI, split_response, render_response
├── memory/
│   ├── memory_manager.py      # Two-tier memory (deque + ChromaDB)
│   └── vector_store.py        # ChromaDB singleton wrapper
├── models/
│   ├── llm_loader.py          # get_llm(agent_type) with @lru_cache
│   └── embedding_model.py     # HuggingFace embedding singleton
├── orchestrator/
│   ├── agent_orchestrator.py  # Central pipeline controller
│   ├── agent_controller.py    # Backward-compat alias for AgentOrchestrator
│   └── task_manager.py        # StepRecord lifecycle tracking
├── tools/
│   ├── tool_registry.py       # ToolRegistry singleton
│   ├── __init__.py            # Registers all tools at import time
│   ├── python_executor.py     # Sandboxed subprocess Python exec
│   ├── web_search.py          # SerpAPI with DuckDuckGo fallback
│   ├── sql_tool.py            # SQLAlchemy read-only queries (SQLQueryTool)
│   └── dataset_analyzer.py    # sklearn/CSV analysis + auto_experiment
├── tests/
│   ├── test_agents.py
│   ├── test_api.py
│   └── test_memory.py
├── data/                      # chroma_db/, agent_data.db auto-created
├── logs/                      # agent_system.log auto-created
├── docker-compose.yml
├── Dockerfile / Dockerfile.api / Dockerfile.frontend
├── requirements.txt
├── pytest.ini
└── CLAUDE.md                  # ← this file
```

---

## Key Files — What Each Does

### `agents/base_agent.py`
- `AgentMessage` dataclass — canonical envelope: `{agent, task_id, input, output, status, metadata, timestamp}`
- `BaseAgent` — abstract base; all agents inherit this
- `configure_logging()` — called once at startup from `backend/api.py`
- `_success()` / `_fail()` factory helpers for building AgentMessage

### `agents/planner_agent.py`
- `PlanStep` dataclass: `{id, type, task, depends_on[]}`
- `PlannerOutput` dataclass: `{goal, tasks[PlanStep]}` — note: field is `tasks` not `steps`
- `PlannerOutput.to_dict()` returns `{"goal": ..., "tasks": [...]}`
- Parser accepts both `"tasks"` and `"steps"` keys for backward compat
- `tasks_of_type(type)` / `independent_tasks()` helper methods
- Uses `get_llm("planner")` — temperature 0.2

### `agents/research_agent.py`
- RAG retrieval from ChromaDB + optional web_search
- Uses `get_llm("research")` — temperature 0.4
- Returns `AgentMessage` with retrieved context in `output`

### `agents/execution_agent.py`
- ReAct loop up to `MAX_TOOL_ROUNDS=3`
- Dispatches tool calls detected in LLM output by regex:
  - ` ```python ... ``` ` → `python_exec`
  - `WEB_SEARCH: <query>` → `web_search`
  - `SQL_QUERY: <select>` → `sql_query`
  - `DATASET_ANALYZE: sklearn:<name>` → `dataset_analyze`
  - `AUTO_EXPERIMENT: sklearn:<name>` → `auto_experiment`
  - `TOOL_CALL: <name> {json_args}` → generic dispatch
- Injects `past_reflections` from ChromaDB into each prompt
- Uses `get_llm("execution")` — temperature 0.1
- Prompt enforces: always wrap code in fenced blocks with language tag

### `agents/evaluation_agent.py`
- Hybrid evaluation: LLM verdict AND programmatic checks — **both must pass**
- Programmatic checks: NaN in output, traceback present, accuracy=1.0 (leakage), accuracy<0.6, missing train_test_split, no metric reported
- ML sanity checks run as pre-filter before LLM evaluation
- `EvaluationResult`: `{verdict, score, issues, suggestions, llm_verdict, programmatic_issues}`
- Uses `get_llm("evaluation")` — temperature 0.0 (strict, deterministic)

### `orchestrator/agent_orchestrator.py`
- `run(query, session_id)` → returns `ChatResponse`-compatible dict
- Parallel research via `asyncio.gather` on independent tasks (controlled by `PARALLEL_RESEARCH=True`)
- Self-correction loop: up to `MAX_AGENT_ITERATIONS=3`
- On FAIL: calls `memory.store_reflection(task, error, suggestion)` before retry
- `_synthesise()` — final LLM call merging all step outputs into one answer
- `AgentController` is an alias pointing to `AgentOrchestrator` (backward compat)

### `orchestrator/task_manager.py`
- `StepRecord` dataclass tracks step lifecycle
- `set_plan(plan)` iterates `plan.tasks` (not `plan.steps`)
- Statuses: `pending → running → done | failed | skipped`

### `core/query_router.py`
- `RouteType` enum: `DIRECT | RAG | TOOL | PIPELINE`
- `_is_simple_query(q)` — short (≤10 words), no real-time keywords, starts with "what is / explain / define / who is / how does" etc. → always `DIRECT`
- `classify_route(query, has_memory_context, has_conversation)`:
  1. `_is_simple_query` → DIRECT (fires before memory check)
  2. `_REALTIME_KW` match → TOOL
  3. `_PIPELINE_KW` match → PIPELINE
  4. `_FOLLOWUP_KW` match → RAG
  5. both `has_conversation` AND `has_memory_context` → RAG
  6. default → DIRECT
- `ResultCache` — TTL=300s, SHA-256 keyed by `query + recent_history`. Only caches DIRECT and RAG (PIPELINE has side effects)
- `QueryRouter.route()` — non-streaming, returns dict
- `QueryRouter.route_stream()` — streaming generator yielding SSE chunks
- `_run_rag()` — falls back to `_run_direct` if ChromaDB returns 0 docs
- `_run_tool()` — falls back to `_run_direct` if web search fails
- All routes have empty-answer fallback strings
- All prompt builders (`_direct_prompt`, `_rag_prompt`, `_tool_prompt`) inject `_FORMATTING_RULES` (always use fenced code blocks with language tag)

### `models/llm_loader.py`
- `get_llm(agent_type)` with `@lru_cache(maxsize=8)` — one instance per agent type
- `_AGENT_PROFILES`: planner=0.2, research=0.4, execution=0.1, evaluation=0.0, default=`LLM_TEMPERATURE`
- Env-var model overrides: `PLANNER_MODEL`, `RESEARCH_MODEL`, `EXECUTION_MODEL`, `EVALUATION_MODEL`
- Providers: `"ollama"` (default), `"vllm"`, `"openai"`

### `memory/memory_manager.py`
- `Message` dataclass: `{role, content, agent_name, timestamp}`
- `AgentContext` dataclass: mutable pipeline state snapshot
- Short-term: `deque(maxlen=SHORT_TERM_MEMORY_LIMIT)` — in-process, per-session
- Long-term: ChromaDB, persisted to `CHROMA_PERSIST_DIR`
- `add_message(role, content, agent_name)` — adds to short-term deque
- `get_formatted_history(last_n=8)` — returns last N messages as formatted string
- `store_in_long_term(texts, metadatas)` — embeds and stores in ChromaDB
- `retrieve_from_long_term(query, k)` — similarity search, returns `List[Document]`
- `store_reflection(task, error, suggestion)` — stores with `type="reflection"` metadata
- `retrieve_reflections(task, k=3)` — filtered similarity search with `[REFLECTION]` grep fallback
- `save_session_to_long_term()` — called at shutdown, persists conversation to ChromaDB

### `tools/tool_registry.py`
- `ToolRegistry` singleton — `tool_registry` at module level
- `register_tool(name, description, func, parameters)` — registers a callable
- `execute(name, **kwargs)` → `ToolResult`
- `ToolResult`: `{tool_name, success, output, error, duration_s}`
- `ToolResult.format()` — returns human-readable string
- `tools_summary()` — returns formatted list for injection into prompts
- `list_tool_names()` — returns list of registered names

### `tools/__init__.py`
- Registers all 5 tools at import time: `python_exec`, `web_search`, `sql_query`, `dataset_analyze`, `auto_experiment`
- Import this package early; all agents import from here

### `tools/dataset_analyzer.py`
- `analyze(source, dataset_name, max_rows)` — source: `"sklearn:<name>"` or `"csv:<filepath>"`
- `auto_experiment(source, target_column, test_size)` — trains RF, LR, GBT, optional XGBoost; returns ranked leaderboard + best_model dict

### `backend/api.py`
- `POST /chat` — full pipeline via `AgentOrchestrator.run()`
- `POST /chat/smart` — adaptive routing via `QueryRouter.route()`
- `POST /chat/stream` — SSE streaming via `QueryRouter.route_stream()`, bridges sync generator to async via `threading.Thread` + `queue.Queue`
- `POST /memory` — ingest document into ChromaDB
- `GET /history?session_id=` — conversation history
- `GET /health` — liveness probe
- `GET /status` — system info (model, provider, doc count, etc.)
- `DELETE /session/{session_id}` — remove session
- `ChatResponse` fields: `final_answer, session_id, goal, plan (dict), step_records, duration_s, route, from_cache`
- Sessions stored in `_sessions: Dict[str, MemoryManager]` (in-process; swap with Redis for multi-worker)

### `frontend/app.py`
- Dark theme CSS (GitHub dark palette: `#0e1117` bg, `#e6edf3` text)
- `split_response(text)` — splits LLM response into `("text", prose)` and `("code", lang, code)` tuples; applies `html.unescape()` first
- `render_response(text)` — iterates parts: `st.markdown(prose)` for text, `st.code(code, language=lang)` for code — **no HTML div wrapping**
- `render_route_badge(route, from_cache)` — shows which execution path was taken
- `render_chat_message(msg)` — user messages as `st.markdown("**You:** ...")`, AI messages via `render_response()`
- `stream_chat(query, session_id)` — generator yielding `(chunk, route, status)` from SSE stream
- `call_chat_smart(query, session_id)` — POST `/chat/smart`
- Sidebar: New Chat button, document ingestion, example queries, API status indicator
- Streaming: live cursor `▌` in placeholder, replaced by `render_response()` on completion

---

## Configuration & Environment Variables

All settings in `config/settings.py` read from `.env` file or env:

```env
# LLM
LLM_PROVIDER=ollama          # ollama | vllm | openai
LLM_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434
OPENAI_API_KEY=              # required if LLM_PROVIDER=openai

# Per-agent model overrides (optional — all default to LLM_MODEL)
PLANNER_MODEL=
RESEARCH_MODEL=
EXECUTION_MODEL=
EVALUATION_MODEL=

# Embeddings
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DEVICE=cpu         # cpu | cuda | mps

# Vector store
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION_NAME=knowledge_base

# Memory
SHORT_TERM_MEMORY_LIMIT=20
LONG_TERM_MEMORY_TOP_K=5

# Tools
SERPAPI_KEY=                 # optional; falls back to DuckDuckGo
PYTHON_EXEC_TIMEOUT=30
SQL_DATABASE_URL=sqlite:///./data/agent_data.db

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
CORS_ORIGINS=["*"]

# Frontend
FRONTEND_API_URL=http://localhost:8000

# Agent behaviour
MAX_AGENT_ITERATIONS=3       # self-correction retry limit
MAX_TOOL_ROUNDS=3            # ReAct rounds per execution step
PLAN_MAX_STEPS=6             # max tasks per plan
PARALLEL_RESEARCH=true       # run independent research concurrently

# Logging
LOG_LEVEL=INFO
```

---

## Critical Design Decisions

1. **Planner output key is `tasks` not `steps`** — `PlannerOutput.tasks`, `plan.tasks`, `to_dict()["tasks"]`. Parser accepts both for backward compat.

2. **EvaluationAgent has no RETRY verdict** — only PASS/FAIL. The orchestrator owns the retry loop externally.

3. **Tools registered at import time** — importing `tools` package triggers `_register_all()`. Do this before any agent runs.

4. **`SQLQueryTool` not `SQLTool`** — renamed during refactor. `get_sql_tool()` returns it.

5. **`AgentController` is an alias for `AgentOrchestrator`** — defined at bottom of `agent_orchestrator.py` for backward compat.

6. **RAG fallback** — `_run_rag` falls back to `_run_direct` when ChromaDB returns 0 docs.

7. **Tool fallback** — `_run_tool` falls back to `_run_direct` when web search fails.

8. **Cache never stores PIPELINE results** — pipeline writes to memory (side effects). Only DIRECT and RAG are cached.

9. **`_is_simple_query` fires before memory check** — prevents short factual queries like "what is javascript" from hitting RAG just because ChromaDB has documents.

10. **RAG only triggers when BOTH `has_conversation` AND `has_memory_context` are true** — prevents spurious RAG on fresh sessions.

11. **`render_response()` never wraps prose in HTML divs** — uses plain `st.markdown(prose)` so Markdown inside LLM responses renders correctly (bullets, bold, headers, etc.).

12. **`html.unescape()` applied inside `split_response`** — fixes `&gt;` `&lt;` `&amp;` that appear in streamed SSE output.

---

## Data Flow — ChatResponse Dict

All execution paths return a dict that maps to `ChatResponse`:

```python
{
    "final_answer": str,       # the answer text
    "session_id":   str,       # UUID
    "goal":         str,       # user query or plan goal
    "plan":         dict,      # {"goal": ..., "tasks": [...]} or {}
    "step_records": list,      # list of step dicts
    "duration_s":   float,     # wall time
    "route":        str,       # "direct" | "rag" | "tool" | "pipeline" | "cache"
    "from_cache":   bool,
}
```

Each step record in `step_records`:
```python
{
    "step_id":        int,
    "step_type":      str,    # "research" | "execution" | "evaluation"
    "task":           str,
    "status":         str,    # "done" | "failed" | "running" | "pending"
    "output_full":    str,
    "duration_s":     float,
    "verdict":        str,    # "PASS" | "FAIL" (evaluation steps only)
    "eval_score":     float,
    "iterations":     int,
    "feedback_applied": bool,
}
```

---

## SSE Streaming Protocol

`POST /chat/stream` yields Server-Sent Events:

```
data: __ROUTE__:direct\n\n       ← first chunk, route taken
data: __STATUS__:Running...\n\n  ← pipeline status updates
data: Hello, here is...\n\n      ← answer tokens
data: more text...\n\n
```

Frontend `stream_chat()` strips `data: ` prefix and yields `(chunk, route, status)` tuples.

---

## Self-Reflection Memory

When a pipeline step FAILS evaluation:

```python
memory.store_reflection(
    task="...",
    error="...",   # evaluation issues
    suggestion="..." # from evaluation suggestions
)
```

On next run of a similar task, `ExecutionAgent` retrieves:
```python
reflections = memory.retrieve_reflections(instruction, k=3)
```
And injects them under `⚠ Previous mistakes to avoid` in the prompt.

Reflections stored in ChromaDB with `metadata={"type": "reflection"}`.

---

## Tool Call Syntax (in LLM output)

The `ExecutionAgent` detects these patterns via regex:

| Pattern | Tool called |
|---|---|
| ` ```python\n...\n``` ` | `python_exec` |
| `WEB_SEARCH: <query>` | `web_search` |
| `SQL_QUERY: <SELECT ...>` | `sql_query` |
| `DATASET_ANALYZE: sklearn:<name>` | `dataset_analyze` |
| `AUTO_EXPERIMENT: sklearn:<name>` | `auto_experiment` |
| `TOOL_CALL: <name> {json_args}` | generic registry dispatch |

---

## Agent Temperature Profiles

| Agent | Temperature | Reason |
|---|---|---|
| planner | 0.2 | Structured, deterministic JSON planning |
| research | 0.4 | Creative synthesis of retrieved context |
| execution | 0.1 | Precise, literal code generation |
| evaluation | 0.0 | Strict, reproducible judging |

---

## Logging

- Logger names: `multiagent.api`, `multiagent.orchestrator`, `multiagent.agent.<name>`, `multiagent.memory`, `multiagent.core.router`, `multiagent.tools`
- Output: stdout + `./logs/agent_system.log`
- Format: `%(asctime)s | %(levelname)-8s | %(name)s | %(message)s`

---

## Docker

```bash
# Full stack
docker-compose up --build

# Services defined in docker-compose.yml:
# - api       → Dockerfile.api    → port 8000
# - frontend  → Dockerfile.frontend → port 8501
```

---

## Tests

```bash
pytest tests/ -v                  # all tests
pytest tests/test_agents.py -v    # agent unit tests
pytest tests/test_api.py -v       # API integration tests
pytest tests/test_memory.py -v    # memory unit tests
```

Config in `pytest.ini`.

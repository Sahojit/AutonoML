# Changelog

## [v2.3.0] - 2026-04-12

### Added
- `FileReaderTool` (`file_read`): reads .txt .md .csv .json .py .log .pdf with chunking support
- Pluggable `SessionStore`: `InMemorySessionStore` (default) + `RedisSessionStore` for multi-worker persistence
- LangSmith tracing hook: set `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` to enable
- `AgentController`: refactored to current `PlannerOutput` and `TaskManager` API, exported from `orchestrator/__init__.py`
- Rate limiting on `/chat`, `/chat/smart`, `/chat/stream` via slowapi (10 req/min per IP)
- GitHub Actions CI workflow (lint + test on push/PR, Python 3.10 & 3.11)
- `CONTRIBUTING.md`, GitHub issue/PR templates, `pyproject.toml` with ruff config
- `REDIS_URL`, `SESSION_TTL_SECONDS`, `LANGCHAIN_*` settings with `.env.example` entries

### Changed
- `/status` endpoint now includes tracing status fields
- `api.py` session management fully delegated to `SessionStore` abstraction

### Tests
- 386 unit tests total (up from 287)
- New: LLM loader (17), tracing (16), session store (25), AgentController (16), file reader (25), orchestrator (39), dataset analyzer (39)

## [v2.2.0] - 2026-03-22

### Added
- Hierarchical PlannerAgent with DAG-based task decomposition
- AgentOrchestrator with parallel asyncio research execution
- ExecutionAgent with ToolRegistry-based ReAct loop (up to 3 tool rounds)
- EvaluationAgent with 5-criteria scoring (logical correctness, completeness, tool use, code validity, statistical validity)
- Self-correction loop: Execution → Evaluation → feedback → retry (up to MAX_AGENT_ITERATIONS)
- Two-tier MemoryManager: short-term deque + ChromaDB long-term with BGE embeddings
- ToolRegistry singleton with 4 registered tools: python_exec, web_search, dataset_analyze, sql_query
- DatasetAnalyzer: sklearn built-in datasets + CSV source + auto_experiment leaderboard
- SQLQueryTool: read-only SQLAlchemy queries with injection guard
- PythonExecutor: sandboxed subprocess execution with timeout
- FastAPI backend with /health, /status, /chat, /ingest, /history endpoints
- Streamlit dashboard with task graph, execution timeline, tool badges, eval scores
- 287 unit tests across all agents, tools, orchestrator, router, and API

### Changed
- Replaced deprecated `HuggingFaceBgeEmbeddings` with `HuggingFaceEmbeddings` (langchain-huggingface)
- Replaced deprecated `Chroma` from langchain-community with langchain-chroma
- QueryRouter with ResultCache and classify_route pipeline

### Fixed
- test_api.py patching strategy: patches `_get_orchestrator` directly instead of AgentController

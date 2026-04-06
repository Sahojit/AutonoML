# Changelog

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

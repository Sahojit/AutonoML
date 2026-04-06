# Multi-Agent AI System v2.2.0

![CI](https://github.com/Sahojit/AutonoML/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
![Tests](https://img.shields.io/badge/tests-287%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

A production-grade autonomous multi-agent AI platform built with Python, Ollama (Llama 3), ChromaDB, FastAPI, and Streamlit.

---

## Architecture

```
User Query
    │
    ▼
PlannerAgent          ← Hierarchical task decomposition (DAG)
    │
    ▼
AgentOrchestrator     ← Central controller
    │
    ├──► ResearchAgent(s)   ← Parallel asyncio execution
    │         │
    │         ▼
    ├──► ExecutionAgent     ← Tool-calling ReAct loop
    │         │
    │         ▼
    └──► EvaluationAgent    ← Deep multi-criteria scoring
              │
         FAIL? ──► feedback ──► ExecutionAgent (retry)
              │
         PASS or max_iterations reached
              │
              ▼
         MemoryManager      ← Persist to ChromaDB
              │
              ▼
         Final Answer
```

---

## Features

| # | Feature | Detail |
|---|---------|--------|
| 1 | **Hierarchical Planner** | Decomposes queries into typed tasks (research / execution / evaluation) with a DAG dependency graph |
| 2 | **Structured Communication** | All agents exchange `AgentMessage` JSON envelopes with agent, task_id, input, output, status |
| 3 | **Tool Registry** | Central `ToolRegistry` — tools self-register; ExecutionAgent selects and calls tools dynamically |
| 4 | **True Code Execution** | Generates Python → runs via `PythonExecutor` → captures stdout/stderr → feeds result back to LLM |
| 5 | **Self-Correction Loop** | Execution → Evaluation → feedback → retry, up to `MAX_AGENT_ITERATIONS` (default 3) |
| 6 | **Parallel Research** | Independent research tasks run concurrently via `asyncio.gather` |
| 7 | **Vector Memory** | Two-tier memory: short-term deque + ChromaDB long-term with BGE embeddings |
| 8 | **Improved Evaluation** | 5-criteria scoring (logical correctness, completeness, tool use, code validity, statistical validity) |
| 9 | **Observability** | Structured logging to `logs/api.log`; per-step timing, iteration counts, eval scores |
| 10 | **Dashboard** | Streamlit UI with task graph, execution timeline, tool badges, eval scores, retry indicators |

---

## Planner Output Format

```json
{
  "goal": "Train a Random Forest on the Iris dataset and evaluate accuracy",
  "tasks": [
    {"id": 1, "type": "research",   "task": "Retrieve Iris dataset details and feature descriptions", "depends_on": []},
    {"id": 2, "type": "research",   "task": "Research Random Forest hyperparameters and best practices", "depends_on": []},
    {"id": 3, "type": "execution",  "task": "Load Iris dataset, train Random Forest, compute accuracy", "depends_on": [1, 2]},
    {"id": 4, "type": "evaluation", "task": "Validate model accuracy and code correctness",             "depends_on": [3]}
  ]
}
```

Tasks 1 and 2 have no dependencies — run in **parallel**.
Task 3 waits for both — sequential.
Task 4 waits for task 3 — sequential.

---

## Tool Registry

| Tool | Name | Description |
|------|------|-------------|
| `PythonExecutor` | `python_exec` | Sandboxed subprocess execution, captures stdout/stderr |
| `WebSearchTool` | `web_search` | SerpAPI with DuckDuckGo fallback |
| `DatasetAnalyzer` | `dataset_analyze` | Loads sklearn/CSV datasets, returns shape, dtypes, statistics |
| `SQLQueryTool` | `sql_query` | Read-only SQLAlchemy queries |

All tools return:
```json
{"tool": "...", "status": "success|fail", "output": "...", "error": null}
```

---

## Agent Message Format

```json
{
  "agent": "planner | research | execution | evaluation",
  "task_id": "plan_a1b2c3",
  "input": "...",
  "output": "...",
  "status": "success | fail",
  "metadata": {}
}
```

---

## Evaluation Output Format

```json
{
  "verdict": "PASS",
  "score": 0.87,
  "issues": [],
  "suggestions": ["Consider cross-validation for a more robust accuracy estimate"]
}
```

Criteria: logical correctness · task completeness · tool usage · code validity · statistical validity.

---

## Self-Correction Loop

```python
for iteration in range(1, MAX_AGENT_ITERATIONS + 1):
    exec_result = execution_agent.run(task, feedback=previous_feedback)
    eval_result = evaluation_agent.evaluate(exec_result)

    if eval_result.verdict == "PASS":
        break

    previous_feedback = eval_result.feedback_text()  # issues + suggestions
```

---

## Project Structure

```
multi_agent_ai/
├── agents/
│   ├── base_agent.py          # AgentMessage + BaseAgent
│   ├── planner_agent.py       # Hierarchical planner, DAG output
│   ├── research_agent.py      # RAG + web search
│   ├── execution_agent.py     # Tool-calling ReAct loop
│   └── evaluation_agent.py    # Deep 5-criteria validator
├── orchestrator/
│   ├── agent_orchestrator.py  # Central pipeline controller
│   └── task_manager.py        # Per-step lifecycle tracking
├── tools/
│   ├── tool_registry.py       # ToolRegistry singleton
│   ├── python_executor.py     # Sandboxed code runner
│   ├── web_search.py          # SerpAPI / DuckDuckGo
│   ├── dataset_analyzer.py    # sklearn + CSV analysis
│   └── sql_tool.py            # Read-only SQL queries
├── memory/
│   ├── memory_manager.py      # Two-tier memory (deque + Chroma)
│   └── vector_store.py        # ChromaDB wrapper
├── models/
│   ├── llm_loader.py          # Ollama LLM loader
│   └── embedding_model.py     # BGE embedding model
├── backend/
│   └── api.py                 # FastAPI endpoints
├── frontend/
│   └── app.py                 # Streamlit dashboard
├── config/
│   └── settings.py            # Pydantic settings
├── logs/                      # Structured log files
├── data/
│   └── chroma_db/             # Persistent vector store
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) running locally with `llama3.2:3b`

```bash
ollama pull llama3.2:3b
```

### Install

```bash
cd multi_agent_ai
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run

```bash
# Terminal 1 — Backend
uvicorn backend.api:app --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend
streamlit run frontend/app.py --server.port 8501
```

Open **http://localhost:8501**.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/status` | System status (model, memory, tools) |
| `POST` | `/chat` | Submit query, get full pipeline response |
| `POST` | `/ingest` | Add documents to knowledge base |
| `GET` | `/history/{session_id}` | Conversation history |

### Example

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Train a Random Forest on the Iris dataset"}'
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2:3b` | LLM model |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Embedding model |
| `MAX_AGENT_ITERATIONS` | `3` | Self-correction retry limit |
| `MAX_TOOL_ROUNDS` | `3` | Max tool calls per execution step |
| `PARALLEL_RESEARCH` | `true` | Run independent research tasks in parallel |
| `PLAN_MAX_STEPS` | `6` | Max tasks per plan |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Vector store path |
| `SERPAPI_KEY` | _(optional)_ | SerpAPI key for web search |

---

## Docker

```bash
docker build -t multi-agent-ai .
docker run -p 8000:8000 -p 8501:8501 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  multi-agent-ai
```

---

## Example Queries

```
# ML / Data Science
Analyse the Iris dataset and train a Random Forest classifier
Compare accuracy of KNN vs Decision Tree on the wine dataset

# Code Execution
Write a Python function to detect outliers using IQR and test it
Generate Fibonacci numbers up to 1000 and find all primes in the result

# Research + Execution
Research gradient boosting, then implement XGBoost on the breast cancer dataset
Explain LSTM networks and write Python code to build one with Keras

# SQL
Create a SQLite table of employees and query the top 5 salaries by department
```

---

## Stack

| Component | Technology |
|-----------|-----------|
| LLM | Ollama + `llama3.2:3b` |
| Orchestration | LangChain 0.2.x |
| Vector DB | ChromaDB |
| Embeddings | `BAAI/bge-large-en-v1.5` |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Data | SQLAlchemy, pandas, scikit-learn |
| Python | 3.9+ |
# Multi-Agent AI System

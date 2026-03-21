"""
FastAPI endpoint tests using TestClient and mocked AgentController.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

MOCK_CHAT_RESULT = {
    "final_answer": "Here is your analysis result.",
    "session_id": "test-session-123",
    "plan_summary": "Step 1 [RESEARCH]: Find docs\nStep 2 [EXECUTION]: Analyse",
    "step_records": [
        {
            "step_number": 1,
            "agent": "research",
            "instruction": "Find docs",
            "status": "done",
            "result_preview": "Found 3 documents",
            "error": None,
            "retries": 0,
            "duration_s": 1.2,
        }
    ],
    "duration_s": 5.3,
}


@pytest.fixture
def client():
    # Patch heavy dependencies before importing the app
    with patch("models.embedding_model.get_embedding_model"), \
         patch("memory.vector_store.chromadb.PersistentClient"), \
         patch("orchestrator.agent_controller.AgentController") as mock_ctrl_cls:

        mock_ctrl = MagicMock()
        mock_ctrl.run.return_value = MOCK_CHAT_RESULT
        mock_ctrl.get_history.return_value = [
            {"role": "user", "content": "Hello", "timestamp": 0.0}
        ]
        mock_ctrl.add_document.return_value = ["doc-id-1"]
        mock_ctrl_cls.return_value = mock_ctrl

        from backend.api import app
        yield TestClient(app, raise_server_exceptions=True), mock_ctrl


# ──────────────────────────────────────────────────────────────────────────────
# Health & status
# ──────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoints:

    def test_health(self, client):
        tc, _ = client
        resp = tc.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_status(self, client):
        tc, _ = client
        resp = tc.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "llm_model" in data
        assert "active_sessions" in data


# ──────────────────────────────────────────────────────────────────────────────
# /chat
# ──────────────────────────────────────────────────────────────────────────────

class TestChatEndpoint:

    def test_chat_returns_answer(self, client):
        tc, mock_ctrl = client
        resp = tc.post("/chat", json={"query": "Analyse the iris dataset"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["final_answer"] == "Here is your analysis result."
        assert "step_records" in data

    def test_chat_creates_session(self, client):
        tc, _ = client
        resp = tc.post("/chat", json={"query": "Hello"})
        assert resp.status_code == 200
        assert resp.json()["session_id"] is not None

    def test_chat_with_provided_session_id(self, client):
        tc, _ = client
        resp = tc.post("/chat", json={"query": "Hello", "session_id": "my-session"})
        assert resp.status_code == 200

    def test_chat_empty_query_rejected(self, client):
        tc, _ = client
        resp = tc.post("/chat", json={"query": ""})
        assert resp.status_code == 422  # Pydantic validation error


# ──────────────────────────────────────────────────────────────────────────────
# /memory
# ──────────────────────────────────────────────────────────────────────────────

class TestMemoryEndpoint:

    def test_ingest_document(self, client):
        tc, mock_ctrl = client
        resp = tc.post(
            "/memory",
            json={"text": "Some knowledge content", "metadata": {"source": "test"}},
        )
        assert resp.status_code == 200
        assert "doc_ids" in resp.json()

    def test_ingest_empty_text_rejected(self, client):
        tc, _ = client
        resp = tc.post("/memory", json={"text": ""})
        assert resp.status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# /history
# ──────────────────────────────────────────────────────────────────────────────

class TestHistoryEndpoint:

    def test_history_not_found(self, client):
        tc, _ = client
        resp = tc.get("/history?session_id=nonexistent-session")
        assert resp.status_code == 404

    def test_history_after_chat(self, client):
        tc, mock_ctrl = client
        # First create the session via /chat
        tc.post("/chat", json={"query": "Hello", "session_id": "hist-session"})
        resp = tc.get("/history?session_id=hist-session")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "hist-session"
        assert "messages" in data

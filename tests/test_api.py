import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


MOCK_CHAT_RESULT = {
    "final_answer": "Here is your analysis result.",
    "session_id": "test-session-123",
    "goal": "Analyse the iris dataset",
    "plan": {"goal": "Analyse the iris dataset", "tasks": []},
    "step_records": [
        {
            "step_id": 1,
            "step_type": "research",
            "task": "Find docs",
            "status": "done",
            "output_full": "Found 3 documents",
            "duration_s": 1.2,
            "verdict": "",
            "eval_score": 0.0,
            "iterations": 1,
            "feedback_applied": False,
        }
    ],
    "duration_s": 5.3,
    "route": "pipeline",
    "from_cache": False,
}


@pytest.fixture
def client():
    with patch("models.embedding_model.get_embedding_model"), \
         patch("memory.vector_store.chromadb.PersistentClient"), \
         patch("backend.api._get_orchestrator") as mock_get_orch:

        mock_orch = MagicMock()
        mock_orch.run.return_value = MOCK_CHAT_RESULT
        mock_get_orch.return_value = mock_orch

        from backend.api import app
        yield TestClient(app, raise_server_exceptions=True), mock_orch


class TestHealthEndpoints:

    def test_health_returns_200(self, client):
        tc, _ = client
        resp = tc.get("/health")
        assert resp.status_code == 200

    def test_health_status_ok(self, client):
        tc, _ = client
        resp = tc.get("/health")
        assert resp.json()["status"] == "ok"

    def test_status_returns_200(self, client):
        tc, _ = client
        resp = tc.get("/status")
        assert resp.status_code == 200

    def test_status_has_llm_model(self, client):
        tc, _ = client
        data = tc.get("/status").json()
        assert "llm_model" in data

    def test_status_has_active_sessions(self, client):
        tc, _ = client
        data = tc.get("/status").json()
        assert "active_sessions" in data


class TestChatEndpoint:

    def test_chat_returns_200(self, client):
        tc, _ = client
        resp = tc.post("/chat", json={"query": "Analyse the iris dataset"})
        assert resp.status_code == 200

    def test_chat_final_answer_present(self, client):
        tc, mock_orch = client
        resp = tc.post("/chat", json={"query": "Analyse the iris dataset"})
        assert resp.json()["final_answer"] == "Here is your analysis result."

    def test_chat_step_records_present(self, client):
        tc, _ = client
        resp = tc.post("/chat", json={"query": "Analyse the iris dataset"})
        assert "step_records" in resp.json()

    def test_chat_session_id_returned(self, client):
        tc, _ = client
        resp = tc.post("/chat", json={"query": "Hello"})
        assert resp.json()["session_id"] is not None

    def test_chat_with_explicit_session_id(self, client):
        tc, _ = client
        resp = tc.post("/chat", json={"query": "Hello", "session_id": "my-sess"})
        assert resp.status_code == 200

    def test_chat_empty_query_rejected(self, client):
        tc, _ = client
        resp = tc.post("/chat", json={"query": ""})
        assert resp.status_code == 422

    def test_chat_missing_query_rejected(self, client):
        tc, _ = client
        resp = tc.post("/chat", json={})
        assert resp.status_code == 422

    def test_chat_orchestrator_called_with_query(self, client):
        tc, mock_orch = client
        tc.post("/chat", json={"query": "Train a classifier"})
        mock_orch.run.assert_called_once()
        call_args = mock_orch.run.call_args
        assert "Train a classifier" in call_args[0]


class TestMemoryEndpoint:

    def test_ingest_returns_200(self, client):
        tc, _ = client
        resp = tc.post("/memory", json={"text": "Some knowledge content"})
        assert resp.status_code == 200

    def test_ingest_returns_doc_ids(self, client):
        tc, _ = client
        resp = tc.post("/memory", json={"text": "Some knowledge content"})
        assert "doc_ids" in resp.json()

    def test_ingest_empty_text_rejected(self, client):
        tc, _ = client
        resp = tc.post("/memory", json={"text": ""})
        assert resp.status_code == 422

    def test_ingest_with_metadata(self, client):
        tc, _ = client
        resp = tc.post(
            "/memory",
            json={"text": "Knowledge", "metadata": {"source": "test_doc"}},
        )
        assert resp.status_code == 200


class TestHistoryEndpoint:

    def test_history_unknown_session_returns_404(self, client):
        tc, _ = client
        resp = tc.get("/history?session_id=no-such-session")
        assert resp.status_code == 404

    def test_history_after_chat_returns_200(self, client):
        tc, _ = client
        tc.post("/chat", json={"query": "Hello", "session_id": "hist-sess"})
        resp = tc.get("/history?session_id=hist-sess")
        assert resp.status_code == 200

    def test_history_response_has_session_id(self, client):
        tc, _ = client
        tc.post("/chat", json={"query": "Hello", "session_id": "hist-sess-2"})
        data = tc.get("/history?session_id=hist-sess-2").json()
        assert data["session_id"] == "hist-sess-2"

    def test_history_response_has_messages(self, client):
        tc, _ = client
        tc.post("/chat", json={"query": "Hello", "session_id": "hist-sess-3"})
        data = tc.get("/history?session_id=hist-sess-3").json()
        assert "messages" in data

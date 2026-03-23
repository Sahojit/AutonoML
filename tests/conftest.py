import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock
from langchain.docstore.document import Document
import pytest

from memory.memory_manager import AgentContext, MemoryManager


@pytest.fixture
def mock_memory():
    mem = MagicMock(spec=MemoryManager)
    mem.get_formatted_history.return_value = ""
    mem.get_conversation_history.return_value = []
    mem.add_message = MagicMock()
    mem.update_context = MagicMock()
    mem.store_in_long_term = MagicMock(return_value=["id-1"])
    mem.store_reflection = MagicMock()
    mem.retrieve_reflections.return_value = []
    mem.retrieve_from_long_term.return_value = []
    mem.save_session_to_long_term = MagicMock()
    ctx = AgentContext(user_query="test query")
    mem.get_context.return_value = ctx
    return mem


@pytest.fixture
def mock_memory_with_docs():
    mem = MagicMock(spec=MemoryManager)
    mem.get_formatted_history.return_value = "User: hello\nAssistant: hi"
    mem.get_conversation_history.return_value = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    mem.add_message = MagicMock()
    mem.update_context = MagicMock()
    mem.store_in_long_term = MagicMock(return_value=["id-1", "id-2"])
    mem.store_reflection = MagicMock()
    mem.retrieve_reflections.return_value = []
    mem.retrieve_from_long_term.return_value = [
        Document(page_content="Python is a high-level language.", metadata={"source": "test"}),
        Document(page_content="It was created by Guido van Rossum.", metadata={"source": "test"}),
    ]
    mem.save_session_to_long_term = MagicMock()
    ctx = AgentContext(user_query="what is python")
    mem.get_context.return_value = ctx
    return mem


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="Mock LLM response.")
    llm.stream.return_value = iter([
        MagicMock(content="Mock "),
        MagicMock(content="streamed "),
        MagicMock(content="response."),
    ])
    return llm


@pytest.fixture
def mock_llm_with_json():
    import json
    payload = json.dumps({
        "verdict": "PASS",
        "score": 0.88,
        "issues": [],
        "suggestions": [],
    })
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=payload)
    return llm


@pytest.fixture
def mock_chain(mock_llm):
    chain = MagicMock()
    chain.invoke.return_value = mock_llm.invoke.return_value
    return chain

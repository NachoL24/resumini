"""Tests for the FastAPI app entry point."""

from typing import Any
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from resumini.agent.graph import build_graph
from resumini.main import create_app
from resumini.vault.manager import VaultManager


class _FakeChatLLM(BaseChatModel):
    responses: list = Field(default_factory=list)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = (
            self.responses.pop(0) if self.responses else AIMessage(content="ok")
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "fake"


def _make_test_app(tmp_vault, response_content: str = "hola estudiante"):
    vault = VaultManager(tmp_vault)
    chroma = MagicMock()
    fake = _FakeChatLLM(responses=[AIMessage(content=response_content)])
    with patch("resumini.agent.graph.get_llm", return_value=fake):
        graph = build_graph(vault, chroma)
    return create_app(vault=vault, chroma=chroma, graph=graph)


def test_chat_endpoint_returns_assistant_output(tmp_vault):
    app = _make_test_app(tmp_vault, "hola estudiante")
    client = TestClient(app)
    response = client.post(
        "/chat", json={"message": "hola", "session_id": "t1"}
    )
    assert response.status_code == 200
    body = response.json()
    assert "output" in body
    assert "hola estudiante" in body["output"]


def test_chat_endpoint_accepts_optional_materia(tmp_vault):
    app = _make_test_app(tmp_vault)
    client = TestClient(app)
    response = client.post(
        "/chat",
        json={"message": "hola", "session_id": "t1", "materia": "civil"},
    )
    assert response.status_code == 200


def test_chat_endpoint_defaults_session_id_when_absent(tmp_vault):
    app = _make_test_app(tmp_vault)
    client = TestClient(app)
    response = client.post("/chat", json={"message": "hola"})
    assert response.status_code == 200


def test_chat_endpoint_isolates_sessions(tmp_vault):
    vault = VaultManager(tmp_vault)
    chroma = MagicMock()
    fake = _FakeChatLLM()
    with patch("resumini.agent.graph.get_llm", return_value=fake):
        graph = build_graph(vault, chroma)
    app = create_app(vault=vault, chroma=chroma, graph=graph)
    client = TestClient(app)
    client.post("/chat", json={"message": "mensaje_alpha", "session_id": "a"})
    client.post("/chat", json={"message": "mensaje_beta", "session_id": "b"})

    state_a = graph.get_state({"configurable": {"thread_id": "a"}})
    state_b = graph.get_state({"configurable": {"thread_id": "b"}})
    contents_a = [m.content for m in state_a.values["messages"]]
    contents_b = [m.content for m in state_b.values["messages"]]
    assert any("mensaje_alpha" in c for c in contents_a)
    assert not any("mensaje_alpha" in c for c in contents_b)
    assert any("mensaje_beta" in c for c in contents_b)

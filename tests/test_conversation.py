"""Tests for the conversational agent flow (multi-turn, memory, tool calling)."""

from typing import Any
from unittest.mock import MagicMock, patch

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from resumini.agent.graph import build_graph
from resumini.vault.manager import VaultManager


class _FakeChatLLM(BaseChatModel):
    """Fake chat model that records prompts and returns scripted AI responses."""

    responses: list = Field(default_factory=list)
    captured: list = Field(default_factory=list)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.captured.append(list(messages))
        if self.responses:
            msg = self.responses.pop(0)
        else:
            msg = AIMessage(content="ok")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "fake-chat-llm"


def test_build_graph_returns_invokable_object(tmp_vault):
    vault = VaultManager(tmp_vault)
    chroma = MagicMock()
    with patch("resumini.agent.graph.get_llm", return_value=_FakeChatLLM()):
        graph = build_graph(vault, chroma)
    assert graph is not None
    assert hasattr(graph, "invoke")


def test_invoke_returns_messages_state(tmp_vault):
    vault = VaultManager(tmp_vault)
    chroma = MagicMock()
    fake = _FakeChatLLM(responses=[AIMessage(content="respuesta de prueba")])
    with patch("resumini.agent.graph.get_llm", return_value=fake):
        graph = build_graph(vault, chroma)
        result = graph.invoke(
            {"messages": [HumanMessage(content="hola")]},
            config={"configurable": {"thread_id": "smoke"}},
        )
    assert "messages" in result
    contents = [m.content for m in result["messages"]]
    assert any("hola" in c for c in contents)
    assert any("respuesta de prueba" in c for c in contents)


def test_same_thread_persists_messages_across_invokes(tmp_vault):
    vault = VaultManager(tmp_vault)
    chroma = MagicMock()
    with patch("resumini.agent.graph.get_llm", return_value=_FakeChatLLM()):
        graph = build_graph(vault, chroma)
        cfg = {"configurable": {"thread_id": "t1"}}
        graph.invoke({"messages": [HumanMessage(content="primero")]}, config=cfg)
        graph.invoke({"messages": [HumanMessage(content="segundo")]}, config=cfg)
        state = graph.get_state(cfg)
        contents = [m.content for m in state.values["messages"]]
    assert any("primero" in c for c in contents)
    assert any("segundo" in c for c in contents)


def test_different_threads_are_isolated(tmp_vault):
    vault = VaultManager(tmp_vault)
    chroma = MagicMock()
    with patch("resumini.agent.graph.get_llm", return_value=_FakeChatLLM()):
        graph = build_graph(vault, chroma)
        graph.invoke(
            {"messages": [HumanMessage(content="mensaje en t1")]},
            config={"configurable": {"thread_id": "t1"}},
        )
        graph.invoke(
            {"messages": [HumanMessage(content="mensaje en t2")]},
            config={"configurable": {"thread_id": "t2"}},
        )
        state_t1 = graph.get_state({"configurable": {"thread_id": "t1"}})
        state_t2 = graph.get_state({"configurable": {"thread_id": "t2"}})
        contents_t1 = [m.content for m in state_t1.values["messages"]]
        contents_t2 = [m.content for m in state_t2.values["messages"]]
    assert any("mensaje en t1" in c for c in contents_t1)
    assert not any("mensaje en t1" in c for c in contents_t2)
    assert any("mensaje en t2" in c for c in contents_t2)


def test_system_prompt_carries_materia_into_llm_call(tmp_vault):
    vault = VaultManager(tmp_vault)
    chroma = MagicMock()
    fake = _FakeChatLLM()
    with patch("resumini.agent.graph.get_llm", return_value=fake):
        graph = build_graph(vault, chroma)
        graph.invoke(
            {"messages": [HumanMessage(content="hola")], "materia": "civil"},
            config={"configurable": {"thread_id": "tm"}},
        )
    assert fake.captured, "LLM was never invoked"
    sys_msg = fake.captured[0][0]
    assert "civil" in sys_msg.content


def test_system_prompt_includes_profile(tmp_vault):
    vault = VaultManager(tmp_vault)
    (tmp_vault / "profile.md").write_text(
        "# Perfil\n\nEstilo: bullet points con highlights"
    )
    chroma = MagicMock()
    fake = _FakeChatLLM()
    with patch("resumini.agent.graph.get_llm", return_value=fake):
        graph = build_graph(vault, chroma)
        graph.invoke(
            {"messages": [HumanMessage(content="hola")]},
            config={"configurable": {"thread_id": "tp"}},
        )
    sys_msg = fake.captured[0][0]
    assert "bullet points" in sys_msg.content

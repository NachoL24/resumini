"""Conversational agent: tool-using ReAct loop with conversation memory."""

from typing import Annotated, TypedDict

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import create_react_agent

from resumini.agent.prompts import build_system_prompt
from resumini.agent.tools import make_tools
from resumini.db.chroma import ChromaClient
from resumini.llm.client import get_llm
from resumini.vault.manager import VaultManager


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    remaining_steps: RemainingSteps
    materia: str | None


def build_graph(
    vault: VaultManager,
    chroma: ChromaClient,
    checkpointer: BaseCheckpointSaver | None = None,
):
    llm = get_llm()
    tools = make_tools(vault, chroma)

    def make_prompt(state: ChatState) -> list:
        materia = state.get("materia")
        profile = vault.read_profile()
        sys = build_system_prompt(materia, profile)
        return [SystemMessage(content=sys)] + list(state["messages"])

    if checkpointer is None:
        checkpointer = MemorySaver()

    return create_react_agent(
        llm,
        tools=tools,
        state_schema=ChatState,
        prompt=make_prompt,
        checkpointer=checkpointer,
    )

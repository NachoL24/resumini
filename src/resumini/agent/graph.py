from typing import TypedDict

from langgraph.graph import END, StateGraph

from resumini.agent.edit_node import run_edit
from resumini.agent.ingest_node import run_ingest
from resumini.agent.memory_node import update_memory
from resumini.agent.query_node import run_query
from resumini.agent.response_node import format_response
from resumini.agent.router import Intent, classify_intent
from resumini.agent.summarize_node import run_summarize
from resumini.agent.validate_node import validate_output
from resumini.db.chroma import ChromaClient
from resumini.db.sqlite import SQLiteStore
from resumini.vault.manager import VaultManager


class AgentState(TypedDict):
    message: str
    intent: Intent | None
    materia: str
    source_file: str | None
    output_file: str | None
    pdf_path: str | None
    edit_instruction: str | None
    output: str | None
    valid: bool
    errors: list[str]
    metadata: dict


def _route_intent(state: AgentState) -> str:
    if state["intent"] == Intent.INGEST:
        return "ingest"
    elif state["intent"] == Intent.SUMMARIZE:
        return "summarize"
    elif state["intent"] == Intent.EDIT:
        return "edit"
    return "query"


def build_graph(vault: VaultManager, chroma: ChromaClient, sqlite: SQLiteStore):
    graph = StateGraph(AgentState)

    def router_node(state: AgentState) -> dict:
        intent = classify_intent(state["message"])
        return {"intent": intent}

    def ingest_node(state: AgentState) -> dict:
        from pathlib import Path

        content = run_ingest(
            Path(state["pdf_path"]), state["materia"], state["source_file"], vault, chroma
        )
        return {
            "output": content,
            "metadata": {"materia": state["materia"], "file": state["source_file"]},
        }

    def summarize_node(state: AgentState) -> dict:
        source = state.get("source_file", "")
        output = state.get("output_file", source.replace(".md", "_resumen.md"))
        content = run_summarize(state["materia"], source, output, vault, chroma)
        return {"output": content, "metadata": {"materia": state["materia"], "file": output}}

    def query_node_fn(state: AgentState) -> dict:
        content = run_query(state["message"], vault, chroma)
        return {"output": content}

    def edit_node_fn(state: AgentState) -> dict:
        content = run_edit(
            state["materia"], state["source_file"], state["edit_instruction"], vault, chroma
        )
        return {
            "output": content,
            "metadata": {"materia": state["materia"], "file": state["source_file"]},
        }

    def validate_node(state: AgentState) -> dict:
        result = validate_output(state["output"] or "")
        return {"valid": result["valid"], "errors": result["errors"]}

    def memory_node(state: AgentState) -> dict:
        if state.get("materia") and state.get("source_file") and state.get("output"):
            update_memory(
                state["materia"], state["source_file"], state["output"], vault, chroma, sqlite
            )
        return {}

    def response_node(state: AgentState) -> dict:
        return {"output": format_response(state["output"] or "", state.get("metadata"))}

    graph.add_node("router", router_node)
    graph.add_node("ingest", ingest_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("query", query_node_fn)
    graph.add_node("edit", edit_node_fn)
    graph.add_node("validate", validate_node)
    graph.add_node("memory", memory_node)
    graph.add_node("response", response_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        _route_intent,
        {"ingest": "ingest", "summarize": "summarize", "query": "query", "edit": "edit"},
    )
    graph.add_edge("ingest", "validate")
    graph.add_edge("summarize", "validate")
    graph.add_edge("query", "response")
    graph.add_edge("edit", "validate")
    graph.add_conditional_edges(
        "validate",
        lambda s: "memory" if s["valid"] else "response",
        {"memory": "memory", "response": "response"},
    )
    graph.add_edge("memory", "response")
    graph.add_edge("response", END)

    return graph.compile()

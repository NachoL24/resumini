import argparse
import sqlite3
from typing import Any

from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

from resumini.agent.graph import build_graph
from resumini.config import get_settings
from resumini.db.chroma import ChromaClient
from resumini.vault.manager import VaultManager


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    materia: str | None = None


class ChatResponse(BaseModel):
    output: str


def _build_default_components():
    settings = get_settings()
    vault = VaultManager(settings.vault_dir)
    chroma = ChromaClient(
        persist_dir=str(settings.chroma_dir),
        embedding_model_name=settings.embedding_model,
        nvidia_api_key=settings.nvidia_api_key or None,
        nvidia_base_url=settings.nvidia_base_url,
    )
    checkpoint_db = str(settings.db_path.parent / "checkpoints.db")
    conn = sqlite3.connect(checkpoint_db, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = build_graph(vault, chroma, checkpointer=checkpointer)
    return vault, chroma, graph


def create_app(
    vault: VaultManager | None = None,
    chroma: ChromaClient | None = None,
    graph: Any = None,
) -> FastAPI:
    if vault is None or chroma is None or graph is None:
        default_vault, default_chroma, default_graph = _build_default_components()
        vault = vault or default_vault
        chroma = chroma or default_chroma
        graph = graph or default_graph

    app = FastAPI(title="Resumini", version="0.2.0")
    app.state.vault = vault
    app.state.chroma = chroma
    app.state.graph = graph

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        input_state: dict = {"messages": [HumanMessage(content=req.message)]}
        if req.materia is not None:
            input_state["materia"] = req.materia
        result = graph.invoke(
            input_state,
            config={"configurable": {"thread_id": req.session_id}},
        )
        last = result["messages"][-1]
        text = last.content if hasattr(last, "content") else str(last)
        return ChatResponse(output=text)

    return app


def cli():
    parser = argparse.ArgumentParser(
        description="Resumini — agente conversacional de estudio"
    )
    parser.add_argument("--serve", action="store_true", help="Levantar la API")
    parser.add_argument("--chat", type=str, help="Mensaje al agente")
    parser.add_argument(
        "--session",
        type=str,
        default="default",
        help="ID de sesion (mantiene memoria entre llamadas)",
    )
    parser.add_argument(
        "--materia", type=str, default=None, help="Materia activa para la sesion"
    )
    args = parser.parse_args()

    if args.serve:
        import uvicorn

        uvicorn.run(create_app(), host="0.0.0.0", port=8000)
        return
    if args.chat:
        _, _, graph = _build_default_components()
        input_state: dict = {"messages": [HumanMessage(content=args.chat)]}
        if args.materia:
            input_state["materia"] = args.materia
        result = graph.invoke(
            input_state,
            config={"configurable": {"thread_id": args.session}},
        )
        last = result["messages"][-1]
        print(last.content if hasattr(last, "content") else str(last))
        return
    parser.print_help()


if __name__ == "__main__":
    cli()

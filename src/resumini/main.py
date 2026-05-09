import argparse

from fastapi import FastAPI
from pydantic import BaseModel

from resumini.config import get_settings
from resumini.vault.manager import VaultManager
from resumini.db.chroma import ChromaClient
from resumini.db.sqlite import SQLiteStore
from resumini.agent.graph import AgentState, build_graph


def create_app() -> FastAPI:
    settings = get_settings()
    vault = VaultManager(settings.vault_dir)
    chroma = ChromaClient(
        persist_dir=str(settings.chroma_dir),
        embedding_model_name=settings.embedding_model,
        nvidia_api_key=settings.nvidia_api_key or None,
        nvidia_base_url=settings.nvidia_base_url,
    )
    sqlite = SQLiteStore(str(settings.db_path))
    sqlite.initialize()
    graph = build_graph(vault, chroma, sqlite)

    app = FastAPI(title="Resumini", version="0.1.0")
    app.state.vault = vault
    app.state.chroma = chroma
    app.state.sqlite = sqlite
    app.state.graph = graph

    class MessageRequest(BaseModel):
        message: str
        materia: str = ""
        source_file: str | None = None
        output_file: str | None = None
        pdf_path: str | None = None
        edit_instruction: str | None = None

    class MessageResponse(BaseModel):
        output: str
        intent: str | None
        valid: bool
        errors: list[str]

    @app.post("/chat", response_model=MessageResponse)
    async def chat(req: MessageRequest):
        state = AgentState(
            message=req.message,
            intent=None,
            materia=req.materia,
            source_file=req.source_file,
            output_file=req.output_file,
            pdf_path=req.pdf_path,
            edit_instruction=req.edit_instruction,
            output=None,
            valid=False,
            errors=[],
            metadata={},
        )
        result = graph.invoke(state)
        return MessageResponse(
            output=result.get("output", ""),
            intent=result.get("intent"),
            valid=result.get("valid", False),
            errors=result.get("errors", []),
        )

    return app


def cli():
    parser = argparse.ArgumentParser(description="Resumini - Agente de resumenes")
    parser.add_argument("--serve", action="store_true", help="Start the API server")
    parser.add_argument("--chat", type=str, help="Send a message to the agent")
    parser.add_argument("--materia", type=str, default="", help="Materia context")
    parser.add_argument("--ingest-pdf", type=str, help="Path to PDF to ingest")
    parser.add_argument("--source", type=str, help="Source file name")
    parser.add_argument("--output", type=str, help="Output file name")
    args = parser.parse_args()

    if args.serve:
        import uvicorn

        uvicorn.run(create_app(), host="0.0.0.0", port=8000)
    elif args.chat:
        settings = get_settings()
        vault = VaultManager(settings.vault_dir)
        chroma = ChromaClient(
            persist_dir=str(settings.chroma_dir),
            embedding_model_name=settings.embedding_model,
            nvidia_api_key=settings.nvidia_api_key or None,
            nvidia_base_url=settings.nvidia_base_url,
        )
        sqlite = SQLiteStore(str(settings.db_path))
        sqlite.initialize()
        graph = build_graph(vault, chroma, sqlite)

        state = AgentState(
            message=args.chat,
            intent=None,
            materia=args.materia,
            source_file=args.source,
            output_file=args.output,
            pdf_path=args.ingest_pdf,
            edit_instruction=None,
            output=None,
            valid=False,
            errors=[],
            metadata={},
        )
        result = graph.invoke(state)
        print(result.get("output", "Sin respuesta"))
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()

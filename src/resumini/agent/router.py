from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage

from resumini.llm.client import get_llm


class Intent(str, Enum):
    INGEST = "ingest"
    SUMMARIZE = "summarize"
    QUERY = "query"
    EDIT = "edit"


ROUTER_SYSTEM_PROMPT = """Eres un clasificador de intenciones para un agente de resumenes academicos.
Clasifica el mensaje del usuario en una de estas categorias:
- ingest: el usuario quiere procesar/ingestar material nuevo (PDF, video, apunte)
- summarize: el usuario quiere generar un resumen de material ya procesado
- query: el usuario pregunta sobre contenido estudiado
- edit: el usuario quiere modificar un resumen existente

Responde SOLO con la categoria, sin explicacion."""


def classify_intent(message: str) -> Intent:
    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=message),
        ]
    )
    intent_str = response.content.strip().lower()
    intent_map = {
        "ingest": Intent.INGEST,
        "summarize": Intent.SUMMARIZE,
        "query": Intent.QUERY,
        "edit": Intent.EDIT,
    }
    return intent_map.get(intent_str, Intent.QUERY)

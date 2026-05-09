import re

from langchain_core.messages import HumanMessage, SystemMessage

from resumini.db.chroma import ChromaClient
from resumini.llm.client import get_llm
from resumini.vault.manager import VaultManager

WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")

QUERY_PROMPT = """Eres un asistente academico que responde preguntas basandose en resumenes y apuntes.
Usa el contexto proporcionado para dar una respuesta precisa.
Si la informacion no esta en el contexto, dilo claramente.

Perfil del estudiante:
{profile}

Contexto relevante:
{context}

Pregunta: {question}"""


def _resolve_wikilinks_in_question(question: str, vault: VaultManager) -> list[str]:
    extra_context = []
    for match in WIKILINK_RE.findall(question):
        content = vault.get_embed_content(match)
        if content:
            extra_context.append(f"[Referencia wikilink: {match}]\n{content}")
    return extra_context


def run_query(question: str, vault: VaultManager, chroma: ChromaClient) -> str:
    search_results = chroma.search(question, n_results=5)
    context_parts = []
    for r in search_results:
        context_parts.append(f"[{r['metadata']['materia']}/{r['metadata']['file']}] {r['content']}")
    wikilink_context = _resolve_wikilinks_in_question(question, vault)
    for wc in wikilink_context:
        context_parts.append(wc)
    context = "\n\n".join(context_parts)
    profile = vault.read_profile()
    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(
                content=QUERY_PROMPT.format(
                    profile=profile, context=context, question=question
                )
            ),
            HumanMessage(content=question),
        ]
    )
    return response.content

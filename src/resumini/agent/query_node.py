from langchain_core.messages import HumanMessage, SystemMessage

from resumini.db.chroma import ChromaClient
from resumini.llm.client import get_llm
from resumini.vault.manager import VaultManager

QUERY_PROMPT = """Eres un asistente academico que responde preguntas basandote en resumenes y apuntes.
Usa el contexto proporcionado para dar una respuesta precisa.
Si la informacion no esta en el contexto, dilo claramente.

Perfil del estudiante:
{profile}

Contexto relevante:
{context}

Pregunta: {question}"""


def run_query(question: str, vault: VaultManager, chroma: ChromaClient) -> str:
    search_results = chroma.search(question, n_results=5)
    context_parts = []
    for r in search_results:
        context_parts.append(f"[{r['metadata']['materia']}/{r['metadata']['file']}] {r['content']}")
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

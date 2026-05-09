from langchain_core.messages import HumanMessage, SystemMessage

from resumini.db.chroma import ChromaClient
from resumini.llm.client import get_llm
from resumini.vault.manager import VaultManager

SUMMARIZE_PROMPT = """Eres un asistente academico que genera resumenes de material de estudio.
Usa el perfil del estudiante para adaptar el formato y nivel de detalle.
Genera un resumen claro, estructurado, en markdown.

Perfil del estudiante:
{profile}

Contenido fuente:
{content}

Genera el resumen:"""


def run_summarize(
    materia: str,
    source_file: str,
    output_file: str,
    vault: VaultManager,
    chroma: ChromaClient,
) -> str:
    source_content = vault.read_note(materia, source_file)
    profile = vault.read_profile()
    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(
                content=SUMMARIZE_PROMPT.format(profile=profile, content=source_content)
            ),
            HumanMessage(
                content=f"Genera un resumen de {source_file} para la materia {materia}"
            ),
        ]
    )
    summary = response.content
    vault.write_note(materia, output_file, summary)
    chroma.index_document(
        doc_id=f"{materia}_{output_file.replace('.md', '')}",
        content=summary,
        metadata={"materia": materia, "file": output_file, "type": "summary"},
    )
    return summary

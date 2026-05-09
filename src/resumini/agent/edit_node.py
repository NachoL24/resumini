from langchain_core.messages import HumanMessage, SystemMessage

from resumini.db.chroma import ChromaClient
from resumini.llm.client import get_llm
from resumini.vault.manager import VaultManager

EDIT_PROMPT = """Eres un asistente academico que edita resumenes existentes.
Aplica la edicion solicitada al contenido actual, manteniendo el formato y estilo.

Contenido actual:
{current_content}

Edicion solicitada:
{edit_instruction}

Devuelve el contenido completo editado en markdown:"""


def run_edit(
    materia: str,
    filename: str,
    instruction: str,
    vault: VaultManager,
    chroma: ChromaClient,
) -> str:
    current = vault.read_note(materia, filename)
    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(
                content=EDIT_PROMPT.format(
                    current_content=current, edit_instruction=instruction
                )
            ),
            HumanMessage(content=instruction),
        ]
    )
    edited = response.content
    vault.write_note(materia, filename, edited)
    chroma.index_document(
        doc_id=f"{materia}_{filename.replace('.md', '')}",
        content=edited,
        metadata={"materia": materia, "file": filename, "type": "summary"},
    )
    return edited

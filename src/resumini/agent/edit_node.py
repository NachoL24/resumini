from langchain_core.messages import HumanMessage, SystemMessage

from resumini.db.chroma import ChromaClient
from resumini.llm.client import get_llm
from resumini.vault.manager import VaultManager
from resumini.vault.obsidian import extract_frontmatter, render_frontmatter

EDIT_PROMPT = """Eres un asistente academico que edita resumenes existentes.
Aplica la edicion solicitada al contenido actual, manteniendo el formato y estilo.

Devuelve SOLO el contenido editado (sin YAML frontmatter, solo el body markdown).

Contenido actual (body sin frontmatter):
{current_content}

Edicion solicitada:
{edit_instruction}

Devuelve el contenido completo editado en markdown (sin frontmatter):"""


def run_edit(
    materia: str,
    filename: str,
    instruction: str,
    vault: VaultManager,
    chroma: ChromaClient,
) -> str:
    raw = vault.read_note(materia, filename)
    fm, body = extract_frontmatter(raw)
    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(
                content=EDIT_PROMPT.format(current_content=body, edit_instruction=instruction)
            ),
            HumanMessage(content=instruction),
        ]
    )
    edited_body = response.content
    fm_block = render_frontmatter(fm)
    edited = f"{fm_block}\n{edited_body}" if fm_block else edited_body
    vault.write_note(materia, filename, edited)
    chroma.index_document(
        doc_id=f"{materia}_{filename.replace('.md', '')}",
        content=edited,
        metadata={"materia": materia, "file": filename, "type": "summary"},
    )
    return edited

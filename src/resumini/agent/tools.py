"""Tools exposed to the conversational agent."""

from pathlib import Path

from langchain_core.tools import tool

from resumini.agent.edit_node import run_edit
from resumini.agent.ingest_node import run_ingest
from resumini.agent.query_node import search_context
from resumini.agent.summarize_node import run_summarize
from resumini.db.chroma import ChromaClient
from resumini.vault.manager import VaultManager


def make_task_tools(vault: VaultManager, chroma: ChromaClient) -> list:
    """Build the four task tools bound to vault and chroma instances."""

    @tool
    def ingest_pdf(pdf_path: str, materia: str, filename: str) -> str:
        """Ingesta un PDF en el vault. Extrae el texto, lo guarda como .md con
        frontmatter Obsidian y lo indexa para busqueda. Usalo cuando el
        estudiante te indica que tiene material nuevo en PDF para procesar.

        Args:
            pdf_path: ruta absoluta al PDF en disco.
            materia: nombre de la materia (ej: 'civil', 'penal').
            filename: nombre del archivo .md de destino (ej: 'unidad_1.md').
        """
        try:
            content = run_ingest(Path(pdf_path), materia, filename, vault, chroma)
        except FileNotFoundError:
            return f"Error: PDF no encontrado en {pdf_path}"
        except ValueError as e:
            return f"Error: PDF invalido — {e}"
        pages = content.count("## Pagina ")
        return f"Ingestado {pdf_path} en {materia}/{filename} ({pages} paginas)."

    @tool
    def summarize(
        materia: str,
        source_file: str,
        output_file: str | None = None,
        instructions: str | None = None,
    ) -> str:
        """Genera un resumen Obsidian de una nota existente del vault. El
        resumen lleva frontmatter, callouts y wikilinks. Si el estudiante te
        pidio foco en algo (ejemplos, conceptos, sin tablas, etc.), pasalo en
        `instructions`.

        Args:
            materia: nombre de la materia.
            source_file: nota fuente a resumir (.md).
            output_file: nombre del .md de salida; default `{source}_resumen.md`.
            instructions: instrucciones especificas del estudiante para este resumen.
        """
        out = output_file or source_file.replace(".md", "_resumen.md")
        try:
            run_summarize(
                materia, source_file, out, vault, chroma, instructions=instructions
            )
        except FileNotFoundError:
            return f"Error: nota fuente {materia}/{source_file} no existe"
        return f"Resumen generado en {materia}/{out}"

    @tool
    def search_vault(question: str, n_results: int = 5) -> str:
        """Busca en el vault notas y fragmentos relacionados con la pregunta.
        Devuelve fragmentos formateados como [materia/archivo] contenido. Usalo
        para conseguir contexto antes de responder una pregunta del estudiante.

        Args:
            question: pregunta o tema a buscar.
            n_results: cantidad maxima de resultados (default 5).
        """
        fragments = search_context(question, vault, chroma, n_results=n_results)
        if not fragments:
            return "No se encontraron resultados en el vault."
        return "\n\n".join(fragments)

    @tool
    def edit_note(materia: str, filename: str, instruction: str) -> str:
        """Edita una nota existente aplicando la instruccion del estudiante,
        preservando el frontmatter Obsidian.

        Args:
            materia: nombre de la materia.
            filename: archivo .md a editar.
            instruction: que cambio hacer en la nota.
        """
        try:
            run_edit(materia, filename, instruction, vault, chroma)
        except FileNotFoundError:
            return f"Error: nota {materia}/{filename} no existe"
        return f"Nota {materia}/{filename} editada."

    return [ingest_pdf, summarize, search_vault, edit_note]


def make_readonly_tools(vault: VaultManager) -> list:
    """Build the read-only navigation tools bound to the vault."""

    @tool
    def list_materias() -> str:
        """Lista todas las materias presentes en el vault. Sin argumentos."""
        materias = vault.list_materias()
        if not materias:
            return "No hay materias en el vault todavia."
        return "Materias: " + ", ".join(sorted(materias))

    @tool
    def list_notes(materia: str) -> str:
        """Lista las notas (.md) de una materia.

        Args:
            materia: nombre de la materia.
        """
        notes = vault.list_notes(materia)
        if not notes:
            return f"La materia '{materia}' esta vacia o no existe."
        return f"Notas en {materia}: " + ", ".join(sorted(notes))

    @tool
    def read_note(materia: str, filename: str) -> str:
        """Lee el contenido completo (incluyendo frontmatter) de una nota.

        Args:
            materia: nombre de la materia.
            filename: nombre del archivo .md.
        """
        try:
            return vault.read_note(materia, filename)
        except FileNotFoundError:
            return f"Error: la nota {materia}/{filename} no existe"

    return [list_materias, list_notes, read_note]


def make_tools(vault: VaultManager, chroma: ChromaClient) -> list:
    """Build the full tool set (task + read-only) for the conversational agent."""
    return make_task_tools(vault, chroma) + make_readonly_tools(vault)

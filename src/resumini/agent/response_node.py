def format_response(content: str, metadata: dict | None = None) -> str:
    parts = [content]
    if metadata:
        if metadata.get("materia"):
            parts.append(f"\n---\n*Materia: {metadata['materia']}*")
        if metadata.get("file"):
            parts.append(f" *Archivo: {metadata['file']}*")
    return "\n".join(parts)

def validate_output(content: str) -> dict:
    errors = []
    if not content or not content.strip():
        errors.append("empty_output")
    if len(content.strip()) < 20:
        errors.append("too_short")
    return {"valid": len(errors) == 0, "errors": errors}

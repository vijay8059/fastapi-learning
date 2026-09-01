_MAX_CHARS = 1200
_OVERLAP_CHARS = 150


def chunk_markdown(text: str, max_chars: int = _MAX_CHARS, overlap_chars: int = _OVERLAP_CHARS) -> list[str]:
    """Greedy paragraph packing — accumulate paragraphs up to max_chars, and
    carry the tail of each chunk into the next one so a fact split across a
    paragraph boundary isn't orphaned in a chunk with no surrounding context.
    """

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars or not current:
            current = candidate
            continue
        chunks.append(current)
        current = f"{current[-overlap_chars:]}\n\n{paragraph}"
    if current:
        chunks.append(current)
    return chunks

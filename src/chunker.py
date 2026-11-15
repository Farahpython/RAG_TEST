
def chunk_text(text: str, max_size: int) -> list[tuple[int, int]]:
    
    """Split text into consecutive (start, end) spans, cutting at line ends.
       text[start:end] is the chunk
    """
    if max_size <= 0:
        raise ValueError("max_size must be greater than 0")

    chunks: list[tuple[int, int]] = []
    chunk_start = 0 
    pos = 0
    for line in text.splitlines(keepends=True):
        if len(line) > max_size:
            if pos > chunk_start:
                chunks.append((chunk_start, pos))
            line_end = pos + len(line)
            while pos < line_end:
                end = min(pos + max_size, line_end)
                chunks.append((pos, end))
                pos = end
            chunk_start = pos
            continue
        if pos + len(line) - chunk_start > max_size:
            chunks.append((chunk_start, pos))
            chunk_start = pos
        pos += len(line)
    if pos > chunk_start:
        chunks.append((chunk_start, pos))
    return chunks

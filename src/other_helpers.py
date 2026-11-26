from typing import Any
from .models import MinimalSource
from llm_sdk import Small_LLM_Model

def get_chunk_text(source: MinimalSource) -> str:
    with open(source.file_path) as f:
        content = f.read()
    return content[
        source.first_character_index:source.last_character_index
    ]


def is_match(source1: MinimalSource, source2: MinimalSource) -> Any:
    if source1.file_path != source2.file_path:
        return False
    intersection = max(
        0,
        min(source1.last_character_index, source2.last_character_index)
        - max(source1.first_character_index, source2.first_character_index)
    )
    union = (
        max(source1.last_character_index, source2.last_character_index)
        - min(source1.first_character_index, source2.first_character_index)
    )
    return union > 0 and intersection / union >= 0.05


def chunk_text(text: str, max_size: int) -> list[tuple[int, int]]:
    """Split text into consecutive character spans."""
    if max_size <= 0:
        raise ValueError("max_size must be greater than 0")
    chunks = []
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

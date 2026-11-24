import re
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())


def markdown_only_chunk_text(text, max_size, overlap=300):
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    if overlap < 0 or overlap >= max_size:
        raise ValueError("overlap must be between 0 and max_size")
    heading_pattern = re.compile(
        r"^#{1,6}[ \t]+.+$",
        re.MULTILINE
    )
    headings = list(heading_pattern.finditer(text))
    if not headings:
        sections = [(0, len(text))]
    else:
        sections = []
        if headings[0].start() > 0:
            sections.append((0, headings[0].start()))
        for i, heading in enumerate(headings):
            start = heading.start()
            if i + 1 < len(headings):
                end = headings[i + 1].start()
            else:
                end = len(text)
            sections.append((start, end))
    chunks = []
    for section_start, section_end in sections:
        start = section_start
        while start < section_end:
            end = min(start + max_size, section_end)
            chunks.append((start, end))
            if end == section_end:
                break
            start = end - overlap
    return chunks


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


def python_code_chunker(text: str, max_size: int) -> list[tuple[int, int]]:
    """Split Python source into structure-aware character spans."""
    if max_size <= 0:
        raise ValueError("max_size must be greater than 0")
    parser = Parser(PY_LANGUAGE)
    source_bytes = text.encode("utf-8")
    tree = parser.parse(source_bytes)
    chunks: list[tuple[int, int]] = []

    def byte_to_char(offset: int) -> int:
        return len(source_bytes[:offset].decode("utf-8"))

    current_start = None
    current_end = None

    def flush() -> None:
        nonlocal current_start, current_end
        if current_start is not None:
            chunks.append((
                byte_to_char(current_start),
                byte_to_char(current_end)
            ))
        current_start = None
        current_end = None
    for node in tree.root_node.named_children:
        node_start = node.start_byte
        node_end = node.end_byte
        start_char = byte_to_char(node_start)
        end_char = byte_to_char(node_end)
        if end_char - start_char > max_size:
            flush()
            if node.text is None:
                continue
            node_text = node.text.decode("utf-8")
            for start, end in chunk_text(node_text, max_size):
                chunks.append((
                    start_char + start,
                    start_char + end
                ))
            continue
        if current_start is None:
            current_start = node_start
            current_end = node_end
            continue
        candidate_size = byte_to_char(node_end) - byte_to_char(current_start)
        if candidate_size > max_size:
            flush()
            current_start = node_start
            current_end = node_end
        else:
            current_end = node_end
    flush()
    return chunks

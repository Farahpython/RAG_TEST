import tree_sitter_python as tspython
from markdown_it import MarkdownIt
from tree_sitter import Language, Parser
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import PunktTokenizer

PY_LANGUAGE = Language(tspython.language())
EMBEDDING_MODEL = SentenceTransformer("all-mpnet-base-v2")
TOKENIZER = PunktTokenizer("english")



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


def semantic_chunker(
    text: str,
    max_size: int,
    similarity_threshold: float = 0.3,
) -> list[tuple[int, int]]:
    """Split text into semantically related sentence spans."""

    if max_size <= 0:
        raise ValueError("max_size must be greater than 0")

    sentence_positions = list(TOKENIZER.span_tokenize(text))

    if not sentence_positions:
        return []

    sentences = [
        text[start:end]
        for start, end in sentence_positions
    ]

    embeddings = EMBEDDING_MODEL.encode(
        sentences,
        convert_to_tensor=True,
    )

    chunks = []
    chunk_start = None
    current_end = None

    for i, (start, end) in enumerate(sentence_positions):

        # Handle a sentence that exceeds the maximum size.
        if end - start > max_size:
            if chunk_start is not None:
                chunks.append((chunk_start, current_end))

            for sub_start, sub_end in chunk_text(
                text[start:end], max_size
            ):
                chunks.append((
                    start + sub_start,
                    start + sub_end,
                ))

            chunk_start = None
            current_end = None
            continue

        if chunk_start is None:
            chunk_start = start
            current_end = end
            continue

        similarity = util.cos_sim(
            embeddings[i - 1],
            embeddings[i],
        ).item()

        candidate_size = end - chunk_start

        if (
            similarity < similarity_threshold
            or candidate_size > max_size
        ):
            chunks.append((chunk_start, current_end))
            chunk_start = start

        current_end = end

    if chunk_start is not None:
        chunks.append((chunk_start, current_end))

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


if __name__ == "__main__":
    text = (
        "Python is a programming language. "
        "It is widely used for data science and machine learning. "
        "The weather is sunny today. "
        "The temperature is around 25 degrees."
    )

    spans = semantic_chunker(
        text,
        max_size=200,
        similarity_threshold=0.3,
    )

    for start, end in spans:
        print("--- CHUNK ---")
        print(repr(text[start:end]))
        print("Length:", end - start)
        print("Offsets:", start, end)

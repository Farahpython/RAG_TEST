import re
from tqdm import tqdm
from pathlib import Path
from .chunker import chunk_text, markdown_chunker, python_code_chunker
from .models import MinimalSource

EXT = {".py", ".md", ".txt"}


def get_markdown_context(text: str, position: int) -> str:
    """Get the heading hierarchy surrounding a Markdown position."""

    code_pattern = re.compile(
        r"(?ms)^[ \t]*(`{3,}|~{3,})[^\n]*\n.*?^[ \t]*\1[ \t]*$"
    )

    protected = code_pattern.sub(
        lambda m: "".join(
            "\n" if char == "\n" else " "
            for char in m.group(0)
        ),
        text,
    )

    heading_pattern = re.compile(
        r"^(#{1,6})[ \t]+(.+?)\s*$",
        re.MULTILINE,
    )

    hierarchy = []

    for match in heading_pattern.finditer(protected):
        if match.start() > position:
            break

        level = len(match.group(1))
        title = match.group(2).strip()

        while hierarchy and hierarchy[-1][0] >= level:
            hierarchy.pop()

        hierarchy.append((level, title))

    return " > ".join(title for _, title in hierarchy)


def read_corpus(root: str = "data/raw") -> list[tuple[str, str]]:
    """Return (relative_path, text) pairs for every indexable file under root.
       Unreadable files are skipped with a warning.
    """
    files = sorted(
        p for p in Path(root).rglob("*")
        if p.is_file() and p.suffix in EXT
    )
    result: list[tuple[str, str]] = []
    for p in files:
        try:
            with open(p, encoding="utf-8") as f:
                text = f.read()
        except (UnicodeDecodeError, OSError):
            print(f"Skipping unreadable file: {p}")
            continue
        result.append((str(p), text))
    return result


def build_chunks(
    corpus: list[tuple[str, str]],
    max_size: int,
) -> list[tuple[str, MinimalSource]]:
    """Build text chunks with their source metadata.
       Each chunk is tagged with metadata for quick filtering.
    """
    chunks = []
    if max_size <= 0:
        raise ValueError("max_chunk_size must be greater than zero")
    for file_path, content in tqdm(corpus, desc="Chunking", unit="file"):
        if Path(file_path).suffix == ".py":
            spans = python_code_chunker(content, max_size)
        elif Path(file_path).suffix == ".md":
            spans = markdown_chunker(content, max_size, overlap=600)
        else:
            spans = chunk_text(content, max_size)
        for start, end in spans:
            source = MinimalSource(
                file_path=file_path,
                first_character_index=start,
                last_character_index=end
            )
            enriched_text = content[start:end]
            if Path(file_path).suffix == ".md":
                context = get_markdown_context(content, start)
                enriched_text = (
                    f"Document: {Path(file_path).name}\n"
                    f"Section: {context}\n\n"
                    f"{enriched_text}"
                )
            chunks.append((enriched_text, source))
    return chunks


def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", text)
    return re.findall(r"[A-Za-z0-9]+", text.lower())

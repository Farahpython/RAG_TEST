# import fire
from pathlib import Path
import re
import bm25s

EXT = {".py", ".md", ".txt"}

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


_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens, breaking snake_case and camelCase."""
    tokens: list[str] = []
    for word in _NON_ALNUM.split(text):
        if not word:
            continue
        parts = _CAMEL.sub(" ", word).split()
        tokens.extend(p.lower() for p in parts)
    return tokens

chunks = [
    "def max_chunk_size(): return 2000",
    "How to configure the OpenAI server in vLLM",
    "LoRA adapters can be loaded at runtime",
]


if __name__ == "__main__":
    corpus_tokens = [tokenize(c) for c in chunks]
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    query_tokens = tokenize("configure openai server")
    results, scores = retriever.retrieve([query_tokens], k=2)
    print(results, scores)

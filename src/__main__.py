# import fire
import json
import re
from pathlib import Path
import bm25s
from .chunker import chunk_text, python_code_chunker
from .models import MinimalSource

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


def build_chunks(corpus: list[tuple[str, str]],
                 max_size: int) -> list[tuple[str, MinimalSource]]:
    """Build text chunks with their source metadata."""
    chunks = []

    for file_path, content in corpus:
        if Path(file_path).suffix == ".py":
            spans = python_code_chunker(content, max_size)
        else:
            spans = chunk_text(content, max_size)

        for start, end in spans:
            source = MinimalSource(
                file_path=file_path,
                first_character_index=start,
                last_character_index=end
            )
            chunks.append((content[start:end], source))

    return chunks


def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", text)

    words = re.split(r"[^A-Za-z0-9]+", text)

    return [word.lower() for word in words if word]


def search(query, retriever, chunks, valid_indices, k=10):
    query_tokens = tokenize(query)

    results, scores = retriever.retrieve([query_tokens], k=k)

    sources = []
    for idx in results[0]:
        original_idx = valid_indices[idx]
        text, source = chunks[original_idx]
        sources.append(source)

    return sources


if __name__ == "__main__":
    corpus = read_corpus()
    chunks = build_chunks(corpus, max_size=2000)
    tokenized_chunks = [
        tokenize(text)
        for text, source in chunks
    ]
    valid_indices = [
        i for i, tokens in enumerate(tokenized_chunks) if tokens
    ]
    documents = [tokenized_chunks[i] for i in valid_indices]
    retriever = bm25s.BM25()
    retriever.index(documents)
    retriever.save("data/processed/bm25_index")
    retriever = bm25s.BM25.load("data/processed/bm25_index")
    with open("data+moulinette/datasets_public/public/AnsweredQuestions/dataset_code_public.json") as f:
        dataset = json.load(f)
    search_results = []
    for item in dataset["rag_questions"]:
        sources = search(item["question"], retriever, chunks, valid_indices)
        search_results.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "retrieved_sources": [
                source.model_dump() for source in sources
            ]
        })
    output = {
        "search_results": search_results,
        "k": 10
    }
    with open("data/processed/code_public_results.json", "w") as f:
        json.dump(output, f, indent=4)

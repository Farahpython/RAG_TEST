import fire
import json
import bm25s
from pathlib import Path
from typing import List
from .indexer_helpers import read_corpus, build_chunks, tokenize
# from .chunker import chunk_text, python_code_chunker
from .models import MinimalSource, RagDataset, MinimalSearchResults, StudentSearchResults


def index(max_chunk_size: int = 2000) -> None:
    """Read the corpus, build the BM25 index, and save it."""
    if type(max_chunk_size) is not int or max_chunk_size <= 0:
        print("max_chunk_size must be a positive integer!!")
        return 
    corpus = read_corpus()
    chunks = build_chunks(corpus, max_size=max_chunk_size)
    tokenized_chunks = [
        tokenize(text)
        for text, source in chunks
    ]
    # checks whether the token list is non-empty
    valid_indices = [
        i for i, tokens in enumerate(tokenized_chunks) if tokens
    ]
    # lists of validated tokens
    documents = [tokenized_chunks[i] for i in valid_indices]
    retriever = bm25s.BM25()
    retriever.index(documents)
    retriever.save("data/processed/bm25_index")
    with open("data/processed/valid_indices.json", "w") as f:
        json.dump(valid_indices, f)
    with open("data/processed/sources.json", "w") as f:
        json.dump(
            [source.model_dump() for text, source in chunks],
            f
        )
    print("Ingestion complete! Indices saved under data/processed/")


def search(query: str, k: int = 10) -> list[MinimalSource]:
    """Retrieve the top-k sources for a query."""
    if not query.strip():
        print("Empty Query!!")
        return []
    if k == 0:
        print("you asked for no results btw, who puts k at zero? are you insane?")
        return []
    if k < 0:
        print("k cannot be less than zero!!")
        return
    retriever = bm25s.BM25.load("data/processed/bm25_index")
    with open("data/processed/valid_indices.json") as f:
        valid_indices = json.load(f)
    with open("data/processed/sources.json") as f:
        sources = json.load(f)
    query_tokens = tokenize(query)
    results, scores = retriever.retrieve([query_tokens], k=k)
    retrieved_sources = []
    for idx in results[0]:
        original_idx = valid_indices[idx]
        source = MinimalSource(**sources[original_idx])
        retrieved_sources.append(source)
    return retrieved_sources


def search_dataset(dataset_path: str, k: int = 10, save_directory: str = "data/output/search_results") -> None:
    """Retrieve sources for every question in a dataset."""
    with open(dataset_path) as f:
        dataset = RagDataset.model_validate_json(f.read())
    search_results = []
    for item in dataset.rag_questions:
        retrieved_sources = search(item.question, k)
        result = MinimalSearchResults(
            question_id=item.question_id,
            question=item.question,
            retrieved_sources=retrieved_sources
        )
        search_results.append(result)
    student_results = StudentSearchResults(
        search_results=search_results,
        k=k
    )
    Path(save_directory).mkdir(parents=True, exist_ok=True)
    with open(f"{save_directory}/{Path(dataset_path).name}", "w") as f:
        f.write(student_results.model_dump_json(indent=4))
    print(f"Saved student_search_results: {save_directory}/{Path(dataset_path).name}")


if __name__ == "__main__":
    fire.Fire()

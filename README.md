*This activity has been created as part of the 42 curriculum by fadarwis.*

# Description

**RAG against the machine** is the 2nd project of the 42 Common Core Artificial Intelligence module, it introduces RAG,
or Retrieval-Augmented Generation. Retrieval-Augmented Generation) is retrieving the the most relevant information dynamically
for the user’s query real time. You Hook up the model to a searchable library of documents, and use when you have a large or growing set of information.
 
**Retrieval**: look up and retrieve the most relevant information.

**Augmented**: improve or augment your prompt with the retrieved information.

**Generation**: the LLM generated a response based on the augmented prompt.


What issues did LLMs face before RAG?

- Missing facts which resulted in outdated answers and gaps in knowledge.
- Hallucinations & No citation, Unverifiable claims, risky errors.

# Instructions

## To set up the environment 

```
uv sync
make install
```
## To activate the virtual environment 

```
make activate
```
## To launch the program using python's debugger

```
make debug
```
## To launch the RAG pipeline

```
make run
```
## For flake8 and mypy checks

```
make lint
```
## To remove generated python files and the virtual environment

```
make clean
```


# Resources
1. https://aws.amazon.com/what-is/retrieval-augmented-generation/
2. https://www.youtube.com/watch?v=swvzKSOEluc
3. https://www.youtube.com/watch?v=TW9vHU1GpU4
4. https://vxrl.medium.com/enhancing-llm-code-generation-with-rag-and-ast-based-chunking-5b81902ae9fc
5. https://www.meilisearch.com/blog/rag-indexing
6. https://ai-tldr.dev/learn/rag/chunking-and-ingestion/chunk-code-tables-markdown/

# System architecture

1. Indexing the corpus: Organizing the documents/ data in way that enables the system to quickly locate relevant documents for a search query. The index serves as a map of the knowledge base. (Read files, chunking, tokenizing)

2. Contextualized sparse retrieval using BM25s.

3. Augmentation: passing Query + Retrieved data + Prompt to the LLM

**CLI Commands:** 
    1. index –max_chunk_size <int>
    2. search <query> –k <int>
    3. search_dataset –dataset_path <path> –k <int> –save_directory <dir>
    4. answer <query> –k <int>
    5. answer_dataset –student_search_results_path <path> –save_directory <dir>
    6. evaluate –student_search_results_path <path> –dataset_path <path>
  
# Chunking strategy

**Markdown Chunking**

For Markdown files, the system uses a heading-based chunking strategy.

1. **Heading detection:** Uses regular expressions to identify Markdown headings (levels 1–6).
2. **Section extraction:** Divides the document into sections based on heading positions. Text preceding the first heading is treated as a separate section. Documents without headings are treated as a single section.
3. **Fixed-size chunking:** Splits each section into chunks that do not exceed the configured maximum character size.
4. **Overlapping chunks:** Applies a configurable overlap between consecutive chunks within the same section to preserve contextual continuity.

The chunker returns character-based start and end offsets, allowing the original text to be extracted without modifying the source document.


**Python Code Chunking with Tree-sitter**

For Python files, the system uses Tree-sitter to parse source code into an Abstract Syntax Tree (AST)-like concrete syntax tree.
Rather than splitting Python code at arbitrary character positions, the parser identifies top-level syntactic nodes, such as function definitions, class definitions, and import statements.

The chunking algorithm:

Parses the Python source code using the Tree-sitter Python grammar.
Extracts the top-level named nodes from the syntax tree.
Groups consecutive nodes into chunks while respecting the maximum chunk size.

If an individual node exceeds the maximum size, it is split using the general text chunking strategy.

Converts Tree-sitter's byte offsets into character offsets to maintain compatibility with Python string slicing.
This approach helps preserve meaningful code structures and reduces the likelihood of splitting functions or classes unnecessarily.

# Retrieval method

# Performance analysis

# Design decisions

# Challenges faced

# Example usage

# AI Usage
AI was used to help suggest methods to improve docs recall, explain concepts, and assist with identifying errors and their causes.

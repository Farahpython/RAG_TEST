
.PHONY: install run debug clean activate lint

VENV ?= $(if $(wildcard /goinfre),/goinfre/fadarwis/venvs/RAG,.venv)

install:
	UV_PROJECT_ENVIRONMENT=$(VENV) uv sync
	UV_PROJECT_ENVIRONMENT=$(VENV) uv pip install flake8 mypy
run:
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python -m src index --max_chunk_size 2000
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
	@echo "Docs Evaluation Results 📝"
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python -m src evaluate \
	--student_search_results_path=data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
	--dataset_path=data/datasets/AnsweredQuestions/dataset_docs_public.json
	@echo "Code Evaluation Results 📝"
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python -m src evaluate \
    --student_search_results_path=data/output/search_results/UnansweredQuestions/dataset_code_public.json \
    --dataset_path=data/datasets/AnsweredQuestions/dataset_code_public.json
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions
debug:
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf $(VENV)
activate:
	@echo "source $(VENV)/bin/activate"
lint:
	$(VENV)/bin/flake8 . --exclude=$(VENV),llm_sdk,data
	$(VENV)/bin/mypy . --exclude='($(VENV)|llm_sdk|data)' \
	--warn-return-any \
	--warn-unused-ignores \
	--ignore-missing-imports \
	--disallow-untyped-defs \
	--check-untyped-defs \

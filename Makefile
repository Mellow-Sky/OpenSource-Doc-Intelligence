.PHONY: install install-models install-telemetry dev test test-unit test-integration lint format format-check typecheck precommit migrate migration seed seed-eval ingest eval-dataset evaluate benchmark docker-up docker-down clean

UV ?= uv
COMPOSE ?= docker compose
# ``git rev-parse HEAD`` prints the literal string "HEAD" on an unborn
# repository before returning an error.  ``--verify`` stays silent, so a fresh
# source tree leaves provenance empty instead of injecting an invalid revision.
GIT_COMMIT ?= $(shell git rev-parse --verify HEAD 2>/dev/null)

install:
	$(UV) sync --extra dev

install-models:
	$(UV) sync --extra dev --extra models --extra telemetry

install-telemetry:
	$(UV) sync --extra dev --extra telemetry

dev:
	$(UV) run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	$(UV) run pytest

test-unit:
	$(UV) run pytest tests/unit

test-integration:
	$(UV) run pytest -m integration tests/integration

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

format-check:
	$(UV) run ruff format --check .

typecheck:
	$(UV) run mypy app

precommit:
	$(UV) run pre-commit run --all-files

migrate:
	$(UV) run alembic upgrade head

migration:
	@test -n "$(name)" || (echo "usage: make migration name=describe_change" >&2; exit 2)
	$(UV) run alembic revision --autogenerate -m "$(name)"

seed:
	$(UV) run python scripts/seed_demo_data.py

seed-eval:
	$(UV) run python scripts/seed_demo_data.py --evaluation-fixtures evaluation/datasets/kubernetes_source_catalog.jsonl $(SEED_EVAL_ARGS)

ingest:
	$(UV) run python scripts/ingest_kubernetes.py

eval-dataset:
	$(UV) run python scripts/build_eval_dataset.py

evaluate:
	$(UV) run python scripts/run_evaluation.py --dataset evaluation/datasets/kubernetes_eval.jsonl --output evaluation/reports/latest

benchmark:
	$(UV) run python scripts/benchmark.py $(BENCHMARK_ARGS)

docker-up:
	GIT_COMMIT=$(GIT_COMMIT) $(COMPOSE) up --build -d

docker-down:
	$(COMPOSE) down

clean:
	rm -rf -- .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build

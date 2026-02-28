.PHONY: install lint format test check clean

## install: editable install with dev extras
install:
	pip install -e ".[dev]"

## lint: run ruff linter
lint:
	ruff check .

## format: auto-format with ruff
format:
	ruff format .

## check: lint + format check + tests (mirrors CI)
check: lint
	ruff format --check .
	pytest

## test: run pytest
test:
	pytest

## clean: remove build / cache artefacts
clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

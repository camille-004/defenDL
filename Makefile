.PHONY: install lint format type-check test

install:
	poetry install

lint:
	poetry run ruff check .

format:
	poetry run ruff format .
	poetry run ruff check --fix .

type-check:
	poetry run mypy --config-file pyproject.toml .

test:
	poetry run pytest

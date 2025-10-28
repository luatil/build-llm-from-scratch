data/the-verdict.txt: get_data.py
	mkdir -p data
	uv run get_data.py


test:
	uv run pytest tests/*

format:
	uv run ruff format .

lint:
	uv run ruff check . --fix

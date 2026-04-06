# Contributing

## Setup

```bash
git clone https://github.com/Sahojit/AutonoML.git
cd AutonoML
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Running Tests

```bash
pytest tests/ -q
```

## Code Style

This project uses [ruff](https://github.com/astral-sh/ruff) for linting.

```bash
pip install ruff
ruff check . --fix
```

## Pull Requests

- Keep PRs focused on a single change
- All tests must pass before merging
- Write a clear PR description explaining what and why

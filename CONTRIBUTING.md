# Contributing to DriftWave-Lab

Thank you for your interest in contributing! This document covers the basics.

## Development setup

```bash
# Clone the repo
git clone https://github.com/davidgisbertortiz-arch/driftwave-lab.git
cd driftwave-lab

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Editable install with dev dependencies
pip install -e ".[dev]"
```

## Code quality

We use **ruff** for linting and formatting. Before committing:

```bash
ruff check .          # lint
ruff format .         # auto-format
pytest                # tests
```

Or simply:

```bash
make check
```

## Branch workflow

1. Create a feature branch from `main`.
2. Make small, focused commits.
3. Open a pull request against `main`.
4. Ensure CI is green before requesting review.

## Testing

- All new functionality should include tests in `tests/`.
- Run `pytest` to verify nothing is broken.

## Style

- Follow existing conventions in the codebase.
- Use clear docstrings (Google style preferred).
- Keep modules focused and imports explicit.

## Questions?

Open an issue or reach out via the repository discussions.

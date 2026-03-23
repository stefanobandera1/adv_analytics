# Contributing to adsat

Thank you for your interest in contributing! This document covers everything you need
to get started — from filing bugs to submitting pull requests.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Report a Bug](#how-to-report-a-bug)
- [How to Request a Feature](#how-to-request-a-feature)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Releasing a New Version](#releasing-a-new-version)

---

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/).
Please be respectful and constructive in all interactions.

---

## How to Report a Bug

1. Search [existing issues](https://github.com/stefanobandera1/adsat/issues) first.
2. Open a new issue using the **Bug Report** template.
3. Include a **minimal reproducible example** — the smaller the better.
4. Paste the full traceback and your environment info (`python --version`, OS, adsat version).

---

## How to Request a Feature

1. Open a new issue using the **Feature Request** template.
2. Describe the use case clearly — *why* is this useful, not just *what* it should do.
3. If you're planning to implement it yourself, say so in the issue before starting work.

---

## Development Setup

```bash
# 1. Fork the repository on GitHub, then clone your fork
git clone https://github.com/stefanobandera1/adsat.git
cd adsat

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 4. Install pre-commit hooks
pre-commit install
```

---

## Running Tests

```bash
# Run the full test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=adsat --cov-report=term-missing

# Run a single test file
pytest tests/test_adsat.py -v -k "TestCampaignSaturationAnalyzer"

# Run the end-to-end integration test
python run_end_to_end.py
```

---

## Code Style

This project uses:

- **black** for formatting (`black adsat/ tests/`)
- **ruff** for linting (`ruff check adsat/ tests/`)

Both are enforced by pre-commit hooks. Run manually:

```bash
black adsat/ tests/ examples/
ruff check adsat/ tests/ examples/ --fix
```

Key conventions:
- All public functions and classes must have **docstrings** (NumPy style).
- New modules must be exported from `adsat/__init__.py`.
- Type hints on all public function signatures.
- No bare `except:` clauses — always catch specific exceptions.

---

## Submitting a Pull Request

1. Create a branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes, write/update tests.
3. Ensure tests pass and linting is clean.
4. Update `CHANGELOG.md` under `[Unreleased]`.
5. Push your branch and open a PR against `main`.
6. Fill in the PR template — describe *what* changed and *why*.
7. A maintainer will review within 5 business days.

---

## Releasing a New Version

> This section is for maintainers only.

1. Update `version` in `pyproject.toml` and `adsat/__init__.py`.
2. Move `[Unreleased]` entries in `CHANGELOG.md` to the new version heading.
3. Commit: `git commit -m "chore: release v0.x.y"`
4. Tag: `git tag v0.x.y && git push origin v0.x.y`
5. The GitHub Actions CI workflow will automatically build and publish to PyPI.
   (Or publish manually — see the PyPI publishing guide in `PUBLISHING.md`.)

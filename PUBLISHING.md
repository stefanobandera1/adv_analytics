# Publishing adsat — Step-by-Step Guide

This document covers everything from creating accounts to having your package
live on PyPI and GitHub. Follow the steps in order.

---

## Part 1 — One-Time Account & Tool Setup

### Step 1.1 — Create a PyPI account

1. Go to **https://pypi.org/account/register/**
2. Choose a username, email and password.
3. Verify your email address.
4. Go to **Account Settings → Add 2FA** and enable two-factor authentication
   (PyPI requires this for publishing).

### Step 1.2 — Create a TestPyPI account (for dry runs)

TestPyPI is a sandbox — mistakes here don't matter.

1. Go to **https://test.pypi.org/account/register/**
2. Register with the **same username** (keeps things consistent).
3. Enable 2FA there too.

### Step 1.3 — Install publishing tools locally

```bash
pip install build twine
```

### Step 1.4 — Create a GitHub account (if you don't have one)

1. Go to **https://github.com/join**
2. Choose a username — this becomes part of your package's URL.
3. Verify your email.

---

## Part 2 — Prepare the Package Locally

### Step 2.1 — Update your name and URLs throughout the package

Before publishing, replace all placeholder values:

| File | What to change |
|------|----------------|
| `README.md` | Badge URLs, GitHub links |
| `CHANGELOG.md` | GitHub comparison URLs |
| `CONTRIBUTING.md` | GitHub clone URL |

### Step 2.2 — Verify the version number

`pyproject.toml` and `adsat/__init__.py` must show the **same** version:

```toml
# pyproject.toml
version = "0.2.0"
```

```python
# adsat/__init__.py
__version__ = "0.2.0"
```

Use **Semantic Versioning**: `MAJOR.MINOR.PATCH`
- `PATCH` (0.2.1) — bug fixes
- `MINOR` (0.3.0) — new features, backwards-compatible
- `MAJOR` (1.0.0) — breaking changes

### Step 2.3 — Run the test suite

```bash
pip install -e ".[dev]"
pytest tests/ -v
python run_end_to_end.py
```

All tests must pass before publishing.

### Step 2.4 — Build the distribution files

```bash
python -m build
```

This creates two files inside `dist/`:

```
dist/
  adsat-0.2.0.tar.gz          # source distribution (sdist)
  adsat-0.2.0-py3-none-any.whl  # wheel (binary distribution)
```

### Step 2.5 — Validate the build

```bash
twine check dist/*
```

You must see `PASSED` for both files. Fix any warnings before continuing.

---

## Part 3 — Dry Run on TestPyPI

Always test on TestPyPI first — you cannot delete a release from the real PyPI.

### Step 3.1 — Upload to TestPyPI

```bash
twine upload --repository testpypi dist/*
```

Enter your **TestPyPI** username and password when prompted
(or use an API token — see Step 3.2).

### Step 3.2 — Use an API token (recommended over password)

Tokens are safer than passwords and required for CI.

1. Log in to **https://test.pypi.org**
2. Go to **Account Settings → API tokens → Add API token**
3. Scope: **Entire account** for first publish; **project-scoped** afterwards
4. Copy the token (starts with `pypi-`)

Now upload using the token:

```bash
twine upload --repository testpypi dist/* -u __token__ -p pypi-YOUR_TOKEN_HERE
```

Or store it in `~/.pypirc` (never commit this file — it's in `.gitignore`):

```ini
[testpypi]
  username = __token__
  password = pypi-YOUR_TEST_TOKEN_HERE

[pypi]
  username = __token__
  password = pypi-YOUR_REAL_TOKEN_HERE
```

### Step 3.3 — Verify on TestPyPI

After upload, visit:
```
https://test.pypi.org/project/adsat/
```

Check the description renders correctly, version is right, classifiers look good.

### Step 3.4 — Test install from TestPyPI

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            adsat==0.2.0
python -c "import adsat; print(adsat.__version__)"
```

---

## Part 4 — Publish to Real PyPI

### Step 4.1 — Upload to PyPI

```bash
twine upload dist/*
```

Enter your **PyPI** API token when prompted.

### Step 4.2 — Verify on PyPI

Visit: **https://pypi.org/project/adsat/**

### Step 4.3 — Test the real install

```bash
pip install adsat==0.2.0
python -c "from adsat import predict_saturation_per_campaign; print('OK')"
```

---

## Part 5 — Set Up the GitHub Repository

### Step 5.1 — Create the repository

1. Log in to GitHub and click **New repository**
2. Name: `adsat`
3. Description: `Advertising Saturation Analysis Toolkit`
4. Set to **Public**
5. Do **NOT** tick "Add README" — you already have one
6. Click **Create repository**

### Step 5.2 — Push your code

From inside the `adsat/` project folder:

```bash
# Initialise git (if not already done)
git init
git branch -M main

# Stage everything
git add .
git commit -m "feat: initial release v0.2.0"

# Add the remote and push
git remote add origin https://github.com/stefanobandera1/adsat.git
git push -u origin main
```

### Step 5.3 — Add a version tag

Tags trigger the automatic PyPI publish workflow (see Part 6).

```bash
git tag v0.2.0
git push origin v0.2.0
```

### Step 5.4 — Create a GitHub Release

1. Go to your repo → **Releases → Create a new release**
2. Tag: `v0.2.0`
3. Title: `v0.2.0 — Initial Release`
4. Paste the relevant section from `CHANGELOG.md` into the description
5. Click **Publish release**

---

## Part 6 — Automated CI/CD with GitHub Actions

The `.github/workflows/` directory contains two workflows already configured:

### `ci.yml` — runs on every push and pull request

- Tests on Python 3.9, 3.10, 3.11, 3.12
- Tests on Ubuntu, Windows, macOS
- Runs linting (ruff) and formatting check (black)
- Uploads coverage to Codecov (optional)

### `publish.yml` — publishes to PyPI on version tags

This uses **PyPI Trusted Publishing** (OIDC) — no tokens stored in GitHub Secrets.

**One-time setup:**

1. Go to **https://pypi.org/manage/project/adsat/settings/publishing/**
2. Click **Add a new publisher**
3. Fill in:
   - Publisher: **GitHub Actions**
   - Owner: `YOUR_GITHUB_USERNAME`
   - Repository: `adsat`
   - Workflow: `publish.yml`
   - Environment: `pypi`
4. Save

5. In your GitHub repository, go to **Settings → Environments**
6. Create an environment called `pypi`
7. (Optionally) add a required reviewer for extra safety

**To trigger a release:**

```bash
# Bump version in pyproject.toml and __init__.py, commit, then:
git tag v0.3.0
git push origin v0.3.0
```

GitHub Actions will automatically build and publish to PyPI.

---

## Part 7 — Ongoing Maintenance

### Releasing a new version checklist

```
[ ] Update version in pyproject.toml
[ ] Update __version__ in adsat/__init__.py
[ ] Move [Unreleased] entries to new version in CHANGELOG.md
[ ] Run: pytest tests/ -v && python run_end_to_end.py
[ ] Run: python -m build && twine check dist/*
[ ] Upload to TestPyPI and verify
[ ] git add . && git commit -m "chore: release vX.Y.Z"
[ ] git tag vX.Y.Z && git push origin main --tags
[ ] Create GitHub Release with changelog notes
[ ] Verify package on https://pypi.org/project/adsat/
```

### Useful commands reference

```bash
# Build
python -m build

# Validate
twine check dist/*

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ adsat

# Check what's in your wheel
unzip -l dist/adsat-*.whl

# Clean build artifacts
rm -rf dist/ build/ *.egg-info/
```

---

## Part 8 — Optional: Badges & Community

Once published, add real badge URLs to your README:

```markdown
[![PyPI version](https://img.shields.io/pypi/v/adsat.svg)](https://pypi.org/project/adsat/)
[![Downloads](https://img.shields.io/pypi/dm/adsat)](https://pypi.org/project/adsat/)
[![CI](https://github.com/stefanobandera1/adsat/actions/workflows/ci.yml/badge.svg)](...)
[![codecov](https://codecov.io/gh/stefanobandera1/adsat/branch/main/graph/badge.svg)](...)
```

Consider also:
- **Codecov** (https://codecov.io) — free coverage tracking
- **ReadTheDocs** (https://readthedocs.org) — free hosted documentation
- **PyPI Stats** (https://pypistats.org/packages/adsat) — download analytics

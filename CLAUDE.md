# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install all workspace packages with dev dependencies
uv sync --all-packages --group dev

# Run tests
uv run pytest

# Run a single test file
uv run pytest src/budget/tests/test_next_payment_calculator.py

# Type checking (matches the CI job's underlying check on src/, minus Pants' change-scoping)
uv run mypy src/

# Run a specific app
uv run python -m <module>

# Add a dependency to a workspace package
uv add <package> --package <workspace-member>
```

## Architecture

This is a **UV workspace** with two layers:

### Shared Libraries (`src/`)
Reusable packages imported by apps. All use the `wiz.*` namespace:

- `wiz.shared` — ML estimators (XGBoost, LightGBM, PyTorch, sklearn wrappers), preprocessors, target transformers
- `wiz.interface` — Abstract base classes (`EstimatorInterface`, `ModelingInterface`, `PreprocessorInterface`, `TargetInterface`) that shared estimators implement
- `wiz.evaluation` — ML evaluation metrics and helpers
- `wiz.budget` — Payment date logic, schemas, and Google Sheets data loading
- `wiz.job_app_backend` — spaCy NER model for extracting skills (TECH/TOOL/DOMAIN/SOFT) from job postings; also has an Ollama-based LLM alternative

### Apps (`apps/`)
Self-contained applications, each with their own `pyproject.toml`:

- **tools-app** — Streamlit multi-page hub that surfaces budget, job-app, power usage, transcription, and transport tax deduction tools
- **house-prices** — Streamlit app for real estate price prediction using `wiz.shared`/`wiz.interface`
- **churn** — Customer churn prediction (XGBoost/sklearn)
- **finance** — Financial data analysis with yfinance + polars
- **stock-analyzer** — Stock analysis with Jupyter notebooks
- **weather-data-db** — ETL pipeline writing weather data to MSSQL via pyspark/delta-spark

### Workspace Configuration
- Root `pyproject.toml` declares workspace members and mypy config
- `mypy_path` includes all `src/` packages so cross-package imports type-check correctly
- CI type-checks on every push/PR via Pants, not directly via `uv run mypy src/`: the `mypy (src)` job runs `pants check src::` (full) or `pants --changed-since=origin/main --changed-dependents=transitive --tag='-app' check` (PR, scoped to what changed), and each affected app is checked separately by its own `mypy (apps/<app>)` job running `uv run mypy --explicit-package-bases .` in that app's own environment — see `.github/workflows/ci.yml`. Type errors in either block merge.
- Python 3.13 (`.python-version`)

### Namespace Packages
All `src/` packages use implicit namespace packages under `wiz/`. There are no `__init__.py` files at the `wiz/` level — only at the sub-package level (e.g., `wiz/budget/__init__.py`). Mypy is configured with `namespace_packages = true` and `explicit_package_bases = true` to handle this.

## Optional: keep the graphify knowledge graph fresh

This repo can be explored as a knowledge graph via the `/graphify` skill (code, docs,
architecture, cross-app relationships). The graph itself (`graphify-out/`) is a local,
gitignored build artifact — nothing here is required to work on the repo.

To have it auto-rebuild after every commit and branch switch (code changes only, no
LLM tokens spent — doc/image changes still need a manual `/graphify --update`), install
the graphify CLI once per clone:

```bash
uv tool install graphifyy
graphify hook install
```

This writes a post-commit + post-checkout hook local to your clone (`.git/hooks/`,
never shared via git). Check with `graphify hook status`, remove with `graphify hook
uninstall`.

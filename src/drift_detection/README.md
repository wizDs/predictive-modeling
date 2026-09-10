# drift_detection

Experiments with [alibi-detect](https://docs.seldon.io/projects/alibi-detect/) drift
detectors.

This is a standalone `uv` project (not a workspace member of the repo root) because
`alibi-detect` pins `numpy<2.0.0`, which conflicts with the `numpy>=2` used elsewhere in
the repo. It's also pinned to Python 3.12 (`.python-version`) since `alibi-detect`'s
`numba` dependency doesn't yet support 3.13.

## Setup

```bash
cd src/drift_detection
uv sync
uv run python -m ipykernel install --user --name drift-detection --display-name drift-detection
```

`uv sync` downloads Python 3.12 automatically if it isn't already installed. The
`ipykernel install` step registers the kernel so Jupyter/VS Code can find it (only needed
once).

## Notebooks

- `notebooks/chi_square_drift.ipynb` — categorical feature drift on the Titanic dataset
  using `ChiSquareDrift`, comparing a held-out split with the same distribution as the
  reference set against a deliberately-shifted split (survivors only).

Select the `drift-detection` kernel when running the notebook in Jupyter/VS Code.

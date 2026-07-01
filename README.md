# BiHPR

BiHPR is a Python implementation of **Bidirectional Homogeneity Pursuit in
High-Dimensional Regression**. It estimates individualized regression
coefficients while recovering sample subgroups, feature clusters, and sparse
active features.

The package provides:

- a large-scale MCP-ADMM solver for one fixed penalty setting;
- a warm-started two-dimensional BIC tuning path;
- simulation utilities for the Gaussian paper-style design;
- a command-line script for multi-dimensional Gaussian grid experiments;
- a worked notebook in `examples/BiHPR_usage_demo.ipynb`.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone <your-github-repository-url>
cd Regression_biclustering_github
pip install -e .
```

Install notebook dependencies when you want to run the example notebook:

```bash
pip install -e ".[examples]"
```

For development tools:

```bash
pip install -e ".[dev]"
```

## Quick Start

The fastest sanity check is a small paper-style simulation:

```python
from BiHPR import run_single_simulation

run = run_single_simulation(
    n=30,
    p=60,
    p0=15,
    lambda_grid=(2000.0, 1000.0),
    lambda3_grid=(1000.0, 500.0),
    niter=50,
    pilot_niter=50,
    tol=1e-3,
    cluster_tol=1e-3,
)

print(run["best_record"])
print(run["metrics"])
```

For larger experiments, increase `niter`, `pilot_niter`, and the tuning grids.

## Main API

```python
from BiHPR import (
    fit_bihpr_mcp_large,
    fit_bihpr_path,
    generate_paper_simulation,
    prepare_bihpr_workspace,
    run_single_simulation,
)
```

### `fit_bihpr_mcp_large`

Fits one fixed penalty setting:

```python
result = fit_bihpr_mcp_large(
    Y,
    X,
    lambda_col=2000.0,
    lambda3=500.0,
    niter=200,
    tol=1e-4,
)
```

Important returned fields:

- `beta`: estimated `n x p` coefficient matrix;
- `row_labels`: estimated sample subgroup labels;
- `col_labels`: estimated feature cluster labels;
- `active_features`: Boolean active-feature mask;
- `rss`: residual sum of squares;
- `objective`: objective value at the final iterate;
- `iterations`, `converged`, `final_change`, `primal_residual`: convergence
  diagnostics;
- `timings`: graph precomputation and solver runtime;
- `memory_bytes`: estimated memory footprint of the current implementation.

### `fit_bihpr_path`

Runs a two-dimensional BIC grid with warm starts:

```python
path = fit_bihpr_path(
    Y,
    X,
    lambda_grid=(4000.0, 2000.0, 1000.0),
    lambda3_grid=(1000.0, 500.0, 100.0),
    niter=200,
    pilot_niter=200,
    tol=1e-4,
)

best = path["best"]
best_record = path["best_record"]
records = path["records"]
```

Returned objects:

- `best`: full solver result at the selected BIC minimum;
- `best_record`: compact row describing the selected tuning point;
- `records`: list of all BIC grid rows;
- `results`: dictionary mapping `(lambda_col, lambda3)` to full solver results;
- `pilots`: pilot-path results with `lambda3=0`;
- `workspace`: reusable graph and Sylvester precomputation object.

### `generate_paper_simulation`

Creates a balanced simulation design with three row clusters, three active
feature clusters, and one zero feature cluster:

```python
truth = generate_paper_simulation(n=150, p=400, p0=30, sigma=0.5, seed=2026)

Y = truth["Y"]
X = truth["X"]
beta_true = truth["beta_true"]
```

## Full Example

```python
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from BiHPR import generate_paper_simulation, fit_bihpr_path

truth = generate_paper_simulation(n=30, p=60, p0=15, sigma=0.5, seed=2026)

path = fit_bihpr_path(
    truth["Y"],
    truth["X"],
    lambda_grid=(4000.0, 2000.0, 1000.0),
    lambda3_grid=(1000.0, 500.0, 100.0),
    niter=100,
    pilot_niter=100,
    tol=1e-3,
    cluster_tol=1e-3,
)

best = path["best"]
record_table = pd.DataFrame(path["records"])

row_ari = adjusted_rand_score(truth["row_labels"], best["row_labels"])
active_mask = truth["active_features"]
col_ari = adjusted_rand_score(
    truth["col_labels"][active_mask],
    best["col_labels"][active_mask],
)

print(path["best_record"])
print({"row_ari": row_ari, "active_col_ari": col_ari})
print(record_table.sort_values("bic").head())
```

## Command-Line Gaussian Grid

After installation, the Gaussian grid experiment is available as:

```bash
bihpr-gaussian-grid --n-jobs 3
```

Useful options:

```bash
# Run a coarse pilot grid
bihpr-gaussian-grid --grid pilot --p-values 100 300 --repeats 5 --n-jobs 2

# Run only selected seeds
bihpr-gaussian-grid --seeds 0 1 2 --p-values 100 --n-jobs 3

# Rerun only previously failed seeds
bihpr-gaussian-grid --failed-only --n-jobs 3

# Save outputs to a custom directory
bihpr-gaussian-grid --result-dir my_results --n-jobs 3
```

By default, outputs are written to `Gaussian_noise05_results/` in the current
working directory. The script stores per-seed candidate tables and final summary
tables for the BIC and target-df selection rules.

## Notebook Tutorial

Open the worked example notebook:

```bash
jupyter notebook examples/BiHPR_usage_demo.ipynb
```

The notebook demonstrates:

1. importing the package;
2. generating a small simulation dataset;
3. fitting a single penalty setting;
4. running a BIC tuning path;
5. computing recovery metrics;
6. inspecting the tuning table.

## Repository Layout

```text
BiHPR/
    __init__.py              # public package API
    BiHPR_MCP_large.py       # MCP-ADMM solver
    BiHPR_single_test.py     # BIC path and paper-style simulation helpers
    Cluster.py               # graph and label utilities
    Weight.py                # pairwise kernel-weight utilities
examples/
    BiHPR_usage_demo.ipynb   # tutorial notebook
run_gaussian_noise05_grid.py # command-line Gaussian grid experiment
pyproject.toml               # package metadata
README.md                    # this tutorial
```

## GitHub Upload Checklist

Before pushing to GitHub:

```bash
python -m compileall -q .
python -m pip install -e . --no-deps --dry-run
python run_gaussian_noise05_grid.py --help
```

Then commit the source files:

```bash
git add BiHPR examples README.md pyproject.toml .gitignore run_gaussian_noise05_grid.py
git commit -m "Package BiHPR solver with docs and examples"
git push
```

# BiHPR

BiHPR implements bidirectional homogeneity pursuit for high-dimensional regression. It jointly estimates sample subgroups, covariate-effect groups, and inactive covariates.

## Installation

Clone the repository and install the dependencies needed for the notebook you plan to run.

## Complete tuning demo

The paper-style simulation demo is:

```bash
pip install -e ".[examples]"
jupyter notebook examples/BiHPR_usage_demo.ipynb
```

The notebook runs the complete warm-started BIC tuning path for one fixed simulation setting. It reads its initialization from `examples/data/paper_simulation/initialization.csv` and saves the tuning table and summary under `outputs/BiHPR_usage_demo_n150_p400_seed2026/`.

## TCGA lung real-data analysis

The real-data analysis is:

```bash
pip install -e ".[realdata]"
jupyter notebook examples/TCGA_lung_real_data.ipynb
```

Start Jupyter from the repository root. The notebook reads `LUNG_data.csv` and `lung_coef.xlsx` from `examples/data/tcga_lung/`, fits BiHPR to the selected genes, and saves the final results under `outputs/tcga_lung_real_data/`.


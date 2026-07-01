"""BiHPR: bidirectional homogeneity pursuit for high-dimensional regression.

The package exposes the large-scale MCP-ADMM solver, the BIC tuning path, and
paper-style Gaussian simulation helpers. Most users should start with
``fit_bihpr_path`` for tuning or ``fit_bihpr_mcp_large`` for fitting a single
penalty setting.
"""

from __future__ import annotations

from BiHPR.BiHPR_MCP_large import (
    BiHPRWorkspace,
    adaptive_feature_weights,
    bihpr_objective,
    estimate_legacy_state_bytes,
    extract_clusters,
    fit_bihpr_mcp_large,
    fit_bihpr_path,
    prepare_bihpr_workspace,
    run_single_simulation,
    weighted_mcp_prox,
)
from BiHPR.BiHPR_single_test import (
    generate_paper_simulation,
    label_beta_by_threshold,
)

__version__ = "0.1.0"

__all__ = [
    "BiHPRWorkspace",
    "__version__",
    "adaptive_feature_weights",
    "bihpr_objective",
    "estimate_legacy_state_bytes",
    "extract_clusters",
    "fit_bihpr_mcp_large",
    "fit_bihpr_path",
    "generate_paper_simulation",
    "label_beta_by_threshold",
    "prepare_bihpr_workspace",
    "run_single_simulation",
    "weighted_mcp_prox",
]

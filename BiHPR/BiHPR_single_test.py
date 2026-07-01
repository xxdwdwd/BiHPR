"""BIC tuning paths and paper-style simulations for the BiHPR solver.

The numerical ADMM solver lives in :mod:`BiHPR.BiHPR_MCP_large`. This module
adds the experiment-level workflow used by the Gaussian simulations:

1. run a pilot path with ``lambda3=0``;
2. compute adaptive feature-selection weights from the pilot solutions;
3. evaluate the two-dimensional ``(lambda_col, lambda3)`` BIC grid;
4. generate and evaluate one balanced paper-style simulation.
"""

from __future__ import annotations

import math
import time

import numpy as np
from sklearn.metrics import adjusted_rand_score

from BiHPR.Cluster import (
    compute_difference_matrix,
    find_clusters,
    label_columns_by_threshold,
    similarity_to_adjacency,
)
from BiHPR.BiHPR_MCP_large import (
    adaptive_feature_weights,
    fit_bihpr_mcp_large,
    prepare_bihpr_workspace,
)

# Wide tuning grids used for the p=400 paper-style simulation. The point
# lambda_col=200000 and lambda3=500000 is a known recovery setting.
STRUCTURE_LAMBDA_GRID = (
    400000.0, 300000.0, 200000.0, 150000.0,
    100000.0, 70000.0, 50000.0, 30000.0,
)
SPARSITY_LAMBDA_GRID = (
    2000000.0, 1000000.0, 500000.0, 200000.0,
    100000.0, 50000.0, 20000.0,
)


def label_beta_by_threshold(
    beta,
    *,
    feature_threshold=0.9,
    cluster_threshold=10.0,
):
    """Assign row, column, and active-feature labels by fixed thresholds.

    This post-processing rule mirrors the original Gaussian notebook:

    1. A feature is marked inactive when at least ``feature_threshold`` of its
       coefficients lie in ``[-0.1, 0.1)``; otherwise it is active.
    2. Euclidean distances are computed separately between rows and columns of
       ``beta``.
    3. Nodes whose distance is no larger than ``cluster_threshold`` are
       connected, and graph connected components become the labels.
    """

    beta = np.asarray(beta, dtype=float)
    active_features = np.asarray(
        label_columns_by_threshold(beta, threshold=feature_threshold),
        dtype=bool,
    )

    row_distance = compute_difference_matrix(beta, metric="euclidean")
    row_adjacency = similarity_to_adjacency(
        row_distance,
        threshold=cluster_threshold,
        similarity_measure="difference",
    )
    row_result = find_clusters(row_adjacency)

    col_distance = compute_difference_matrix(beta.T, metric="euclidean")
    col_adjacency = similarity_to_adjacency(
        col_distance,
        threshold=cluster_threshold,
        similarity_measure="difference",
    )
    col_result = find_clusters(col_adjacency)

    return {
        "active_features": active_features,
        "row_labels": row_result["cluster"],
        "col_labels": col_result["cluster"],
        "row_clusters": row_result["n_clusters"],
        "col_clusters": col_result["n_clusters"],
    }


def _apply_threshold_labels(
    result,
    *,
    feature_threshold=0.9,
    cluster_threshold=10.0,
):
    """Replace solver graph labels with the fixed-threshold labels."""

    labels = label_beta_by_threshold(
        result["beta"],
        feature_threshold=feature_threshold,
        cluster_threshold=cluster_threshold,
    )
    result.update(labels)
    return result


def _bic(result):
    n = result["beta"].shape[0]
    K_row = result["row_clusters"]
    # Match the original Gaussian simulation convention: df is the number of
    # row clusters times the number of column clusters, with zero features
    # counted as one column cluster.
    K_col = result["col_clusters"]
    df = K_row * K_col
    value = math.log(max(result["rss"] / n, np.finfo(float).tiny))
    value += math.log(n) / n * df
    return value, K_row, K_col, df


def fit_bihpr_path(
    Y,
    X,
    lambda_grid,
    lambda3_grid,
    *,
    gamma_mcp=3.0,
    nu0=1.0,
    nu1=1.0,
    nu2=1.0,
    nu3=1.0,
    m=5,
    phi=None,
    niter=500,
    pilot_niter=None,
    tol=1e-5,
    cluster_tol=1e-6,
    feature_threshold=0.9,
    label_cluster_threshold=10.0,
    output=0,
    beta_init=None,
):
    """Evaluate a two-dimensional BIC grid with full warm starts.

    The path first fits pilot models with ``lambda3=0``. Each pilot solution
    yields adaptive feature weights, which are then reused while searching the
    sparsity grid for the same structural penalty.
    """

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    lambda_grid = sorted(set(map(float, lambda_grid)), reverse=True)
    lambda3_grid = sorted(set(map(float, lambda3_grid)), reverse=True)
    pilot_niter = niter if pilot_niter is None else pilot_niter

    # In the Gaussian simulations, X is independent of the true latent groups.
    # When beta_init is supplied, use the noisy truth-based initializer to build
    # the fixed KNN graph, matching the historical regression implementation.
    graph_data = X if beta_init is None else beta_init
    work = prepare_bihpr_workspace(
        Y, X,
        nu0=nu0, nu1=nu1, nu2=nu2, nu3=nu3,
        m=m, phi=phi, graph_data=graph_data,
    )

    # Step 1: pilot models with lambda3=0.
    pilots = {}
    warm = None
    for lam in lambda_grid:
        pilots[lam] = fit_bihpr_mcp_large(
            Y, X,
            lambda_col=lam,
            lambda3=0.0,
            feature_weights=np.ones(X.shape[1]),
            beta_init=beta_init if warm is None else None,
            init=warm,
            workspace=work,
            gamma_mcp=gamma_mcp,
            nu0=nu0, nu1=nu1, nu2=nu2, nu3=nu3,
            niter=pilot_niter,
            tol=tol,
            cluster_tol=cluster_tol,
            output=output,
        )
        warm = pilots[lam]["state"]

    # Step 2: search lambda3 with adaptive weights from the pilot model.
    records, results = [], {}
    warm_for_lambda3 = {}
    for lam in lambda_grid:
        u = adaptive_feature_weights(pilots[lam]["beta"])
        warm_within_lambda = pilots[lam]["state"]

        for lam3 in lambda3_grid:
            warm = warm_for_lambda3.get(lam3, warm_within_lambda)
            result = fit_bihpr_mcp_large(
                Y, X,
                lambda_col=lam,
                lambda3=lam3,
                feature_weights=u,
                init=warm,
                workspace=work,
                gamma_mcp=gamma_mcp,
                nu0=nu0, nu1=nu1, nu2=nu2, nu3=nu3,
                niter=niter,
                tol=tol,
                cluster_tol=cluster_tol,
                output=output,
            )
            # Match the original Gaussian notebook by labeling the final beta
            # through a fixed-threshold distance graph, rather than using the
            # ADMM auxiliary variables v and z.
            result = _apply_threshold_labels(
                result,
                feature_threshold=feature_threshold,
                cluster_threshold=label_cluster_threshold,
            )
            bic, K_row, K_col, df = _bic(result)
            results[(lam, lam3)] = result
            records.append({
                "lambda": lam,
                "lambda_row": result["lambda_row"],
                "lambda3": lam3,
                "bic": bic,
                "k_row": K_row,
                "k_col_active": K_col,
                "df": df,
                "n_active": int(result["active_features"].sum()),
                "rss": result["rss"],
                "iterations": result["iterations"],
                "converged": result["converged"],
                "fit_seconds": result["timings"]["fit"],
            })
            warm_within_lambda = result["state"]
            warm_for_lambda3[lam3] = result["state"]

    best_record = min(records, key=lambda row: row["bic"])
    best = results[(best_record["lambda"], best_record["lambda3"])]
    return {
        "best": best,
        "best_record": best_record,
        "records": records,
        "results": results,
        "pilots": pilots,
        "workspace": work,
    }


def generate_paper_simulation(n=150, p=400, p0=30, sigma=0.5, seed=2026):
    """Generate the balanced three-row-cluster, four-column-cluster design.

    The first ``p0`` features are split into three active feature clusters. All
    remaining features form the zero column cluster. The returned dictionary
    contains the design matrix, response vector, true coefficient matrix, and
    ground-truth row, column, and activity labels.
    """

    rng = np.random.default_rng(seed)
    row_labels = np.repeat(np.arange(3), n // 3)
    active_col_labels = np.repeat(np.arange(3), p0 // 3)
    col_labels = np.r_[active_col_labels, np.full(p - p0, 3)]
    block = np.array([[1, -2, 0], [0, 2, -2], [-3, 0, 3]], dtype=float)

    beta_true = np.zeros((n, p))
    for r in range(3):
        for c in range(3):
            beta_true[np.ix_(row_labels == r, active_col_labels == c)] = block[r, c]

    X = rng.standard_normal((n, p))
    sigma_group = np.repeat(float(sigma), 3) if np.isscalar(sigma) else np.asarray(sigma)
    Y = np.einsum("ij,ij->i", X, beta_true)
    Y += rng.normal(scale=sigma_group[row_labels])
    return {
        "Y": Y,
        "X": X,
        "beta_true": beta_true,
        "row_labels": row_labels,
        "col_labels": col_labels,
        "active_features": np.arange(p) < p0,
    }


def run_single_simulation(
    *,
    n=150,
    p=400,
    p0=30,
    lambda_grid=STRUCTURE_LAMBDA_GRID,
    lambda3_grid=SPARSITY_LAMBDA_GRID,
    seed=2026,
    init_noise_sd=1.5,
    niter=1500,
    pilot_niter=1500,
    tol=5e-4,
    cluster_tol=5e-4,
    output=0,
):
    """Run one paper-style simulation and return full recovery diagnostics."""

    truth = generate_paper_simulation(n=n, p=p, p0=p0, seed=seed)
    # Nonconvex MCP is initialization-sensitive. The simulation protocol uses
    # the true beta plus Gaussian noise:
    #
    #     beta^(0) = beta* + N(0, init_noise_sd^2)
    #
    # Use an independent random stream so the generated X and Y stay unchanged.
    init_rng = np.random.default_rng(seed + 10000)
    beta_init = truth["beta_true"] + init_rng.normal(
        0.0, init_noise_sd, size=truth["beta_true"].shape
    )
    start = time.perf_counter()
    path = fit_bihpr_path(
        truth["Y"], truth["X"], lambda_grid, lambda3_grid,
        niter=niter, pilot_niter=pilot_niter,
        tol=tol, cluster_tol=cluster_tol, output=output,
        beta_init=beta_init,
    )
    best = path["best"]
    true_active = truth["active_features"]
    estimated_active = best["active_features"]
    prediction = np.einsum("ij,ij->i", truth["X"], best["beta"])
    metrics = {
        "row_ari": adjusted_rand_score(truth["row_labels"], best["row_labels"]),
        "col_ari_active": adjusted_rand_score(
            truth["col_labels"][true_active], best["col_labels"][true_active]
        ),
        "bias": np.mean(np.abs(best["beta"] - truth["beta_true"])),
        "rmse": np.sqrt(np.mean((truth["Y"] - prediction) ** 2)),
        "fpr": np.mean(estimated_active[~true_active]),
        "fnr": np.mean(~estimated_active[true_active]),
        "n_active": int(estimated_active.sum()),
    }
    recovery_records = []
    for record in path["records"]:
        result = path["results"][(record["lambda"], record["lambda3"])]
        active = result["active_features"]
        recovery_records.append({
            **record,
            "row_ari": adjusted_rand_score(
                truth["row_labels"], result["row_labels"]
            ),
            "col_ari_active": adjusted_rand_score(
                truth["col_labels"][true_active],
                result["col_labels"][true_active],
            ),
            "fpr": np.mean(active[~true_active]),
            "fnr": np.mean(~active[true_active]),
            "k_col_total": len(np.unique(result["col_labels"])),
        })
    return {
        "truth": truth,
        "beta_init": beta_init,
        "path": path,
        "best": best,
        "best_record": path["best_record"],
        "metrics": metrics,
        "recovery_records": recovery_records,
        "elapsed_seconds": time.perf_counter() - start,
        "memory_reduction": best["legacy_memory_bytes"] / best["memory_bytes"],
    }


if __name__ == "__main__":
    # By default, verify the known recovery point instead of running the full
    # wide grid. Remove the explicit grids below to search the full range.
    run = run_single_simulation(
        lambda_grid=(200000.0,),
        lambda3_grid=(500000.0,),
    )
    print("best:", run["best_record"])
    print("metrics:", run["metrics"])
    print("recovery:", run["recovery_records"][0])
    print(f"time: {run['elapsed_seconds']:.2f}s")
    print(f"memory reduction: {run['memory_reduction']:.1f}x")

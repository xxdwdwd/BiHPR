"""
Large-scale BiHPR MCP-ADMM solver.
==================================

BiHPR estimates individualized regression coefficients while simultaneously
recovering sample subgroups, feature clusters, and sparse active features. For
sample i, the model is

    y_i = x_i^T beta_i + error_i,

where all sample-specific coefficient vectors form the n x p matrix beta. The
method encourages three structures in beta:

1. Row homogeneity: samples with identical coefficient vectors are assigned to
   the same latent subgroup.
2. Column homogeneity: features with identical coefficient profiles across
   samples are assigned to the same feature cluster.
3. Feature sparsity: columns with zero coefficient profiles are treated as
   inactive variables.

The solver uses weighted group MCP penalties on row differences, column
differences, and column norms. Row and column penalties are applied on KNN
graphs, which encode the local similarity structure used for fusion. Feature
selection is handled through adaptive column weights computed from pilot
solutions.

The ADMM formulation introduces a structured coefficient copy A and auxiliary
variables for each constraint group:

    beta = A
    v_l  = A[i2, :] - A[i1, :]
    z_k  = A[:, j2] - A[:, j1]
    g    = A

Each iteration alternates between an explicit Sylvester update for A,
closed-form MCP proximal updates for v, z, and g, a sample-wise closed-form
update for beta, and dual-variable updates. The returned result contains the
coefficient estimate, cluster labels, active-feature labels, convergence
diagnostics, objective value, timing information, and memory estimates.

Common entry points:

    fit_bihpr_mcp_large(...)   # fit one penalty setting
    prepare_bihpr_workspace(...)  # precompute graph and Sylvester quantities
    fit_bihpr_path(...)        # run a two-dimensional BIC tuning grid
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# =============================================================================
# 1. Fixed workspace: compute once for the same data and nu values.
# =============================================================================


@dataclass(frozen=True)
class BiHPRWorkspace:
    """Quantities that remain fixed along an ADMM parameter path.

    Dimensions
    ----------
    X               : (n, p)
    row_pairs       : (L_row, 2), nonzero row-fusion edges
    col_pairs       : (L_col, 2), nonzero column-fusion edges
    E_row           : (n, L_row), columns are e_i2 - e_i1
    E_col           : (p, L_col), columns are e_j2 - e_j1
    eigen_denom     : (n, p), denominator of the explicit Sylvester solution
    """

    X: np.ndarray
    Y: np.ndarray
    row_pairs: np.ndarray
    col_pairs: np.ndarray
    row_weights: np.ndarray
    col_weights: np.ndarray
    E_row: csr_matrix
    E_col: csr_matrix
    eigvals_M: np.ndarray
    eigvecs_M: np.ndarray
    eigvals_N: np.ndarray
    eigvecs_N: np.ndarray
    eigen_denom: np.ndarray
    x_norm_sq: np.ndarray
    timings: dict


def _as_arrays(Y, X):
    """Convert inputs to the float64 arrays used by the algorithm."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64).reshape(-1)
    return Y, X


def _knn_edges(data, m=5, phi=0.5):
    """Compute nonzero Gaussian KNN edges.

    Parameters
    ----------
    data : (q, d)
        q objects to compare, one object per row.
        Row weights use X; column weights use X.T.
    m : int
        Number of nearest neighbors kept for each object.
    phi : float
        Gaussian kernel parameter in exp(-phi * distance^2).

    Returns
    -------
    pairs : (L, 2)
        Edges in the undirected union KNN graph, with pairs[:,0] < pairs[:,1].
    weights : (L,)
        Gaussian weights for the corresponding edges.

    Notes
    -----
    In high dimensions, exp(-phi*d^2) can underflow to all zeros. Subtracting
    the minimum active edge distance only multiplies all weights by the same
    constant, and the weights are normalized later.
    """

    data = np.asarray(data, dtype=np.float64)
    q = data.shape[0]
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a^T b
    norm_sq = np.einsum("ij,ij->i", data, data)
    dist_sq = norm_sq[:, None] + norm_sq[None, :] - 2.0 * data @ data.T
    np.maximum(dist_sq, 0.0, out=dist_sq)
    np.fill_diagonal(dist_sq, np.inf)

    k = min(int(m), q - 1)
    neighbors = np.argpartition(dist_sq, kth=k - 1, axis=1)[:, :k]

    # Use the undirected union graph: i is j's neighbor or j is i's neighbor.
    knn_mask = np.zeros((q, q), dtype=bool)
    knn_mask[np.arange(q)[:, None], neighbors] = True
    knn_mask |= knn_mask.T
    np.fill_diagonal(knn_mask, False)

    i1, i2 = np.where(np.triu(knn_mask, k=1))
    pairs = np.column_stack((i1, i2)).astype(np.int64, copy=False)
    active_dist = dist_sq[i1, i2]
    weights = np.exp(-phi * (active_dist - active_dist.min()))
    return pairs, weights


def _incidence_matrix(size, pairs):
    """Build the graph incidence matrix E.

    For edge l=(i1,i2), column l of E is e_i2-e_i1. Therefore:

        E.T @ A = A[i2,:] - A[i1,:]

    This definition fixes the signs of v, z, and lambda later in the ADMM
    updates, so it is the first convention to check when auditing the algorithm.
    """

    pairs = np.asarray(pairs, dtype=np.int64)
    L = len(pairs)
    rows = np.column_stack((pairs[:, 0], pairs[:, 1])).reshape(-1)
    cols = np.repeat(np.arange(L), 2)
    values = np.tile([-1.0, 1.0], L)
    return csr_matrix((values, (rows, cols)), shape=(size, L))


def _adaptive_phi(row_data, col_data):
    """Median heuristic that makes typical row and column kernel weights near 0.5."""

    def median_distance(data):
        norm_sq = np.einsum("ij,ij->i", data, data)
        dist_sq = norm_sq[:, None] + norm_sq[None, :] - 2.0 * data @ data.T
        distances = dist_sq[np.triu_indices(len(data), k=1)]
        positive = distances[distances > 1e-12]
        return np.median(positive) if len(positive) else 1.0

    phi_row = math.log(2.0) / median_distance(row_data)
    phi_col = math.log(2.0) / median_distance(col_data)
    return 0.5 * (phi_row + phi_col)


def prepare_bihpr_workspace(
    Y,
    X,
    *,
    nu0=1.0,
    nu1=1.0,
    nu2=1.0,
    nu3=1.0,
    m=5,
    phi=None,
    graph_data=None,
):
    """Compute fixed weights, graph Laplacians, and Sylvester eigendecompositions.

    The paper normalizes weights as

        sum(row_weights) = 1/sqrt(p)
        sum(col_weights) = 1/sqrt(n)

    The A update solves the Sylvester equation

        M A + A N = H

    where

        M = nu0 I_n + nu1 E_row E_row^T
        N = nu2 E_col E_col^T + nu3 I_p

    M and N stay fixed along the entire lambda path, so decompose them once:

        M = T diag(mu) T^T
        N = S diag(eta) S^T

    Each later iteration then computes

        A = T [ (T^T H S) / (mu_i + eta_j) ] S^T

    as an explicit solution.
    """

    Y, X = _as_arrays(Y, X)
    graph_data = X if graph_data is None else np.asarray(graph_data, dtype=float)
    phi = _adaptive_phi(graph_data, graph_data.T) if phi is None else phi

    n, p = X.shape
    timings = {}

    start = time.perf_counter()
    row_pairs, row_w = _knn_edges(graph_data, m=m, phi=phi)
    col_pairs, col_w = _knn_edges(graph_data.T, m=m, phi=phi)
    row_w = row_w / row_w.sum() / math.sqrt(p)
    col_w = col_w / col_w.sum() / math.sqrt(n)
    E_row = _incidence_matrix(n, row_pairs)
    E_col = _incidence_matrix(p, col_pairs)
    timings["weights_and_graphs"] = time.perf_counter() - start

    start = time.perf_counter()
    L_row = np.asarray((E_row @ E_row.T).todense())
    L_col = np.asarray((E_col @ E_col.T).todense())
    M = nu0 * np.eye(n) + nu1 * L_row
    N = nu2 * L_col + nu3 * np.eye(p)

    eigvals_M, eigvecs_M = linalg.eigh(M, check_finite=False)
    eigvals_N, eigvecs_N = linalg.eigh(N, check_finite=False)
    eigen_denom = eigvals_M[:, None] + eigvals_N[None, :]
    timings["spectral_precompute"] = time.perf_counter() - start

    return BiHPRWorkspace(
        X=X,
        Y=Y,
        row_pairs=row_pairs,
        col_pairs=col_pairs,
        row_weights=row_w,
        col_weights=col_w,
        E_row=E_row,
        E_col=E_col,
        eigvals_M=eigvals_M,
        eigvecs_M=eigvecs_M,
        eigvals_N=eigvals_N,
        eigvecs_N=eigvecs_N,
        eigen_denom=eigen_denom,
        x_norm_sq=np.einsum("ij,ij->i", X, X),
        timings=timings,
    )


# =============================================================================
# 2. Weighted MCP proximal map
# =============================================================================


def weighted_mcp_prox(
    target,
    penalty_lambda,
    nu,
    gamma_mcp,
    weights,
    *,
    axis=1,
):
    """Closed-form weighted group-MCP update from equation (3) in the paper.

    For target vector t and weight w:

        prox(t) =
            ST(t, w*lambda/nu) / (1-w/(gamma*nu)),
                when ||t|| <= gamma*lambda

            t,  when ||t|| > gamma*lambda

    where ST(t,a) = max(1-a/||t||, 0) * t.

    Axis meaning
    ------------
    axis=0: each row of target is a group, used for v and z.
    axis=1: each column of target is a group, used for g.
    """

    target = np.asarray(target, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    # For axis=0 each row is a group, so norms run across columns; vice versa.
    norms = np.linalg.norm(target, axis=1 - axis)
    result = target.copy()
    inside = norms <= gamma_mcp * penalty_lambda
    if penalty_lambda == 0 or not np.any(inside):
        return result

    w = weights[inside]
    selected_norms = np.maximum(norms[inside], 1e-15)
    shrink = np.maximum(1.0 - w * penalty_lambda / (nu * selected_norms), 0.0)
    shrink /= 1.0 - w / (gamma_mcp * nu)

    if axis == 0:
        result[inside, :] *= shrink[:, None]
    else:
        result[:, inside] *= shrink[None, :]
    return result


# =============================================================================
# 3. ADMM core
# =============================================================================


def _new_state(work, beta_init=None):
    """Create the initial ADMM state.

    When beta_init is provided, both A and beta start from that matrix, and
    the auxiliary variables are initialized consistently:

        v = -E_row.T @ A
        z = -E_col.T @ A.T
        g = A

    This makes all four ADMM primal constraints feasible at iteration 0 and
    avoids zero auxiliary variables corrupting the nonconvex MCP initialization.
    """

    n, p = work.X.shape
    beta = (
        np.zeros((n, p))
        if beta_init is None
        else np.asarray(beta_init, dtype=np.float64).copy()
    )
    L_row = len(work.row_pairs)
    L_col = len(work.col_pairs)
    A = beta.copy()
    v = -np.asarray(work.E_row.T @ A)
    z = -np.asarray(work.E_col.T @ A.T)
    return {
        "A": A,
        "beta": beta,
        "v": v,
        "z": z,
        "g": A.copy(),
        "lambda_0": np.zeros((n, p)),
        "lambda_1": np.zeros((L_row, p)),
        "lambda_2": np.zeros((L_col, n)),
        "lambda_3": np.zeros((n, p)),
    }


def _warm_state(work, init):
    """Copy a complete warm-start state."""

    return {name: np.asarray(value, dtype=np.float64).copy()
            for name, value in init.items()}


def _solve_sylvester(work, H):
    """Solve M A + A N = H using the precomputed eigendecompositions."""

    H_tilde = work.eigvecs_M.T @ H @ work.eigvecs_N
    A_tilde = H_tilde / work.eigen_denom
    return work.eigvecs_M @ A_tilde @ work.eigvecs_N.T


def _update_beta(work, A, lambda_0, nu0):
    """Update beta sample by sample without building the legacy n x (np) block matrix.

    The subproblem for sample i is

        min_b  1/2 (y_i-x_i^T b)^2
             + nu0/2 ||b-A_i+lambda_0i/nu0||^2

    with normal equation

        (x_i x_i^T + nu0 I)b
            = y_i x_i + nu0 A_i - lambda_0i

    The Sherman-Morrison formula gives a vectorized update for all beta_i.
    """

    H = work.Y[:, None] * work.X + nu0 * A - lambda_0
    x_dot_H = np.einsum("ij,ij->i", work.X, H)
    correction = x_dot_H / (nu0 + work.x_norm_sq)
    return (H - correction[:, None] * work.X) / nu0


def _convergence_values(beta, A, v, row_delta, z, col_delta, g):
    """Compute the ADMM primal feasibility residual."""

    residuals = [np.sqrt(np.mean((beta - A) ** 2))]
    if len(v):
        residuals.append(
            np.max(np.linalg.norm(v + row_delta, axis=1)) / math.sqrt(A.shape[1])
        )
    if len(z):
        residuals.append(
            np.max(np.linalg.norm(z + col_delta, axis=1)) / math.sqrt(A.shape[0])
        )
    residuals.append(
        np.max(np.linalg.norm(g - A, axis=0)) / math.sqrt(A.shape[0])
    )
    return float(max(residuals))


def fit_bihpr_mcp_large(
    Y,
    X,
    *,
    lambda_col,
    lambda3,
    lambda_row=None,
    gamma_mcp=3.0,
    nu0=1.0,
    nu1=1.0,
    nu2=1.0,
    nu3=1.0,
    m=5,
    phi=None,
    feature_weights=None,
    beta_init=None,
    init=None,
    workspace=None,
    niter=1000,
    tol=1e-5,
    cluster_tol=1e-6,
    output=0,
):
    """Fit one (lambda, lambda3) setting.

    Parameters
    ----------
    lambda_col : float
        Column-fusion parameter lambda_2.
    lambda_row : float or None
        Row-fusion parameter lambda_1. If None, use the paper setting
        lambda_1 = sqrt(n/p) * lambda_2.
    lambda3 : float
        Feature-selection parameter.
    init : dict or None
        result["state"] returned by the previous fit, used for full warm start.
    workspace : BiHPRWorkspace or None
        Reuse the same workspace along a path to avoid recomputing weights and
        eigendecompositions.
    output : int
        0 disables printing; a positive integer prints every output iterations.

    Returns
    -------
    dict
        Contains A, beta, v, z, g, four dual-variable groups, cluster labels,
        convergence information, and memory estimates.
    """

    Y, X = _as_arrays(Y, X)
    n, p = X.shape
    if lambda_row is None:
        lambda_row = math.sqrt(n / p) * lambda_col

    work = workspace or prepare_bihpr_workspace(
        Y, X,
        nu0=nu0, nu1=nu1, nu2=nu2, nu3=nu3,
        m=m, phi=phi,
        graph_data=beta_init if beta_init is not None else X,
    )
    # The paper requires sum(feature_weights)=1/sqrt(n).
    if feature_weights is None:
        feature_weights = np.ones(p)
    feature_weights = np.asarray(feature_weights, dtype=np.float64).reshape(-1)
    feature_weights = feature_weights / feature_weights.sum() / math.sqrt(n)

    state = _new_state(work, beta_init) if init is None else _warm_state(work, init)
    A, beta = state["A"], state["beta"]
    v, z, g = state["v"], state["z"], state["g"]
    lambda_0 = state["lambda_0"]
    lambda_1 = state["lambda_1"]
    lambda_2 = state["lambda_2"]
    lambda_3_dual = state["lambda_3"]

    start = time.perf_counter()
    converged = False
    max_change = math.inf
    primal_residual = math.inf

    for iteration in range(1, int(niter) + 1):
        # Copy only the primal variables needed by the convergence criteria.
        # The primal residual already checks constraint feasibility directly.
        A_old = A.copy()
        beta_old = beta.copy()
        v_old = v.copy()
        z_old = z.copy()
        g_old = g.copy()

        # ------------------------------------------------------------------
        # Step 1: Update A.
        #
        # The four terms in H come from the beta=A, row-difference,
        # column-difference, and g=A constraints. Because E_row is defined as
        # A[i2]-A[i1], its right-hand-side contribution has a negative sign.
        # ------------------------------------------------------------------
        H = nu0 * beta + lambda_0
        H -= np.asarray(work.E_row @ (lambda_1 + nu1 * v))
        H -= np.asarray(work.E_col @ (lambda_2 + nu2 * z)).T
        H += lambda_3_dual + nu3 * g
        A = _solve_sylvester(work, H)

        # ------------------------------------------------------------------
        # Step 2: Update V, Z, and G.
        #
        # row_delta[l,:] = A[i2,:]-A[i1,:], shape (L_row,p)
        # col_delta[k,:] = A[:,j2]-A[:,j1], shape (L_col,n)
        # ------------------------------------------------------------------
        row_delta = np.asarray(work.E_row.T @ A)
        v_target = -row_delta - lambda_1 / nu1
        v = weighted_mcp_prox(
            v_target, lambda_row, nu1, gamma_mcp, work.row_weights, axis=0
        )

        col_delta = np.asarray(work.E_col.T @ A.T)
        z_target = -col_delta - lambda_2 / nu2
        z = weighted_mcp_prox(
            z_target, lambda_col, nu2, gamma_mcp, work.col_weights, axis=0
        )

        g_target = A - lambda_3_dual / nu3
        g = weighted_mcp_prox(
            g_target, lambda3, nu3, gamma_mcp, feature_weights, axis=1
        )

        # ------------------------------------------------------------------
        # Step 3: Update individualized regression coefficients beta.
        # ------------------------------------------------------------------
        beta = _update_beta(work, A, lambda_0, nu0)

        # ------------------------------------------------------------------
        # Step 4: Update the four dual-variable groups.
        # ------------------------------------------------------------------
        lambda_0 += nu0 * (beta - A)
        lambda_1 += nu1 * (v + row_delta)
        lambda_2 += nu2 * (z + col_delta)
        lambda_3_dual += nu3 * (g - A)

        max_change = max(
            np.mean(np.abs(A - A_old)),
            np.mean(np.abs(beta - beta_old)),
            np.mean(np.abs(v - v_old)),
            np.mean(np.abs(z - z_old)),
            np.mean(np.abs(g - g_old)),
        )
        primal_residual = _convergence_values(
            beta, A, v, row_delta, z, col_delta, g
        )

        if output and (iteration == 1 or iteration % int(output) == 0):
            mse = np.mean((Y - np.einsum("ij,ij->i", X, beta)) ** 2)
            print(
                f"iter={iteration:4d}  change={max_change:.3e}  "
                f"primal={primal_residual:.3e}  mse={mse:.6f}",
                flush=True,
            )

        if iteration > 2 and max_change < tol and primal_residual < tol:
            converged = True
            break

    fit_seconds = time.perf_counter() - start
    final_state = {
        "A": A,
        "beta": beta,
        "v": v,
        "z": z,
        "g": g,
        "lambda_0": lambda_0,
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "lambda_3": lambda_3_dual,
    }

    row_labels, col_labels = extract_clusters(final_state, work, tol=cluster_tol)
    active_features = np.linalg.norm(g, axis=0) > cluster_tol
    residual = Y - np.einsum("ij,ij->i", X, beta)
    memory_bytes = _large_memory_bytes(work, final_state)
    objective = bihpr_objective(
        work, beta, lambda_row, lambda_col, lambda3, gamma_mcp, feature_weights
    )

    return {
        **final_state,
        "state": final_state,
        "workspace": work,
        "row_labels": row_labels,
        "col_labels": col_labels,
        "active_features": active_features,
        "lambda_row": float(lambda_row),
        "lambda_col": float(lambda_col),
        "lambda3_tuning": float(lambda3),
        "feature_weights": feature_weights,
        "objective": objective,
        "rss": float(residual @ residual),
        "iterations": iteration,
        "converged": converged,
        "final_change": float(max_change),
        "primal_residual": primal_residual,
        "timings": {**work.timings, "fit": fit_seconds},
        "memory_bytes": memory_bytes,
        "legacy_memory_bytes": estimate_legacy_state_bytes(n, p),
    }


# =============================================================================
# 4. Clustering, feature weights, and memory statistics
# =============================================================================


def extract_clusters(state, workspace, *, tol=1e-6):
    """Extract graph connected components from edges where v=0 and z=0."""

    n, p = workspace.X.shape
    row_zero = np.linalg.norm(state["v"], axis=1) <= tol
    col_zero = np.linalg.norm(state["z"], axis=1) <= tol

    def labels_from_pairs(size, pairs):
        if len(pairs) == 0:
            return np.arange(size)
        graph = csr_matrix(
            (
                np.ones(2 * len(pairs)),
                (
                    np.r_[pairs[:, 0], pairs[:, 1]],
                    np.r_[pairs[:, 1], pairs[:, 0]],
                ),
            ),
            shape=(size, size),
        )
        return connected_components(graph, directed=False)[1]

    row_labels = labels_from_pairs(n, workspace.row_pairs[row_zero])
    col_labels = labels_from_pairs(p, workspace.col_pairs[col_zero])
    return row_labels, col_labels


def adaptive_feature_weights(beta_pilot, *, floor=1e-8):
    """Compute u_j = 1 / ||beta_(j)^(0)||_2 from the paper.

    Very small column norms are floored to prevent infinite weights. Returned
    weights are normalized so sum(u)=1/sqrt(n).
    """

    beta_pilot = np.asarray(beta_pilot)
    norms = np.linalg.norm(beta_pilot, axis=0)
    positive = norms[norms > floor]
    reference = np.median(positive) if len(positive) else 1.0
    norms = np.maximum(norms, max(floor, reference * 1e-6))
    weights = 1.0 / norms
    return weights / weights.sum() / math.sqrt(beta_pilot.shape[0])


def estimate_legacy_state_bytes(n, p):
    """Estimate persistent float64 memory in the legacy all-pair implementation."""

    n2 = n * (n - 1) // 2
    p2 = p * (p - 1) // 2
    elements = (
        2 * n * n2       # el1, el2
        + 2 * p * p2     # ek1, ek2
        + 2 * p * n2     # v, lambda_1
        + 2 * n * p2     # z, lambda_2
        + 5 * n * p      # A, beta, g, lambda_0, lambda_3
    )
    return int(elements * 8)


def _large_memory_bytes(work, state):
    """Count bytes used by the current state, graphs, and spectral arrays."""

    total = sum(value.nbytes for value in state.values())
    total += sum(
        value.nbytes
        for value in (
            work.X,
            work.Y,
            work.row_pairs,
            work.col_pairs,
            work.row_weights,
            work.col_weights,
            work.eigvals_M,
            work.eigvecs_M,
            work.eigvals_N,
            work.eigvecs_N,
            work.eigen_denom,
            work.x_norm_sq,
        )
    )
    for E in (work.E_row, work.E_col):
        total += E.data.nbytes + E.indices.nbytes + E.indptr.nbytes
    return int(total)


# =============================================================================
# 5. Objective function and compatibility entry points
# =============================================================================


def _mcp_penalty(norms, lam, gamma):
    """MCP P_gamma(t,lambda), used only for reporting the objective."""

    norms = np.asarray(norms)
    return np.where(
        norms <= gamma * lam,
        lam * norms - norms**2 / (2.0 * gamma),
        0.5 * gamma * lam**2,
    )


def bihpr_objective(work, beta, lambda_row, lambda_col, lambda3, gamma, u):
    """Compute the paper objective for comparing implementations."""

    residual = work.Y - np.einsum("ij,ij->i", work.X, beta)
    row_diff = np.asarray(work.E_row.T @ beta)
    col_diff = np.asarray(work.E_col.T @ beta.T)
    return float(
        0.5 * residual @ residual
        + work.row_weights
        @ _mcp_penalty(np.linalg.norm(row_diff, axis=1), lambda_row, gamma)
        + work.col_weights
        @ _mcp_penalty(np.linalg.norm(col_diff, axis=1), lambda_col, gamma)
        + u @ _mcp_penalty(np.linalg.norm(beta, axis=0), lambda3, gamma)
    )


def fit_bihpr_path(*args, **kwargs):
    """Compatibility entry point; BIC path code lives in BiHPR_single_test.py."""

    from BiHPR.BiHPR_single_test import fit_bihpr_path as _fit_path

    return _fit_path(*args, **kwargs)


def run_single_simulation(*args, **kwargs):
    """Compatibility entry point; single-simulation code lives in BiHPR_single_test.py."""

    from BiHPR.BiHPR_single_test import run_single_simulation as _run_once

    return _run_once(*args, **kwargs)


if __name__ == "__main__":
    print("This file only contains the BiHPR solver.")
    print("Run one paper-style simulation with: python -m BiHPR.BiHPR_single_test")

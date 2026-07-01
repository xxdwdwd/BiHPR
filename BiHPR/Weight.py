"""Pairwise kernel-weight helpers for BiHPR graph construction.

The current large-scale solver builds sparse KNN graphs internally, but these
utilities are kept as reusable compatibility helpers for experiments and for
checking parity with the original R implementation. They operate on the
flattened upper triangle of a symmetric pairwise-weight matrix.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix
from scipy.spatial.distance import pdist


def tri2vec(rows, cols, n):
    """Map upper-triangular matrix coordinates to flattened vector indices.

    The flattened vector stores the strict upper triangle of an ``n x n`` matrix
    in row-major order. Inputs may contain either ``(i, j)`` or ``(j, i)``; the
    function normalizes each pair internally before computing the index.

    Parameters
    ----------
    rows, cols : int or array-like of int
        Zero-based matrix coordinates. Diagonal entries are not supported.
    n : int
        Dimension of the original square matrix.

    Returns
    -------
    numpy.ndarray or numpy scalar
        Zero-based flattened index or array of indices.
    """

    rows = np.asarray(rows)
    cols = np.asarray(cols)
    actual_rows = np.minimum(rows, cols)
    actual_cols = np.maximum(rows, cols)

    if np.any(actual_rows == actual_cols):
        raise ValueError("Diagonal entries are not represented in the vector.")
    if np.any(actual_cols >= n) or np.any(actual_rows < 0):
        raise ValueError(f"Index is out of bounds for matrix dimension {n}.")

    return (
        n * actual_rows
        - actual_rows * (actual_rows + 1) // 2
        + actual_cols
        - actual_rows
        - 1
    )


def vec2tri(indices, n):
    """Map flattened upper-triangular indices back to coordinate pairs.

    Parameters
    ----------
    indices : array-like of int
        Zero-based vector indices in the flattened strict upper triangle.
    n : int
        Dimension of the original square matrix.

    Returns
    -------
    list[tuple[int, int]]
        Zero-based coordinate pairs ``(i, j)`` with ``i < j``.
    """

    pairs = []
    for idx in indices:
        k = idx + 1
        m = 2 * n - 1
        sqrt_val = np.sqrt(m**2 - 8 * k)
        i = np.ceil(0.5 * (m - sqrt_val)).astype(int)
        j = k - (i - 1) * (2 * n - i) // 2 + i
        pairs.append((i - 1, j - 1))
    return pairs


def kernel_weights(X, phi=1.0):
    """Compute Gaussian kernel weights between sample columns.

    Parameters
    ----------
    X : array-like of shape (p, n)
        Data matrix with features in rows and samples in columns.
    phi : float, default=1.0
        Gaussian kernel bandwidth parameter used in ``exp(-phi * distance^2)``.

    Returns
    -------
    numpy.ndarray
        Strict-upper-triangle Gaussian weights in row-major vector order.
    """

    X = np.asarray(X, dtype=np.float64)
    phi = float(phi)
    samples = X.T
    dist_sq = pdist(samples, metric="sqeuclidean")
    return np.exp(-phi * dist_sq)


def knn_weights(w, k, n):
    """Keep the union of top-k weighted neighbors for every node.

    Parameters
    ----------
    w : array-like of shape (n * (n - 1) / 2,)
        Flattened strict-upper-triangle pairwise weights.
    k : int
        Number of highest-weight neighbors kept for each node.
    n : int
        Number of nodes in the original pairwise matrix.

    Returns
    -------
    scipy.sparse.csc_matrix
        Sparse column vector with retained weights. All non-neighbor entries
        are set to zero.
    """

    if not isinstance(w, np.ndarray):
        w = np.asarray(w)

    if w.ndim != 1:
        raise ValueError("w must be a one-dimensional NumPy array.")

    expected_len = n * (n - 1) // 2
    if n > 0 and len(w) != expected_len:
        if not (n <= 1 and expected_len == 0 and len(w) == 0):
            raise ValueError(
                f"w has length {len(w)}, but n={n} requires length "
                f"{expected_len}."
            )
    elif n <= 1 and len(w) != 0:
        raise ValueError(
            f"w must be empty when n={n}; got length {len(w)}."
        )

    if k <= 0:
        w_out = np.zeros_like(w, dtype=w.dtype)
        return csc_matrix(w_out[:, np.newaxis])

    keep_indices = set()
    for node in range(n):
        cols_gt = np.arange(node + 1, n)
        indices_a = np.array([], dtype=int)
        if cols_gt.size > 0:
            rows_a = np.full_like(cols_gt, node)
            indices_a = tri2vec(rows_a, cols_gt, n)

        rows_lt = np.arange(0, node)
        indices_b = np.array([], dtype=int)
        if rows_lt.size > 0:
            cols_b = np.full_like(rows_lt, node)
            indices_b = tri2vec(rows_lt, cols_b, n)

        neighbor_w_indices = np.concatenate((indices_a, indices_b))
        if neighbor_w_indices.size > 0:
            actual_k = min(k, len(neighbor_w_indices))
            if actual_k > 0:
                sorted_relative_indices = np.argsort(w[neighbor_w_indices])[::-1]
                top_k_w_indices = neighbor_w_indices[
                    sorted_relative_indices[:actual_k]
                ]
                keep_indices.update(top_k_w_indices)

    w_out = np.zeros_like(w, dtype=w.dtype)
    if keep_indices:
        keep_indices_array = np.array(sorted(keep_indices))
        w_out[keep_indices_array] = w[keep_indices_array]

    return csc_matrix(w_out[:, np.newaxis])


def convert_weights(X, m, phi, feature_weight):
    """Build normalized row, column, and feature weights.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Regression design matrix with samples in rows and features in columns.
    m : int
        Number of KNN neighbors retained for both sample and feature graphs.
    phi : float
        Gaussian kernel bandwidth parameter.
    feature_weight : array-like of shape (p,)
        Raw feature-selection weights.

    Returns
    -------
    tuple
        ``(w_l, u_k, feature_weight)`` as column vectors. ``w_l`` contains
        normalized row-fusion weights, ``u_k`` contains normalized
        column-fusion weights, and ``feature_weight`` contains normalized
        feature-selection weights.
    """

    n, p = X.shape
    k_row = m
    k_col = m

    w_row = kernel_weights(X.T, phi)
    w_col = kernel_weights(X, phi)

    w_row = knn_weights(w_row, k_row, n)
    w_col = knn_weights(w_col, k_col, p)

    w_row = w_row / np.sum(w_row)
    w_col = w_col / np.sum(w_col)
    w_row = w_row / np.sqrt(p)
    w_col = w_col / np.sqrt(n)

    w_l = w_row.reshape(-1, 1)
    u_k = w_col.reshape(-1, 1)

    feature_weight = feature_weight / np.sum(feature_weight) / np.sqrt(p)
    feature_weight = feature_weight.reshape(-1, 1)

    return w_l, u_k, feature_weight

"""Graph and label utilities used by the BiHPR simulation workflow.

The solver itself returns coefficient estimates and graph-derived labels. This
module provides small, explicit helpers for post-processing those estimates:

* convert a distance or similarity matrix into an adjacency matrix;
* compute pairwise distances between rows of a coefficient matrix;
* extract connected components from an adjacency graph;
* mark nearly-zero coefficient columns as inactive features;
* compute false-negative and false-positive rates for binary labels.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import confusion_matrix


def similarity_to_adjacency(
    similarity_matrix,
    threshold,
    similarity_measure="difference",
):
    """Convert a pairwise score matrix into an unweighted adjacency matrix.

    Parameters
    ----------
    similarity_matrix : array-like of shape (n_samples, n_samples)
        Pairwise distances or similarities.
    threshold : float
        Edge threshold. Distances are connected when they are no larger than
        the threshold; cosine similarities are connected when they are at least
        the threshold.
    similarity_measure : {"difference", "euclidean", "cosine"}
        Interpretation of ``similarity_matrix``.

    Returns
    -------
    numpy.ndarray
        Binary adjacency matrix with a zero diagonal, so the graph has no
        self-loops.
    """

    n_samples = similarity_matrix.shape[0]
    adjacency_matrix = np.zeros((n_samples, n_samples), dtype=int)

    if similarity_measure in {"difference", "euclidean"}:
        adjacency_matrix = (similarity_matrix <= threshold).astype(int)
    elif similarity_measure == "cosine":
        adjacency_matrix = (similarity_matrix >= threshold).astype(int)
    else:
        raise ValueError(
            "similarity_measure must be one of 'difference', 'euclidean', "
            "or 'cosine'."
        )

    np.fill_diagonal(adjacency_matrix, 0)
    return adjacency_matrix


def compute_difference_matrix(data_matrix, metric="euclidean"):
    """Compute a pairwise distance matrix for the rows of ``data_matrix``.

    Parameters
    ----------
    data_matrix : array-like of shape (n_samples, n_features)
        Rows to compare.
    metric : str, default="euclidean"
        Any metric accepted by ``sklearn.metrics.pairwise_distances``.

    Returns
    -------
    numpy.ndarray
        Pairwise distance matrix of shape ``(n_samples, n_samples)``.
    """

    from sklearn.metrics.pairwise import pairwise_distances

    return pairwise_distances(data_matrix, metric=metric)


def find_clusters(A):
    """Find connected components in the graph represented by adjacency ``A``.

    This is the SciPy equivalent of the original R helper that used igraph. The
    graph is treated as undirected because BiHPR clustering is based on fused
    pairs rather than directed relationships.

    Parameters
    ----------
    A : array-like or scipy.sparse matrix
        Adjacency matrix, usually symmetric.

    Returns
    -------
    dict
        Dictionary with three entries:

        ``n_clusters``
            Number of connected components.
        ``cluster``
            Component label for each node.
        ``size``
            Number of nodes in each component.
    """

    if not sp.issparse(A):
        A = sp.csc_matrix(A)

    n_nodes = A.shape[0]
    if n_nodes == 0:
        return {
            "n_clusters": 0,
            "cluster": np.array([], dtype=int),
            "size": np.array([], dtype=int),
        }

    n_components, labels = connected_components(
        csgraph=A,
        directed=False,
        connection="weak",
        return_labels=True,
    )

    if n_components == 0:
        return {
            "n_clusters": 0,
            "cluster": np.array([], dtype=int),
            "size": np.array([], dtype=int),
        }

    unique_labels, counts = np.unique(labels, return_counts=True)
    size_array = np.zeros(n_components, dtype=int)
    if len(unique_labels) == n_components and np.all(
        unique_labels == np.arange(n_components)
    ):
        size_array = counts
    else:
        for label_id, count in zip(unique_labels, counts):
            if 0 <= label_id < n_components:
                size_array[label_id] = count

    return {
        "n_clusters": n_components,
        "cluster": labels,
        "size": size_array,
    }


def label_columns_by_threshold(matrix, threshold=0.9):
    """Classify columns as inactive or active using a near-zero proportion rule.

    A column receives label ``0`` when at least ``threshold`` of its entries lie
    in ``[-0.1, 0.1)``. Otherwise the column receives label ``1``. This mirrors
    the post-processing rule used by the Gaussian simulation notebook.

    Parameters
    ----------
    matrix : array-like of shape (n_samples, n_features)
        Coefficient matrix to label column by column.
    threshold : float, default=0.9
        Required proportion of near-zero coefficients for an inactive feature.

    Returns
    -------
    list[int]
        Column labels, where 0 means inactive and 1 means active.
    """

    column_labels = []
    for col_idx in range(matrix.shape[1]):
        column_data = matrix[:, col_idx]
        in_range_count = np.sum((column_data >= -0.1) & (column_data < 0.1))
        total_count = len(column_data)
        ratio = in_range_count / total_count if total_count > 0 else 0
        column_labels.append(0 if ratio >= threshold else 1)
    return column_labels


def calculate_fnr_fpr(y_true, y_pred, pos_label=1):
    """Compute false-negative and false-positive rates for binary labels.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_pred : array-like
        Predicted binary labels.
    pos_label : int, default=1
        Label treated as the positive class.

    Returns
    -------
    tuple[float, float]
        ``(fnr, fpr)``, where ``fnr`` is the false-negative rate and ``fpr`` is
        the false-positive rate.
    """

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[1 - pos_label, pos_label],
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    return fnr, fpr

"""Utility functions for the TCGA lung real-data notebook."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import manhattan_distances

KEY_GENES = ["NQO1", "CBR1", "GPX2", "ALDH3A1", "CYP4F11"]


def _column_letters(reference: str) -> str:
    return "".join(character for character in reference if character.isalpha())


def read_lung_coef_xlsx(path: Path) -> pd.DataFrame:
    """Read the coefficient workbook without requiring openpyxl."""
    namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = ["".join((text.text or "") for text in item.findall(".//a:t", namespace)) for item in root.findall("a:si", namespace)]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: list[dict[str, str]] = []
        for row in sheet.findall(".//a:sheetData/a:row", namespace):
            record: dict[str, str] = {}
            for cell in row.findall("a:c", namespace):
                column = _column_letters(cell.get("r", ""))
                value = cell.find("a:v", namespace)
                if column not in {"A", "B", "C"}:
                    continue
                if cell.get("t") == "inlineStr":
                    record[column] = "".join((text.text or "") for text in cell.findall(".//a:t", namespace))
                elif value is not None:
                    raw = value.text or ""
                    record[column] = shared_strings[int(raw)] if cell.get("t") == "s" else raw
            rows.append(record)
    records = []
    for row in rows[1:]:
        gene = row.get("A", "").strip()
        if gene:
            records.append({"gene": gene, "Subgroup 1": float(row.get("B", 0.0) or 0.0), "Subgroup 2": float(row.get("C", 0.0) or 0.0)})
    return pd.DataFrame(records)


def log_normalization(x: np.ndarray, scale_factor: float = 10000.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    library_size = x.sum(axis=1, keepdims=True)
    library_size[library_size == 0] = 1.0
    return np.log1p(x / library_size * scale_factor)


class LocalFusedLassoInitializer:
    """Local fused-lasso initializer from the original analysis workflow."""

    def __init__(self, k: int = 10, alpha: float = 2.0, lam_sparsity: float = 10.0, lam_fusion: float = 20.0):
        self.k, self.alpha = k, alpha
        self.lam_sparsity, self.lam_fusion = lam_sparsity, lam_fusion

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        try:
            import cvxpy as cp
        except ImportError as error:
            raise ImportError("Install the real-data extras with `pip install -e .[realdata]`.") from error
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float).reshape(-1)
        n, p = x.shape
        feature_distances = np.nan_to_num(pdist(x.T, metric="correlation"), nan=1.0, posinf=1.0, neginf=1.0)
        order = np.asarray(hierarchy.dendrogram(hierarchy.linkage(feature_distances, method="ward"), no_plot=True)["leaves"])
        x_sorted = x[:, order]
        distances = manhattan_distances(x_sorted) + self.alpha * manhattan_distances(y.reshape(-1, 1))
        beta_sorted = np.zeros((n, p))
        beta = cp.Variable(p)
        neighbors_count = min(self.k, n)
        x_parameter, y_parameter = cp.Parameter((neighbors_count, p)), cp.Parameter(neighbors_count)
        loss = cp.sum_squares(y_parameter - x_parameter @ beta) / (2 * neighbors_count)
        problem = cp.Problem(cp.Minimize(loss + self.lam_sparsity * cp.norm(beta, 1) + self.lam_fusion * cp.norm(cp.diff(beta), 1)))
        for index in range(n):
            neighbors = np.argsort(distances[index])[:neighbors_count]
            x_parameter.value, y_parameter.value = x_sorted[neighbors], y[neighbors]
            problem.solve(solver=cp.OSQP, warm_start=False, verbose=False, ignore_dpp=True)
            if beta.value is not None:
                values = np.asarray(beta.value).reshape(-1)
                values[np.abs(values) < 1e-3] = 0.0
                beta_sorted[index] = values
        result = np.zeros_like(beta_sorted)
        result[:, order] = beta_sorted
        return result


def bic_for_result(result: dict, n: int) -> float:
    active = np.asarray(result["active_features"], dtype=bool)
    row_count = len(np.unique(result["row_labels"]))
    column_count = len(np.unique(np.asarray(result["col_labels"])[active])) if active.any() else 1
    return math.log(max(float(result["rss"]) / n, np.finfo(float).tiny)) + math.log(n) * max(1, row_count * column_count) / n


def save_split_heatmap(beta: np.ndarray, features: list[str], clusters: np.ndarray, output_path: Path, title: str = "") -> None:
    order, boundaries = [], []
    for cluster in sorted(np.unique(clusters)):
        order.extend(np.where(clusters == cluster)[0].tolist())
        boundaries.append(len(order))
    beta_sorted = beta[order]
    column_order = hierarchy.leaves_list(hierarchy.linkage(beta_sorted.T, method="ward")) if beta.shape[1] > 1 else np.array([0])
    figure, axis = plt.subplots(figsize=(18 if len(features) <= 600 else 22, 10))
    image = axis.imshow(beta_sorted[:, column_order], aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
    if title:
        axis.set_title(title)
    axis.set_yticks([])
    axis.set_xticks([])
    start, transform = 0, axis.get_yaxis_transform()
    for cluster, end in zip(sorted(np.unique(clusters)), boundaries):
        if end < len(order):
            axis.axhline(end - 0.5, color="green", linewidth=2, linestyle="--")
        axis.text(-0.01, (start + end - 1) / 2, f"Predicted\\nCluster {cluster + 1}", va="center", ha="right", fontweight="bold", transform=transform)
        start = end
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def active_gene_tables(beta: np.ndarray, features: list[str], clusters: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(beta, columns=features)
    mean_rows, summary_rows = [], []
    for cluster in sorted(np.unique(clusters)):
        means = frame.iloc[np.where(clusters == cluster)[0]].mean(axis=0)
        mean_rows.append({"cluster": int(cluster), **means.to_dict()})
        for gene in means.abs().sort_values(ascending=False).head(30).index:
            summary_rows.append({"cluster": int(cluster), "gene": gene, "mean_beta": float(means[gene]), "abs_mean_beta": float(abs(means[gene])), "is_key_gene": gene in KEY_GENES})
    for gene in KEY_GENES:
        if gene in frame.columns:
            for cluster in sorted(np.unique(clusters)):
                mean_value = frame.iloc[np.where(clusters == cluster)[0]][gene].mean()
                summary_rows.append({"cluster": int(cluster), "gene": gene, "mean_beta": float(mean_value), "abs_mean_beta": float(abs(mean_value)), "is_key_gene": True, "forced_key_gene_row": True})
    return pd.DataFrame(mean_rows), pd.DataFrame(summary_rows)

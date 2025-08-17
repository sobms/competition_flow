from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import polars as pl


def _ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
	if labels.size == 0:
		return 0.0
	order = np.argsort(-scores)
	L = min(k, labels.shape[0])
	labels_k = labels[order][:L]
	denom = np.log2(np.arange(2, L + 2))
	dcg = np.sum((2.0 ** labels_k - 1.0) / denom)
	ideal = np.sort(labels)[::-1][:L]
	idcg = np.sum((2.0 ** ideal - 1.0) / denom)
	return float(dcg / idcg) if idcg > 1e-12 else 0.0


def _recall_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
	if labels.size == 0:
		return 0.0
	order = np.argsort(-scores)
	labels_k = labels[order][:k]
	total_pos = max(1.0, float(labels.sum()))
	return float(labels_k.sum() / total_pos)


class Evaluator:
	def __init__(self, k_list: List[int] | None = None) -> None:
		self.k_list = k_list or [5, 10, 20]

	def evaluate(self, df: pl.DataFrame, score_col: str = "score", label_col: str = "label", group_col: str = "uid") -> Dict[str, Any]:
		metrics: Dict[str, float] = {f"ndcg_at_{k}": 0.0 for k in self.k_list}
		metrics.update({f"recall_at_{k}": 0.0 for k in self.k_list})
		agg = (
			df.group_by(group_col)
			.agg([
				pl.col(label_col).implode().alias("__labels__"),
				pl.col(score_col).implode().alias("__scores__"),
			])
		)
		n_groups = len(agg)
		for row in agg.iter_rows(named=True):
			raw_labels = row["__labels__"]  # list with possible None
			raw_scores = row["__scores__"]  # list with possible None
			labels = np.array([0.0 if (v is None) else float(v) for v in raw_labels], dtype=float)
			labels = (labels > 0.0).astype(float)
			scores = np.array([-1e9 if (v is None) else float(v) for v in raw_scores], dtype=float)
			for k in self.k_list:
				metrics[f"ndcg_at_{k}"] += _ndcg_at_k(labels, scores, k)
				metrics[f"recall_at_{k}"] += _recall_at_k(labels, scores, k)
		if n_groups > 0:
			for key in list(metrics.keys()):
				metrics[key] /= n_groups
		return metrics

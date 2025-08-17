from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import polars as pl


def _ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
	order = np.argsort(-scores)
	labels_k = labels[order][:k]
	dcg = np.sum((2 ** labels_k - 1) / np.log2(np.arange(2, k + 2)))
	ideal_k = np.sort(labels)[::-1][:k]
	idcg = np.sum((2 ** ideal_k - 1) / np.log2(np.arange(2, k + 2)))
	return float(dcg / idcg) if idcg > 0 else 0.0


def _recall_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
	order = np.argsort(-scores)
	labels_k = labels[order][:k]
	return float(labels_k.sum() / max(1, labels.sum()))


class Evaluator:
	def __init__(self, k_list: List[int] | None = None) -> None:
		self.k_list = k_list or [5, 10, 20]

	def evaluate(self, df: pl.DataFrame, score_col: str = "score", label_col: str = "label", group_col: str = "uid") -> Dict[str, Any]:
		metrics: Dict[str, float] = {f"ndcg@{k}": 0.0 for k in self.k_list}
		metrics.update({f"recall@{k}": 0.0 for k in self.k_list})
		agg = (
			df.group_by(group_col)
			.agg([
				pl.col(label_col).list().alias("__labels__"),
				pl.col(score_col).list().alias("__scores__"),
			])
		)
		n_groups = len(agg)
		for row in agg.iter_rows(named=True):
			labels = np.asarray(row["__labels__"])  # type: ignore[arg-type]
			scores = np.asarray(row["__scores__"])  # type: ignore[arg-type]
			for k in self.k_list:
				metrics[f"ndcg@{k}"] += _ndcg_at_k(labels, scores, k)
				metrics[f"recall@{k}"] += _recall_at_k(labels, scores, k)
		if n_groups > 0:
			for key in list(metrics.keys()):
				metrics[key] /= n_groups
		return metrics

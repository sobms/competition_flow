from __future__ import annotations
from typing import Dict, List, Tuple, Set
import math

# preds: user_id -> list of (item_id, score) sorted by score desc
# ground_truth: Set[(user_id, item_id)] — содержит только истинные (позитивные) пары

def calculate_eval_metrics(
	preds: Dict[int, List[Tuple[int, float]]],
	ground_truth: Set[Tuple[int, int]],
	ks: List[int],
) -> Dict[str, float]:
	# build per-user GT set
	gt_per_user: Dict[int, Set[int]] = {}
	for (u, i) in ground_truth:
		gt_per_user.setdefault(u, set()).add(i)

	users = [u for u in preds.keys() if u in gt_per_user]
	if not users:
		return {f"recall_at_{k}": 0.0 for k in ks} | {f"ndcg_at_{k}": 0.0 for k in ks}

	rec = {k: 0.0 for k in ks}
	ndcg = {k: 0.0 for k in ks}
	for u in users:
		gt = gt_per_user[u]
		ranked = [i for i, _ in preds[u]]
		for k in ks:
			topk = ranked[:k]
			hits = sum(1 for i in topk if i in gt)
			rec[k] += hits / max(1, len(gt))
			# NDCG_at_k
			dcg = 0.0
			for idx, item in enumerate(topk, start=1):
				if item in gt:
					dcg += 1.0 / math.log2(idx + 1)
			ideal_hits = min(len(gt), k)
			idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1)) if ideal_hits > 0 else 1.0
			ndcg[k] += (dcg / idcg) if idcg > 0 else 0.0

	m: Dict[str, float] = {}
	for k in ks:
		m[f"recall_at_{k}"] = rec[k] / len(users)
		m[f"ndcg_at_{k}"] = ndcg[k] / len(users)
	return m

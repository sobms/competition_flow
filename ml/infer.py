from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Callable, List
import hydra
from omegaconf import DictConfig, OmegaConf
import polars as pl
from datetime import datetime

# Adapter registry (aligned with train.py)
from ml.models.als import ALSAdapter

ADAPTERS: Dict[str, Callable[..., Any]] = {
	"als": ALSAdapter,
}
TOPK = 100

def resolve_model_from_cfg(model_cfg: Dict[str, Any]):
	name = model_cfg.get("name")
	if name not in ADAPTERS:
		raise ValueError(f"Unknown model: {name}")
	return ADAPTERS[name]


def _read_test_user_ids(base_dir: Path) -> List[int]:
	# Scan all parquet files and collect unique user_id
	lf = pl.scan_parquet(str(base_dir / "**/*.parquet"))
	users = lf.select(pl.col("user_id")).unique().collect(streaming=True)
	return users["user_id"].to_list() if users.height else []


def _write_submission(recommendations: Dict[int, List[int]], out_csv: Path) -> None:
	rows = []
	for user_id, items in recommendations.items():
		rows.append({
			"user_id": int(user_id),
			"item_id_1 item_id_2 ... item_id_100": " ".join(str(i) for i in items),
		})
	df = pl.DataFrame(rows)
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	df.write_csv(str(out_csv))


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
	# Resolve adapter class and load model
	adapter_cls = resolve_model_from_cfg(OmegaConf.to_container(cfg.model, resolve=True))
	model_dir = os.path.join(cfg.artifacts_dir, "model")
	model = adapter_cls.load(model_dir)

	# Read users from test_for_participants
	test_base = Path("data/raw_data") / "test_for_participants"
	users = _read_test_user_ids(test_base)
	assert users, "[ERROR] No users found in test_for_participants!"

	# Infer
	preds = model.infer(users, N=TOPK)
	# Convert to submission: keep only item ids
	rec_items: Dict[int, List[int]] = {u: [itm for itm, _ in pairs] for u, pairs in preds.items()}

	# Write submission CSV
	stamp = datetime.now().strftime("%Y%m%d_%H%M")
	out_csv = Path(cfg.artifacts_dir) / f"submission_{stamp}.csv"
	_write_submission(rec_items, out_csv)
	print(f"[infer] Submission written to {out_csv}")


if __name__ == "__main__":
	main()

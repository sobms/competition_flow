from __future__ import annotations
import os
from typing import Any
import hydra
from omegaconf import DictConfig, OmegaConf
import polars as pl

from ml.core.data import DataPaths, load_split
from ml.models.dummy import DummyModel
from ml.models.lgbm import LGBMRegressor


def _load_model(name: str, path: str):
	if name == "dummy":
		return DummyModel.load(path)
	elif name == "lgbm":
		return LGBMRegressor.load(path)
	else:
		raise ValueError(f"Unknown model: {name}")


def _resolve_split(cfg: DictConfig) -> str:
	# Prefer infer.split if provided to avoid group collision
	infer_section = getattr(cfg, "infer", None)
	if infer_section is not None:
		val = getattr(infer_section, "split", None)
		if isinstance(val, str) and val:
			return val
	# Fallback: if cfg.split is a string (not the split group dict)
	val = getattr(cfg, "split", None)
	if isinstance(val, str) and val:
		return val
	return "test"


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
	paths = DataPaths(
		raw_dir=cfg.data.paths.raw_dir,
		splits_dir=cfg.data.paths.splits_dir,
		features_dir=cfg.data.paths.features_dir,
	)
	split = _resolve_split(cfg)
	feat_df = load_split(paths, split)
	model_dir = os.path.join(cfg.artifacts_dir, "model")
	model = _load_model(cfg.model.name, model_dir)
	preds = model.predict(feat_df)
	out = feat_df.select(["uid", "item_id"]).with_columns(pl.Series(name="score", values=preds))
	os.makedirs(cfg.artifacts_dir, exist_ok=True)
	out_path = os.path.join(cfg.artifacts_dir, f"preds_{split}.parquet")
	out.write_parquet(out_path)
	print(f"Predictions written to {out_path}")


if __name__ == "__main__":
	main()

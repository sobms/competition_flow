from __future__ import annotations
import os
import json
from typing import Any, Dict
import hydra
from omegaconf import DictConfig, OmegaConf
import polars as pl
import mlflow

from ml.core.data import load_raw, make_splits, build_features, DataPaths
from ml.core.evaluator import Evaluator
from ml.core.utils import seed_all, setup_mlflow, log_mlflow_params, log_mlflow_metrics
from ml.models.dummy import DummyModel


def resolve_model_from_cfg(model_cfg: Dict[str, Any]):
	name = model_cfg.get("name", "dummy")
	params = model_cfg.get("params", {})
	if name == "dummy":
		return DummyModel(**params)
	elif name == "lgbm":
		from ml.models.lgbm import LGBMRegressor
		return LGBMRegressor(**params)
	elif name == "recbole":
		from ml.models.recbole_adapter import RecBoleAdapter
		return RecBoleAdapter(**params)
	else:
		raise ValueError(f"Unknown model: {name}")


def _paths_from_cfg(cfg: DictConfig) -> DataPaths:
	return DataPaths(
		raw_dir=cfg.data.paths.raw_dir,
		splits_dir=cfg.data.paths.splits_dir,
		features_dir=cfg.data.paths.features_dir,
	)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
	seed_all(int(cfg.seed))
	paths = _paths_from_cfg(cfg)
	mode = getattr(cfg, "pipeline", {}).get("mode", "train")

	if mode == "split":
		df = load_raw(OmegaConf.to_container(cfg, resolve=True))
		make_splits(df, OmegaConf.to_container(cfg, resolve=True), paths)
		return
	elif mode == "features":
		build_features(OmegaConf.to_container(cfg, resolve=True), paths)
		return

	# default: train + validate
	setup_mlflow(cfg.mlflow.tracking_uri, cfg.mlflow.experiment)
	df = load_raw(OmegaConf.to_container(cfg, resolve=True))
	make_splits(df, OmegaConf.to_container(cfg, resolve=True), paths)
	train_df, valid_df = build_features(OmegaConf.to_container(cfg, resolve=True), paths)

	model = resolve_model_from_cfg(OmegaConf.to_container(cfg.model, resolve=True))
	with mlflow.start_run():
		log_mlflow_params({"seed": cfg.seed, "model": cfg.model.name, **(cfg.model.params or {})})
		model.fit(train_df, valid_df)
		valid_scores = pl.Series(name="score", values=model.predict(valid_df))
		valid_eval = valid_df.with_columns(valid_scores)
		metrics = Evaluator().evaluate(valid_eval, score_col="score", label_col="label", group_col="uid")
		os.makedirs(cfg.artifacts_dir, exist_ok=True)
		log_mlflow_metrics(metrics)
		metrics_path = os.path.join(cfg.artifacts_dir, "metrics.json")
		with open(metrics_path, "w") as f:
			json.dump(metrics, f)
		# save model
		model_dir = os.path.join(cfg.artifacts_dir, "model")
		os.makedirs(model_dir, exist_ok=True)
		try:
			model.save(model_dir)
		except TypeError:
			model.save(os.path.join(model_dir, "model.txt"))
		print("Validation metrics:", metrics)


if __name__ == "__main__":
	main()

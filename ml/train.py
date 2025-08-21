from __future__ import annotations
import os
import json
from typing import Any, Dict, Callable
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow

from ml.core.utils import seed_all, setup_mlflow, log_mlflow_params, log_mlflow_metrics
from ml.models.als import ALSAdapter
from ml.models.bpr import BPRAdapter
from ml.core.recsys_metrics import calculate_eval_metrics

# Adapter registry
ADAPTERS: Dict[str, Callable[..., Any]] = {
	"als": ALSAdapter,
	"bpr": BPRAdapter,
}


def resolve_model_from_cfg(model_cfg: Dict[str, Any]):
	name = model_cfg.get("name")
	params = model_cfg.get("params", {})
	if name not in ADAPTERS:
		raise ValueError(f"Unknown model: {name}")
	return ADAPTERS[name](**params)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
	seed_all(int(cfg.seed))
	setup_mlflow(cfg.mlflow.tracking_uri, cfg.mlflow.experiment)
	model = resolve_model_from_cfg(OmegaConf.to_container(cfg.model, resolve=True))

	print("[train] Preparing datasets...")
	train_data, valid_data = model.prepare_dataset(cfg)

	# Train
	with mlflow.start_run():
		log_mlflow_params({"seed": cfg.seed, "model": cfg.model.name, **(cfg.model.params or {})})
		print("[train] Fitting model...")
		model.fit(train_data)

		print("[train] Inference on validation users and metrics computation...")
		eval_cfg = cfg.get("eval", None)
		ks = list(eval_cfg.ks)
		topk = int(eval_cfg.topk)
		preds = model.infer(valid_data["users"], N=topk)
		metrics = calculate_eval_metrics(preds, valid_data["ground_truth"], ks)
		os.makedirs(cfg.artifacts_dir, exist_ok=True)
		log_mlflow_metrics(metrics)
		metrics_path = os.path.join(cfg.artifacts_dir, "metrics.json")
		with open(metrics_path, "w") as f:
			json.dump(metrics, f)

		print("[train] Saving model artifacts...")
		model_dir = os.path.join(cfg.artifacts_dir, "model")
		os.makedirs(model_dir, exist_ok=True)
		model.save(model_dir)
		print("Validation metrics:", metrics)


if __name__ == "__main__":
	main()

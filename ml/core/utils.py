from __future__ import annotations
import os
import random
import numpy as np
import mlflow
from typing import Dict, Any


def seed_all(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)


def setup_mlflow(tracking_uri: str, experiment: str) -> None:
	mlflow.set_tracking_uri(tracking_uri)
	mlflow.set_experiment(experiment)


def log_mlflow_params(params: Dict[str, Any]) -> None:
	for k, v in params.items():
		mlflow.log_param(k, v)


def log_mlflow_metrics(metrics: Dict[str, float]) -> None:
	for k, v in metrics.items():
		mlflow.log_metric(k, float(v))

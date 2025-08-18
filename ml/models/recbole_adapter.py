from __future__ import annotations
from typing import Any, Dict, List
import os
import json
import shutil
import numpy as np
import polars as pl

from ml.core.interfaces import BaseModel


class RecBoleAdapter(BaseModel):
	"""
	Универсальный адаптер RecBole для CTR/General моделей.
	Поддерживает DCNv2 через params: {model: 'DCNv2', config_overrides: {...}}.
	Требуется бинарная колонка 'label' в train/valid.
	"""

	def __init__(self, model: str | None = None, config_overrides: Dict[str, Any] | None = None, **kwargs: Any) -> None:
		# В Hydra конфиге теперь: cfg.model.params.{model, config_overrides}
		if model is None:
			model = kwargs.get("params", {}).get("model", "DCNv2")
		if config_overrides is None:
			config_overrides = kwargs.get("params", {}).get("config_overrides", {})
		super().__init__(model=model, **(config_overrides or {}), **kwargs)
		self.model_name = model
		self.config_overrides = config_overrides or {}
		self.rb_model = None
		self.trainer = None
		self.uid_map: Dict[int, int] = {}
		self.iid_map: Dict[int, int] = {}
		self._cfg = None
		# remember train dataset location for inference-time init
		self._dataset_name: str | None = None
		self._dataset_dir: str | None = None

	@staticmethod
	def _numpy_compat() -> None:
		import numpy as np  # type: ignore
		if not hasattr(np, "float_"):
			setattr(np, "float_", np.float64)  # type: ignore[attr-defined]
		if not hasattr(np, "float"):
			setattr(np, "float", float)  # type: ignore[attr-defined]
		if not hasattr(np, "int_"):
			setattr(np, "int_", np.int64)  # type: ignore[attr-defined]
		if not hasattr(np, "int"):
			setattr(np, "int", int)  # type: ignore[attr-defined]
		if not hasattr(np, "bool_"):
			setattr(np, "bool_", np.bool_)  # type: ignore[attr-defined]
		if not hasattr(np, "complex_"):
			setattr(np, "complex_", np.complex128)  # type: ignore[attr-defined]
		if not hasattr(np, "complex"):
			setattr(np, "complex", complex)  # type: ignore[attr-defined]
		if not hasattr(np, "unicode_"):
			setattr(np, "unicode_", np.str_)  # type: ignore[attr-defined]

	def _ensure_imports(self):
		# NumPy 2.0 compatibility for legacy aliases used by some libs
		import numpy as np  # type: ignore
		if not hasattr(np, "float_"):
			setattr(np, "float_", np.float64)  # type: ignore[attr-defined]
		if not hasattr(np, "int_"):
			setattr(np, "int_", np.int64)  # type: ignore[attr-defined]
		if not hasattr(np, "bool_"):
			setattr(np, "bool_", np.bool_)  # type: ignore[attr-defined]
		if not hasattr(np, "complex_"):
			setattr(np, "complex_", np.complex128)  # type: ignore[attr-defined]
		if not hasattr(np, "unicode_"):
			setattr(np, "unicode_", np.str_)  # type: ignore[attr-defined]
		# Proceed with RecBole imports
		from recbole.config import Config
		from recbole.data import create_dataset
		from recbole.data.utils import data_preparation
		from recbole.trainer import Trainer
		from recbole.utils import get_model
		from recbole.data.interaction import Interaction
		import torch
		return Config, create_dataset, data_preparation, Trainer, get_model, Interaction, torch

	def _write_inter_file(self, path: str, df: pl.DataFrame) -> None:
		with open(path, "w") as f:
			# RecBole expects field types in header, e.g., token/float
			f.write("user_id:token\titem_id:token\tlabel:float\n")
			for u, it, y in df.select(["uid", "item_id", "label"]).iter_rows():
				u_id = self.uid_map.get(int(u))
				it_id = self.iid_map.get(int(it))
				if u_id is None or it_id is None:
					continue
				lbl = 0.0 if y is None else float(y)
				f.write(f"{u_id}\t{it_id}\t{lbl}\n")

	def _build_tmp_dataset(self, train_df: pl.DataFrame, valid_df: pl.DataFrame | None, tmp_dir: str) -> None:
		os.makedirs(tmp_dir, exist_ok=True)
		uvals = train_df["uid"].unique().to_list()
		ivals = train_df["item_id"].unique().to_list()
		self.uid_map = {int(u): i for i, u in enumerate(uvals)}
		self.iid_map = {int(i): j for j, i in enumerate(ivals)}
		dataset_name = "tmp_dataset"
		base_dir = os.path.join(tmp_dir, dataset_name)
		os.makedirs(base_dir, exist_ok=True)
		self._write_inter_file(os.path.join(base_dir, f"{dataset_name}.train.inter"), train_df)
		if valid_df is not None:
			self._write_inter_file(os.path.join(base_dir, f"{dataset_name}.valid.inter"), valid_df)

	def fit(self, train_df: pl.DataFrame, valid_df: pl.DataFrame | None = None) -> "RecBoleAdapter":
		Config, create_dataset, data_preparation, Trainer, get_model, Interaction, torch = self._ensure_imports()
		tmp_path = ".recbole_tmp"
		if os.path.exists(tmp_path):
			shutil.rmtree(tmp_path)
		os.makedirs(tmp_path, exist_ok=True)
		self._build_tmp_dataset(train_df, valid_df, tmp_path)
		cfg_dict = {
			"model": self.model_name,
			"dataset": "tmp_dataset",
			"data_path": tmp_path,
			"field_separator": "\t",
			"USER_ID_FIELD": "user_id",
			"ITEM_ID_FIELD": "item_id",
			"LABEL_FIELD": "label",
			"load_col": {"inter": ["user_id", "item_id", "label"]},
			"benchmark_filename": ["train"] + (["valid"] if valid_df is not None else []),
			"neg_sampling": None,
			"epochs": 10,
			"train_batch_size": 1024,
			"eval_batch_size": 4096,
			"learning_rate": 1e-3,
			"stopping_step": 3,
			"device": "cpu",
			"checkpoint_dir": os.path.join(tmp_path, "checkpoints"),
			"save_dataset": False,
		}
		cfg_dict.update(self.config_overrides)
		self._cfg = Config(model=self.model_name, dataset="tmp_dataset", config_dict=cfg_dict)
		dataset = create_dataset(self._cfg)
		# remember dataset dirs for inference-time
		self._dataset_name = self._cfg["dataset"]  # type: ignore[index]
		self._dataset_dir = os.path.join(tmp_path, self._dataset_name)
		train_data, valid_data = data_preparation(self._cfg, dataset)
		ModelClass = get_model(self._cfg["model"])
		self.rb_model = ModelClass(self._cfg, train_data.dataset).to(self._cfg["device"])
		self.trainer = Trainer(self._cfg, self.rb_model)
		self.trainer.fit(train_data, valid_data)
		return self

	def predict(self, df: pl.DataFrame) -> np.ndarray:
		assert self.rb_model is not None, "Model is not fitted"
		_, _, _, _, _, Interaction, torch = self._ensure_imports()
		users = df["uid"].to_list()
		items = df["item_id"].to_list()
		scores: List[float] = []
		# Safely read eval_batch_size from RecBole Config
		batch_size = 4096
		try:
			batch_size = int(self._cfg["eval_batch_size"])  # type: ignore[index]
		except Exception:
			batch_size = 4096
		for i in range(0, len(users), batch_size):
			batch_u = users[i : i + batch_size]
			batch_i = items[i : i + batch_size]
			u_ids, i_ids, mask = [], [], []
			for u, it in zip(batch_u, batch_i):
				u_id = self.uid_map.get(int(u))
				it_id = self.iid_map.get(int(it))
				ok = u_id is not None and it_id is not None
				u_ids.append(-1 if not ok else u_id)
				i_ids.append(-1 if not ok else it_id)
				mask.append(ok)
			valid_any = any(mask)
			if valid_any:
				u_tensor = torch.tensor([ui for ui, m in zip(u_ids, mask) if m], device=self._cfg["device"])  # type: ignore[index]
				i_tensor = torch.tensor([ii for ii, m in zip(i_ids, mask) if m], device=self._cfg["device"])  # type: ignore[index]
				inter = Interaction({"user_id": u_tensor, "item_id": i_tensor})
				with torch.no_grad():
					y = self.rb_model.forward(inter)
					y = y.detach().cpu().view(-1).numpy().tolist()
				it_scores = iter(y)
				for m in mask:
					scores.append(next(it_scores) if m else 0.0)
			else:
				scores.extend([0.0] * len(batch_u))
		return np.array(scores, dtype=float)

	def save(self, path: str) -> None:
		os.makedirs(path, exist_ok=True)
		with open(os.path.join(path, "uid_map.json"), "w") as f:
			json.dump(self.uid_map, f)
		with open(os.path.join(path, "iid_map.json"), "w") as f:
			json.dump(self.iid_map, f)
		_, _, _, _, _, _, torch = self._ensure_imports()
		if self.rb_model is not None:
			torch.save(self.rb_model.state_dict(), os.path.join(path, "model.pth"))

		# copy dataset files used during training for inference-time dataset init
		dataset_name_saved = None
		if self._dataset_dir and os.path.isdir(self._dataset_dir) and self._dataset_name:
			dataset_root = os.path.join(path, "dataset")
			dest_dir = os.path.join(dataset_root, self._dataset_name)
			os.makedirs(dataset_root, exist_ok=True)
			import shutil as _sh
			if os.path.isdir(dest_dir):
				_sh.rmtree(dest_dir)
			_sh.copytree(self._dataset_dir, dest_dir)
			dataset_name_saved = self._dataset_name

		def _jsonable(x):
			if isinstance(x, (str, int, float, bool)) or x is None:
				return x
			if isinstance(x, dict):
				return {str(k): _jsonable(v) for k, v in x.items()}
			if isinstance(x, (list, tuple)):
				return [_jsonable(v) for v in x]
			# fallback to string representation
			return str(x)

		cfg_to_save = {
			"model": self.model_name,
			"config_overrides": _jsonable(self.config_overrides),
			"dataset_name": dataset_name_saved,
			"dataset_rel_root": "dataset" if dataset_name_saved else None,
		}
		try:
			cfg_to_save["device"] = str(self._cfg["device"])  # type: ignore[index]
			cfg_to_save["eval_batch_size"] = int(self._cfg["eval_batch_size"])  # type: ignore[index]
		except Exception:
			pass
		with open(os.path.join(path, "config.json"), "w") as f:
			json.dump(cfg_to_save, f)

	@classmethod
	def load(cls, path: str) -> "RecBoleAdapter":
		# ensure NumPy aliases before importing recbole
		RecBoleAdapter._numpy_compat()
		from recbole.config import Config
		from recbole.data import create_dataset
		from recbole.utils import get_model
		import torch
		obj = cls()
		with open(os.path.join(path, "uid_map.json")) as f:
			obj.uid_map = {int(k): int(v) for k, v in json.load(f).items()}
		with open(os.path.join(path, "iid_map.json")) as f:
			obj.iid_map = {int(k): int(v) for k, v in json.load(f).items()}
		cfg_path = os.path.join(path, "config.json")
		if os.path.exists(cfg_path):
			with open(cfg_path) as f:
				cfg_dict = json.load(f)
			obj.model_name = cfg_dict.get("model", obj.model_name)
			config_overrides = cfg_dict.get("config_overrides", {}) or {}
			# point to saved dataset copy if available
			dataset_name = cfg_dict.get("dataset_name")
			dataset_rel_root = cfg_dict.get("dataset_rel_root") or "dataset"
			data_path = os.path.join(path, dataset_rel_root)
			# auto-discover dataset folder if name is missing
			if (not dataset_name) and os.path.isdir(data_path):
				candidates = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
				if candidates:
					dataset_name = candidates[0]
			if dataset_name:
				config_overrides = dict(config_overrides)
				config_overrides.update({
					"dataset": dataset_name,
					"data_path": data_path,
					"benchmark_filename": ["train"],
					"load_col": {"inter": ["user_id", "item_id", "label"]},
					"USER_ID_FIELD": "user_id",
					"ITEM_ID_FIELD": "item_id",
					"LABEL_FIELD": "label",
				})
			# recreate Config and Dataset
			obj._cfg = Config(model=obj.model_name, dataset=(dataset_name or "tmp_dataset_infer"), config_dict=config_overrides)
			dataset_obj = create_dataset(obj._cfg)
			ModelClass = get_model(obj._cfg["model"])
			obj.rb_model = ModelClass(obj._cfg, dataset_obj).to(obj._cfg["device"])  # type: ignore[index]
			state = torch.load(os.path.join(path, "model.pth"), map_location=obj._cfg["device"])  # type: ignore[index]
			obj.rb_model.load_state_dict(state)
			obj.rb_model.eval()
		return obj

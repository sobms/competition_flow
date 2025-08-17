from __future__ import annotations
from typing import Any
import os
import numpy as np
import polars as pl
from ml.core.interfaces import BaseModel

# RecBole is heavy; import lazily


class RecBoleAdapter(BaseModel):
	def __init__(self, model: str = "BPR", config_overrides: dict[str, Any] | None = None, **kwargs: Any) -> None:
		super().__init__(model=model, **(config_overrides or {}), **kwargs)
		self.model_name = model
		self.config_overrides = config_overrides or {}
		self.rb_model = None
		self.trainer = None
		self.uid_map: dict[int, int] = {}
		self.iid_map: dict[int, int] = {}

	def _ensure_imports(self):
		global create_dataset, data_preparation, Trainer, get_model, Config
		from recbole.config import Config
		from recbole.data import create_dataset
		from recbole.data.utils import data_preparation
		from recbole.trainer import Trainer
		from recbole.model.general_recommender import BPR
		from recbole.utils import get_model
		return Config, create_dataset, data_preparation, Trainer, get_model

	def fit(self, train_df: pl.DataFrame, valid_df: pl.DataFrame | None = None) -> "RecBoleAdapter":
		Config, create_dataset, data_preparation, Trainer, get_model = self._ensure_imports()
		# Build RecBole dataset from interactions
		tmp_path = ".recbole_tmp"
		os.makedirs(tmp_path, exist_ok=True)
		# Map uids/items to contiguous ids
		uvals = train_df["uid"].unique().to_list()
		ivals = train_df["item_id"].unique().to_list()
		self.uid_map = {int(u): i for i, u in enumerate(uvals)}
		self.iid_map = {int(i): j for j, i in enumerate(ivals)}
		inter_list = []
		for row in train_df.select(["uid", "item_id"]).iter_rows():
			u, it = int(row[0]), int(row[1])
			inter_list.append(f"{self.uid_map[u]}\t{self.iid_map[it]}")
		inter_file = os.path.join(tmp_path, "train.inter")
		with open(inter_file, "w") as f:
			f.write("user_id\titem_id\n")
			f.write("\n".join(inter_list))
		cfg = Config(model=self.model_name, dataset="tmp_dataset", config_dict={
			"data_path": tmp_path,
			"USER_ID_FIELD": "user_id",
			"ITEM_ID_FIELD": "item_id",
			"load_col": {"inter": ["user_id", "item_id"]},
			**self.config_overrides,
		})
		dataset = create_dataset(cfg)
		train_data, valid_data, test_data = data_preparation(cfg, dataset)
		ModelClass = get_model(cfg["model"])
		self.rb_model = ModelClass(cfg, train_data.dataset).to(cfg["device"])
		self.trainer = Trainer(cfg, self.rb_model)
		self.trainer.fit(train_data, valid_data)
		return self

	def predict(self, df: pl.DataFrame):
		assert self.rb_model is not None and self.trainer is not None
		# Score each (uid,item) pair; map ids; O(N) loop for simplicity
		scores = []
		for u, it in df.select(["uid", "item_id"]).iter_rows():
			u_id = self.uid_map.get(int(u), None)
			it_id = self.iid_map.get(int(it), None)
			if u_id is None or it_id is None:
				scores.append(0.0)
				continue
			import torch
			u_tensor = torch.tensor([u_id])
			it_tensor = torch.tensor([it_id])
			with torch.no_grad():
				s = self.rb_model.full_sort_predict(u_tensor)
				scores.append(float(s[0, it_id].cpu().item()))
		return np.array(scores)

	def save(self, path: str) -> None:
		os.makedirs(path, exist_ok=True)
		# RecBole trainer can save checkpoint; fall back to torch
		if hasattr(self.trainer, "save_checkpoint"):
			self.trainer.save_checkpoint(path)
		# Save id maps
		import json
		with open(os.path.join(path, "uid_map.json"), "w") as f:
			json.dump(self.uid_map, f)
		with open(os.path.join(path, "iid_map.json"), "w") as f:
			json.dump(self.iid_map, f)

	@classmethod
	def load(cls, path: str) -> "RecBoleAdapter":
		obj = cls()
		import json
		with open(os.path.join(path, "uid_map.json")) as f:
			obj.uid_map = {int(k): int(v) for k, v in json.load(f).items()}
		with open(os.path.join(path, "iid_map.json")) as f:
			obj.iid_map = {int(k): int(v) for k, v in json.load(f).items()}
		# Loading full RecBole model checkpoint programmatically can depend on config; users can retrain in CI
		return obj

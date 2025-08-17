from __future__ import annotations
from typing import Any
import os
import lightgbm as lgb
import polars as pl
from ml.core.interfaces import BaseModel


class LGBMRegressor(BaseModel):
	def __init__(self, **params: Any) -> None:
		super().__init__(**params)
		self.params = params
		self.model: lgb.Booster | None = None

	def fit(self, train_df: pl.DataFrame, valid_df: pl.DataFrame | None = None) -> "LGBMRegressor":
		feature_cols = [c for c in train_df.columns if c not in {"label"}]
		train = lgb.Dataset(train_df[feature_cols].to_pandas(), label=train_df["label"].to_pandas())
		valid_sets = []
		if valid_df is not None:
			valid = lgb.Dataset(valid_df[feature_cols].to_pandas(), label=valid_df["label"].to_pandas())
			valid_sets = [valid]
		self.model = lgb.train(self.params, train, valid_sets=valid_sets)
		return self

	def predict(self, df: pl.DataFrame):
		assert self.model is not None, "Model is not fitted"
		feature_cols = [c for c in df.columns if c not in {"label"}]
		return self.model.predict(df[feature_cols].to_pandas())

	def save(self, path: str) -> None:
		assert self.model is not None, "Nothing to save"
		os.makedirs(os.path.dirname(path) if path.endswith(".txt") else path, exist_ok=True)
		if path.endswith(".txt"):
			self.model.save_model(path)
		else:
			self.model.save_model(os.path.join(path, "model.txt"))

	@classmethod
	def load(cls, path: str) -> "LGBMRegressor":
		obj = cls()
		model_path = path
		if not (model_path.endswith(".txt") or model_path.endswith(".json")):
			model_path = os.path.join(path, "model.txt")
		obj.model = lgb.Booster(model_file=model_path)
		return obj

from __future__ import annotations
from typing import Any
import os
import polars as pl
from ml.core.interfaces import BaseModel


class DummyModel(BaseModel):
	def __init__(self, **kwargs: Any) -> None:
		super().__init__(**kwargs)
		self.user_mean: pl.DataFrame | None = None

	def fit(self, train_df: pl.DataFrame, valid_df: pl.DataFrame | None = None) -> "DummyModel":
		if "label" in train_df.columns:
			self.user_mean = train_df.group_by("uid").agg(pl.col("label").mean().alias("mean_label"))
		else:
			self.user_mean = pl.DataFrame({"uid": train_df["uid"].unique(), "mean_label": 0.5})
		return self

	def predict(self, df: pl.DataFrame):
		assert self.user_mean is not None, "Model is not fitted"
		out = df.join(self.user_mean, on="uid", how="left")
		return out["mean_label"].fill_null(0.5)

	def save(self, path: str) -> None:
		# path can be either directory or .parquet file
		if path.endswith(".parquet"):
			os.makedirs(os.path.dirname(path), exist_ok=True)
			assert self.user_mean is not None, "Nothing to save"
			self.user_mean.write_parquet(path)
		else:
			os.makedirs(path, exist_ok=True)
			assert self.user_mean is not None, "Nothing to save"
			self.user_mean.write_parquet(os.path.join(path, "user_mean.parquet"))

	@classmethod
	def load(cls, path: str) -> "DummyModel":
		obj = cls()
		parquet_path = path
		if not parquet_path.endswith(".parquet"):
			parquet_path = os.path.join(path, "user_mean.parquet")
		obj.user_mean = pl.read_parquet(parquet_path)
		return obj

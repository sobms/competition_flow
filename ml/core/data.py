from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import polars as pl
import numpy as np
import os


@dataclass
class DataPaths:
	raw_dir: str
	splits_dir: str
	features_dir: str


def load_raw(cfg: Dict[str, Any]) -> pl.DataFrame:
	source = cfg["data"]["source"]
	if source == "hf":
		paths = cfg["data"]["hf"]
		df = pl.read_parquet(paths["flat_path"])  # minimal: flat interactions
		# derive binary label: like vs not-like, if event_type present
		if "event_type" in df.columns:
			df = df.with_columns(
				(pl.col("event_type").cast(pl.Utf8) == "like").cast(pl.UInt8).alias("label")
			)
		return df
	else:
		raise ValueError(f"Unknown data source: {source}")


def make_splits(df: pl.DataFrame, cfg: Dict[str, Any], paths: DataPaths) -> None:
	os.makedirs(paths.splits_dir, exist_ok=True)
	type_ = cfg["split"]["type"]
	if type_ != "time_holdout":
		raise ValueError("Only time_holdout split is implemented in scaffold")
	train_end = cfg["split"]["train_end_ts"]
	valid_end = cfg["split"]["valid_end_ts"]
	test_end = cfg["split"]["test_end_ts"]

	train_df = df.filter(pl.col("timestamp") <= train_end)
	valid_df = df.filter((pl.col("timestamp") > train_end) & (pl.col("timestamp") <= valid_end))
	test_df = df.filter((pl.col("timestamp") > valid_end) & (pl.col("timestamp") <= test_end))

	train_df.write_parquet(os.path.join(paths.splits_dir, "train.parquet"))
	valid_df.write_parquet(os.path.join(paths.splits_dir, "valid.parquet"))
	test_df.write_parquet(os.path.join(paths.splits_dir, "test.parquet"))


def load_split(paths: DataPaths, name: str) -> pl.DataFrame:
	path = os.path.join(paths.splits_dir, f"{name}.parquet")
	return pl.read_parquet(path)


def build_features(cfg: Dict[str, Any], paths: DataPaths) -> Tuple[pl.DataFrame, pl.DataFrame]:
	# Minimal scaffold: identity features
	os.makedirs(paths.features_dir, exist_ok=True)
	train = load_split(paths, "train")
	valid = load_split(paths, "valid")
	pl.DataFrame(train).write_parquet(os.path.join(paths.features_dir, "train.parquet"))
	pl.DataFrame(valid).write_parquet(os.path.join(paths.features_dir, "valid.parquet"))
	return train, valid

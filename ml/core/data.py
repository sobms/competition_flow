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
		# derive binary label if not present
		if "label" not in df.columns:
			if "played_ratio_pct" in df.columns:
				df = df.with_columns((pl.col("played_ratio_pct") >= 50).cast(pl.UInt8).alias("label"))
			else:
				df = df.with_columns(pl.lit(0, dtype=pl.UInt8).alias("label"))
		return df
	elif source == "synthetic":
		return _generate_synthetic(cfg["data"]["synthetic"]) 
	else:
		raise ValueError(f"Unknown data source: {source}")


def _generate_synthetic(p: Dict[str, Any]) -> pl.DataFrame:
	rng = np.random.default_rng(42)
	uids = rng.integers(1, p["num_users"] + 1, size=p["interactions"])
	item_ids = rng.integers(1, p["num_items"] + 1, size=p["interactions"])
	timestamps = rng.integers(p["start_ts"], p["end_ts"] + 1, size=p["interactions"])
	played_ratio = rng.integers(0, 101, size=p["interactions"]).astype(np.uint16)
	labels = (played_ratio >= 50).astype(np.uint8)
	return pl.DataFrame({
		"uid": uids.astype(np.uint32),
		"item_id": item_ids.astype(np.uint32),
		"timestamp": timestamps.astype(np.uint32),
		"played_ratio_pct": played_ratio,
		"label": labels,
	})


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
	# Minimal scaffold: identity features + simple group column for ranking
	os.makedirs(paths.features_dir, exist_ok=True)
	train = load_split(paths, "train")
	valid = load_split(paths, "valid")
	# Example of basic features; here we just keep columns
	pl.DataFrame(train).write_parquet(os.path.join(paths.features_dir, "train.parquet"))
	pl.DataFrame(valid).write_parquet(os.path.join(paths.features_dir, "valid.parquet"))
	return train, valid

from __future__ import annotations

from typing import Any, Dict
from pathlib import Path

import polars as pl
from tqdm import tqdm

from .base import BaseFeaturesFactory
from .io import iter_parquet_shards, scan_selected


class UsersFeaturesFactory(BaseFeaturesFactory):
	def __init__(self, cfg: Dict[str, Any]) -> None:
		super().__init__(cfg)
		self.num_shards = int(cfg.get("shards", 128))

	def build(self, split: str) -> None:
		prefix = self.cfg.get("users_out", "users_out")
		tasks = []
		for split in ["train", "valid"]:
			tracker_dir = Path(self.cfg[split]["tracker_dir"])  # required per split
			tasks.extend([(self.cfg, str(s), prefix) for s in iter_parquet_shards(tracker_dir, "shard=*.parquet")])
		for task in tqdm(tasks, desc=f"{self.__class__.__name__}"):
			users_worker(task)

	def _base_user_features(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
		# Unique user ids per shard
		return df.select([
			pl.col("user_id").cast(pl.Int64),
		]).unique()

	def _write_parquet(self, df: pl.DataFrame, prefix: str, src_path: Path) -> None:
		out_dir = self.out_root / prefix / "raw"
		out_dir.mkdir(parents=True, exist_ok=True)
		df.write_parquet(str(out_dir / src_path.name))


def users_worker(task: tuple) -> None:
	cfg, shard_path, prefix = task
	f = UsersFeaturesFactory(cfg)
	lf = scan_selected(Path(shard_path), [
		"user_id",
	])
	lf_feat = f._base_user_features(lf)
	df_feat = lf_feat.collect(streaming=True)
	f._write_parquet(df_feat, prefix, Path(shard_path))



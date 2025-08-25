from __future__ import annotations

from typing import Any, Dict
from pathlib import Path

import polars as pl
from tqdm import tqdm

from .base import BaseFeaturesFactory
from .io import iter_parquet_shards, scan_selected


class ItemsFeaturesFactory(BaseFeaturesFactory):
	def __init__(self, cfg: Dict[str, Any]) -> None:
		super().__init__(cfg)
		self.items_dir = Path(cfg["items_dir"])  # required
		self.num_shards = int(cfg.get("shards", 128))

	def build(self, split: str) -> None:
		prefix = self.cfg.get("items_out", "items_features")
		tasks = [(self.cfg, str(s), prefix) for s in iter_parquet_shards(self.items_dir, "part-*.parquet")]
		# self._parallel_map(tasks, items_worker)
		for task in tqdm(tasks, desc=f"{self.__class__.__name__}"):
			items_worker(task)

	def _base_item_features(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
		return df.select([
			pl.col("item_id").cast(pl.Int64),
			pl.col("fclip_embed"),
			pl.col("catalogid").cast(pl.Utf8),
			pl.col("variant_id").cast(pl.Utf8),
			pl.col("model_id").cast(pl.Utf8),
		])
	
	def _write_parquet(self, df: pl.DataFrame, prefix: str, src_path: Path) -> None:
		out_dir = self.out_root / prefix / "raw"
		out_dir.mkdir(parents=True, exist_ok=True)
		df.write_parquet(str(out_dir / src_path.name))

def items_worker(task: tuple) -> None:
	cfg, shard_path, prefix = task
	f = ItemsFeaturesFactory(cfg)
	lf = scan_selected(Path(shard_path), [
		"item_id","catalogid","variant_id","model_id","itemname","attributes","fclip_embed"
	])
	lf_feat = f._base_item_features(lf)
	df_feat = lf_feat.collect(streaming=True)
	f._write_parquet(df_feat, prefix, Path(shard_path))

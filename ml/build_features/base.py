from __future__ import annotations

from typing import Any, Dict, Callable, List, Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import signal

import polars as pl
from tqdm import tqdm

from .io import write_shard, out_shard_path


class BaseFeaturesFactory:
	def __init__(self, cfg: Dict[str, Any]) -> None:
		self.cfg = cfg
		self.num_workers: int = int(cfg.get("num_workers", 4))
		self.batch_rows: int = int(cfg.get("batch_rows", 2_000_000))
		self.out_root = Path(cfg.get("out_root", "data/processed_features"))

	def build(self, split: str) -> None:
		raise NotImplementedError

	def _parallel_map(self, tasks: List[Tuple], worker: Callable[[Tuple], None]) -> None:
		print(f"[build_features] {self.__class__.__name__}: starting {len(tasks)} task(s) with {self.num_workers} worker(s)...")
		if self.num_workers <= 1:
			for t in tqdm(tasks, desc=f"{self.__class__.__name__}"):
				worker(t)
			return
		interrupted = False
		def signal_handler(signum, frame):
			nonlocal interrupted
			print(f"\n[build_features] Received signal {signum}, interrupting...")
			interrupted = True
			# ProcessPoolExecutor автоматически завершит процессы при выходе из контекста
		
		original_sigint = signal.signal(signal.SIGINT, signal_handler)
		original_sigterm = signal.signal(signal.SIGTERM, signal_handler)
		try:
			with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
				# executor.map() проще, но менее гибкий
				results_iter = ex.map(worker, tasks)
				
				# Потребляем результаты с прогресс-баром
				for _ in tqdm(results_iter, total=len(tasks), desc=f"{self.__class__.__name__}"):
					if interrupted:
						print("[build_features] Processing interrupted")
						break
		except KeyboardInterrupt:
			print("[build_features] Interrupted by user")
			raise
		finally:
			signal.signal(signal.SIGINT, original_sigint)
			signal.signal(signal.SIGTERM, original_sigterm)

	def _write_sharded(self, df: pl.DataFrame, key_col: str, prefix: str, num_shards: int) -> None:
		if df.height == 0:
			return
		# Compute shard index vectorized to avoid Python loops over rows
		df_with_shard = df.with_columns(((pl.col(key_col).cast(pl.Int64) % num_shards).alias("_shard")))
		for shard_idx in range(num_shards):
			part = df_with_shard.filter(pl.col("_shard") == shard_idx).drop("_shard")
			if part.height == 0:
				continue
			out_path = out_shard_path(self.out_root, prefix, shard_idx)
			# write_parquet_atomic(part, out_path)
			write_shard(part, out_path)

	def _build_basis_from_interactions(self, df: pl.LazyFrame) -> pl.LazyFrame:
		"""Build per-day interaction basis and derived user basis.
		Вuilds basis only for that shard DF and returns df_inter. Accepts DataFrame or LazyFrame.
		"""
		lf = df.select([
			pl.col("user_id").cast(pl.Int64),
			pl.col("item_id").cast(pl.Int64),
			pl.col("timestamp").dt.date().alias("date"),
		])
		return lf.select(["date","user_id","item_id"]).unique()
		


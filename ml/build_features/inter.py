from __future__ import annotations

from typing import Any, Dict, List, Tuple
from pathlib import Path

import polars as pl
from tqdm import tqdm

# Enable global string cache to optimize categorical-like operations
pl.enable_string_cache()

from .base import BaseFeaturesFactory
from .io import iter_parquet_shards, scan_selected


class InteractionsFeaturesFactory(BaseFeaturesFactory):
	def __init__(self, cfg: Dict[str, Any]) -> None:
		super().__init__(cfg)
		self.num_shards = int(cfg.get("shards", 128))
		self.windows = [w.strip().lower() for w in cfg.get("windows", ["7d", "30d"])]
		self.half_life = int(cfg.get("time_decay", {}).get("half_life_days", 7)) * 86400
		# outputs
		self.inter_out = cfg.get("inter_out", "inter_features")
		self.target_out = cfg.get("target_out", "target")
		# target config
		tcfg = (cfg.get("target") or {})
		self.target_type = tcfg.get("type", "combined")
		self.target_horizon_days = int(tcfg.get("horizon_days", 0))
		self.drop_duplicate_targets = bool(tcfg.get("drop_duplicate_targets", False))

	# -------- label properties --------
	@property
	def target_competition_expr(self) -> pl.Expr:
		return (pl.col("last_status") == "delivered_orders").cast(pl.Int8)

	@property
	def combined_label_expr(self) -> pl.Expr:
		return (
			pl.when(pl.col("last_status") == "delivered_orders").then(1)
			.when(pl.col("action_type") == "to_cart").then(1)
			.otherwise(0)
		).cast(pl.Int8)

	def _label_expr(self) -> pl.Expr:
		lt = self.target_type
		if lt == "delivered":
			return self.target_competition_expr
		if lt == "combined":
			return self.combined_label_expr
		raise ValueError(f"Invalid target type: {lt}")

	def build(self, split: str) -> None:
		tracker_dir = Path(self.cfg[split]["tracker_dir"])  # required per split
		# Build task tuples for process-safe worker
		tasks: List[Tuple[Dict[str, Any], str, str]] = [
			(self.cfg, split, str(s)) for s in iter_parquet_shards(tracker_dir, "shard=*.parquet")
		]
		for task in tqdm(tasks, desc=f"{self.__class__.__name__}"):
			inter_worker(task)
		# self._parallel_map(tasks, inter_worker)

	def _compute_target_for_shard(self, lf_bi: pl.LazyFrame, df_tracker_shard: pl.LazyFrame, orders_shard_path: Path) -> pl.LazyFrame:
		label_expr = self._label_expr().alias("label")
		lf_tracker = df_tracker_shard.select([
			pl.col("user_id").cast(pl.Int64),
			pl.col("item_id").cast(pl.Int64),
			pl.col("timestamp").dt.date().alias("date_t"),
			pl.col("action_type"),
		])
		if orders_shard_path.exists():
			lf_orders = pl.scan_parquet(str(orders_shard_path)).with_columns(pl.col("last_status_timestamp").dt.date().alias("date_o"))
			lf_events = lf_tracker.join(
				lf_orders.select(["user_id","item_id","last_status","date_o"]),
				left_on=["user_id","item_id","date_t"],
				right_on=["user_id","item_id","date_o"],
				how="left",
			).select([
				pl.col("user_id"), pl.col("item_id"), pl.col("date_t").alias("date"), label_expr,
			]).filter(pl.col("label") > 0)
		else:
			lf_events = lf_tracker.select([
				pl.col("user_id"), pl.col("item_id"), pl.col("date_t").alias("date"), label_expr,
			]).filter(pl.col("label") > 0)
		lf_first = lf_events.group_by(["user_id","item_id"]).agg(pl.col("date").min().alias("first_pos_date"))
		lf_join = lf_bi.join(lf_first, on=["user_id","item_id"], how="left")
		if self.target_horizon_days > 0:
			lf_join = lf_join.with_columns((pl.col("date") + pl.duration(days=self.target_horizon_days)).alias("limit")).with_columns(
				pl.when(pl.col("first_pos_date").is_not_null() & (pl.col("first_pos_date") >= pl.col("date")) & (pl.col("first_pos_date") <= pl.col("limit"))).then(1).otherwise(0).alias("target")
			).drop("limit")
		else:
			lf_join = lf_join.with_columns(
				pl.when(pl.col("first_pos_date").is_not_null() & (pl.col("first_pos_date") >= pl.col("date"))).then(1).otherwise(0).alias("target")
			)
		lf_target = lf_join.select(["date","user_id","item_id","target"])  # LazyFrame
		# optional: drop duplicate targets by (user_id,item_id) keeping max target and attach to basis
		if self.drop_duplicate_targets:
			lf_best = lf_target.group_by(["user_id","item_id"]).agg(pl.col("target").max().alias("target"))

		return lf_bi.join(lf_best, on=["user_id","item_id"], how="inner").select(["date","user_id","item_id","target"])

	def _compute_action_types_aggregates(self, lf_tracker_shard: pl.LazyFrame) -> pl.LazyFrame:
		# daily counts per (date, user, item, action) in long form
		lf_long = lf_tracker_shard.select([
			pl.col("user_id").cast(pl.Int64),
			pl.col("item_id").cast(pl.Int64),
			pl.col("action_type").cast(pl.Categorical),
			pl.col("timestamp").dt.date().alias("date"),
		])
		lf_counts = lf_long.group_by(["date","user_id","item_id","action_type"]).agg(pl.len().alias("cnt"))
		# derive action list with a small eager step (only unique values)
		acts_df = lf_counts.select(pl.col("action_type").unique()).collect(streaming=True)
		actions = [v for v in acts_df.get_column("action_type").to_list() if v is not None]
		# wide aggregation without pivot: conditional sums per action type
		lf_daily = lf_counts.group_by(["date","user_id","item_id"]).agg([
			(
				pl.when(pl.col("action_type") == pl.lit(a))
				 .then(pl.col("cnt"))
				 .otherwise(0)
				 .sum()
				 .alias(f"cnt_{a}")
			)
			for a in actions
		]).fill_null(0)
		agg_columns = [f"cnt_{a}" for a in actions]
		
		outs: List[pl.LazyFrame] = []
		for w in self.windows:
			agg = lf_daily.sort(["user_id","item_id","date"]).group_by_dynamic(
				index_column="date", every="1d", period=w, closed="left", by=["user_id","item_id"]
			).agg([
				pl.col(col).sum().alias(f"{col}_sum_{w}_past") 
				for col in agg_columns
			])
			outs.append(agg)
		lf_roll = pl.concat(outs, how="diagonal_relaxed") if outs else lf_daily
		return lf_roll

def inter_worker(task: Tuple[Dict[str, Any], str, str]) -> None:
	cfg, split, shard_path = task
	# Recreate factory in child process (spawn-safe) and process shard inline
	f = InteractionsFeaturesFactory(cfg)
	print(f"[inter] Processing shard: {shard_path}")
	lf_tracker = scan_selected(Path(shard_path), ["user_id", "item_id", "action_type", "timestamp"])  # lazy
	basis_inter = f._build_basis_from_interactions(lf_tracker)
	lf_target = f._compute_target_for_shard(basis_inter, lf_tracker, Path(f.cfg[split]["orders_dir"]) / Path(shard_path).name)
	lf_feats = f._compute_action_types_aggregates(lf_tracker)
	# merge basis + target + features (lazy + streaming collect for faster join)
	keys = ["date","user_id","item_id"]
	lf_final = (
		basis_inter.lazy()
		.join(lf_target.select(keys + ["target"]), on=keys, how="inner")
		.join(lf_feats, on=keys, how="left")
		.fill_null(0)
	)
	df_final = lf_final.collect(streaming=True)
	f._write_sharded(df_final, key_col="user_id", prefix=f"{f.inter_out}/{split}", num_shards=f.num_shards)

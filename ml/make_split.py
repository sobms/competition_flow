from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any

import polars as pl
from hydra import initialize, compose
from omegaconf import OmegaConf
from tqdm import tqdm
import shutil

# Constants
USER_COL = "user_id"

# Base paths
RAW_BASE = Path("data/raw_data")
OUT_BASE = Path("data/splits")

# Directories subject to time split: dir -> timestamp column name
TIMESPLIT_DIRS: Dict[str, str] = {
    "ml_ozon_recsys_orders_data": "created_timestamp",
    "ml_ozon_recsys_tracker_data": "timestamp",
}


def _write_sharded(lf: pl.LazyFrame, out_dir: Path, num_shards: int, desc: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for shard in tqdm(range(num_shards), desc=desc):
        shard_lf = (
            lf.with_columns(((pl.col(USER_COL).cast(pl.Int64) % num_shards).alias("_shard")))
              .filter(pl.col("_shard") == shard)
              .drop("_shard")
        )
        fn = out_dir / f"shard={shard:04d}.parquet"
        shard_lf.sink_parquet(str(fn))


def _ts_expr(lf: pl.LazyFrame, col: str) -> pl.Expr:
    schema = lf.schema
    e = pl.col(col)
    if schema.get(col) == pl.Datetime:
        e = e.dt.epoch("s")
    return e.cast(pl.Int64)


def _user_bucket_expr(mod: int, seed: int) -> pl.Expr:
    # простая детерминированная функция разбиения: (user_id + seed) % mod
    return ((pl.col(USER_COL).cast(pl.Int64) + pl.lit(seed, pl.Int64)) % pl.lit(mod, pl.Int64)).alias("_ubkt")


def _fast_mode_slice_lf(from_dir: str, count: int) -> pl.LazyFrame:
    lf = pl.scan_parquet(str(RAW_BASE / from_dir / "*.parquet")).select(pl.col(USER_COL)).unique()
    return lf.sort(USER_COL).limit(count)


def _process_dir(dir_name: str, ts_col: str, split_cfg: Dict[str, Any], users_slice_lf: pl.LazyFrame | None) -> None:
    split_type = split_cfg.get("type", "time_split")
    train_end = int(split_cfg["train_end_ts"])
    valid_end = int(split_cfg["valid_end_ts"])
    num_user_shards = int(split_cfg.get("num_user_shards", 64))
    fast_mode = bool(split_cfg.get("fast_mode", False))
    only_train_mode = bool(split_cfg.get("only_train_mode", False))

    src_glob = RAW_BASE / dir_name / "*.parquet"
    lf = pl.scan_parquet(str(src_glob))
    if fast_mode and users_slice_lf is not None:
        lf = lf.join(users_slice_lf, on=USER_COL, how="semi")

    if only_train_mode:
        _write_sharded(lf, OUT_BASE / "train" / dir_name, num_user_shards, desc=f"{dir_name} train")
        return

    ts = _ts_expr(lf, ts_col)

    if split_type == "time_split":
        train_lf = lf.filter(ts <= train_end)
        valid_lf = lf.filter((ts > train_end) & (ts <= valid_end))
    elif split_type == "time_user_split":
        valid_ratio = float(split_cfg.get("valid_ratio", 0.2))
        user_split_mod = int(split_cfg.get("user_split_mod", 1000))
        user_split_seed = int(split_cfg.get("user_split_seed", 17))
        ubkt = _user_bucket_expr(user_split_mod, user_split_seed)
        valid_th = int(valid_ratio * user_split_mod)
        valid_lf = (
            lf.filter((ts > train_end) & (ts <= valid_end))
              .with_columns(ubkt)
              .filter(pl.col("_ubkt") < valid_th)
              .drop("_ubkt")
        )
        train_lf = (
            lf.filter(ts <= train_end)
              .with_columns(ubkt)
              .filter(pl.col("_ubkt") >= valid_th)
              .drop("_ubkt")
        )
    else:
        raise ValueError(f"Unknown split.type: {split_type}")

    _write_sharded(train_lf, OUT_BASE / "train" / dir_name, num_user_shards, desc=f"{dir_name} train")
    _write_sharded(valid_lf, OUT_BASE / "valid" / dir_name, num_user_shards, desc=f"{dir_name} valid")


def main() -> None:
    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name="config")
    split_cfg: Dict[str, Any] = OmegaConf.to_container(cfg.split, resolve=True)  # type: ignore[assignment]

    fast_mode = bool(split_cfg.get("fast_mode", False))
    fast_count = int(split_cfg.get("fast_mode_users_count", 10000))

    # Cleanup output base before writing
    if OUT_BASE.exists():
        shutil.rmtree(OUT_BASE)

    (OUT_BASE / "train").mkdir(parents=True, exist_ok=True)
    (OUT_BASE / "valid").mkdir(parents=True, exist_ok=True)

    users_slice_lf = _fast_mode_slice_lf("ml_ozon_recsys_orders_data", fast_count) if fast_mode else None

    for dir_name, ts_col in TIMESPLIT_DIRS.items():
        print(f"Processing {dir_name}")
        src_dir = RAW_BASE / dir_name
        if not src_dir.exists():
            raise FileNotFoundError(f"Источник не найден: {src_dir}")
        _process_dir(dir_name, ts_col, split_cfg, users_slice_lf)

    print(
        f"Готово: {OUT_BASE}/train и {OUT_BASE}/valid; шардов: {int(split_cfg.get('num_user_shards', None))}; "
        f"fast_mode={bool(split_cfg.get('fast_mode', False))}; only_train_mode={bool(split_cfg.get('only_train_mode', False))}"
        f"split_type={split_cfg.get('type', None)}"
    )


if __name__ == "__main__":
    main()

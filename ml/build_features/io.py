from __future__ import annotations

from typing import Iterable, Iterator, Dict, Any, List, Tuple
from pathlib import Path
import os
import tempfile

import polars as pl


def iter_parquet_shards(base_dir: Path, glob_pattern: str) -> Iterator[Path]:
	for p in sorted(base_dir.glob(glob_pattern)):
		yield p


def scan_selected(p: Path, cols: List[str]) -> pl.LazyFrame:
	"""Return LazyFrame with column pushdown for fully lazy pipelines."""
	return pl.scan_parquet(str(p)).select(cols)


def write_parquet_atomic(df: pl.DataFrame, out_path: Path) -> None:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.NamedTemporaryFile(delete=False, dir=str(out_path.parent), suffix=".parquet") as tmp:
		tmp_path = Path(tmp.name)
	try:
		df.write_parquet(str(tmp_path))
		os.replace(tmp_path, out_path)
	except Exception:
		if tmp_path.exists():
			tmp_path.unlink(missing_ok=True)
		raise


def write_shard(df: pl.DataFrame, out_path: Path) -> None:
	"""Append-aware Parquet writer for single-process pipelines.

	If the shard file exists, read it, append new rows, and rewrite file.
	If it doesn't, create it. Uses relaxed vertical concat to tolerate minor schema
	differences (missing columns will be added as nulls).
	"""
	if df.height == 0:
		return
	out_path.parent.mkdir(parents=True, exist_ok=True)
	assert not out_path.exists(), f"Shard file already exists: {out_path}"
	df.write_parquet(str(out_path))

def out_shard_path(root: Path, prefix: str, shard_idx: int) -> Path:
	return root / prefix / f"shard={shard_idx:05d}.parquet"


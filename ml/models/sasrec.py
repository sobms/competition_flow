from __future__ import annotations

from typing import Any, List
import polars as pl
from pathlib import Path
from torch.utils.data import DataLoader

from .recbole_base import RecBoleBaseAdapter, _ParquetInteractionsDataset


class _SASRecParquetDataset(_ParquetInteractionsDataset):
	# Adds per-row sequential features: '<item_f>_list' and 'item_length'
	def _enrich_df(self, df: pl.DataFrame) -> pl.DataFrame:
		item_f = self.rb_fields.get("item_f")
		time_f = self.rb_fields.get("time_f")
		max_seq_len = int(getattr(self, "max_seq_len", 50))

		ui_col = "__ui__"
		ii_col = "__ii__"

		# Sort to ensure chronological order within user when time is available
		if time_f and time_f in df.columns:
			df = df.sort([ui_col, time_f])
		else:
			df = df.sort([ui_col])

		ui = df[ui_col].to_numpy()
		ii = df[ii_col].to_numpy()

		item_list_col = f"{item_f}_list"
		length_col = "item_length"

		lists: List[List[int]] = []
		lengths: List[int] = []
		prev_u = None
		hist: List[int] = []
		for j in range(len(ui)):
			u = int(ui[j])
			if prev_u is None or u != prev_u:
				hist = []
				prev_u = u
			seq = hist[-max_seq_len:]
			lists.append(seq.copy())
			lengths.append(len(seq))
			hist.append(int(ii[j]))

		# Register produced feature names so the base can extract numpy arrays for them
		self._extra_feature_names = [item_list_col, length_col]
		return df.with_columns([
			pl.Series(item_list_col, lists),
			pl.Series(length_col, lengths),
		])


class SASRecAdapter(RecBoleBaseAdapter):
	def __init__(self, **params: Any) -> None:
		# Ensure Recbole model is SASRec while keeping other params intact
		p = {**params}
		p.setdefault("model", "SASRec")
		super().__init__(**p)

	def _make_streaming_loader(self, files: List[Path]) -> DataLoader:
		batch_size = int(self.params.get("batch_size", 1024))
		paths = self._recbole_paths({"model": self.params})
		max_seq_len = int(self.params.get("max_seq_length", 50))
		ds = _SASRecParquetDataset(
			files=files,
			rb_fields=self._rb_fields,
			user_id_to_idx=self.user_id_to_idx,
			item_id_to_idx=self.item_id_to_idx,
			items_dir=paths.get("items_dir"),
			users_dir=paths.get("users_dir"),
			float_seq_dtype=str(self.params.get("float_seq_dtype", "float16")),
		)
		setattr(ds, "max_seq_len", max_seq_len)
		return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)

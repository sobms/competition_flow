from __future__ import annotations

from typing import Any, Dict, List, Iterator, Tuple, Iterable
from pathlib import Path

import duckdb
from tqdm import tqdm
from math import ceil

from torch.utils.data import IterableDataset
from recbole.data.interaction import Interaction
import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore
import pyarrow.compute as pc  # type: ignore
import torch

from .recbole_base import RecBoleBaseAdapter


class SASRecAdapter(RecBoleBaseAdapter):
	def __init__(self, **params: Any) -> None:
		# Ensure Recbole model is SASRec while keeping other params intact
		p = {**params}
		p.setdefault("model", "SASRec")
		super().__init__(**p)

	def _create_streaming_dataset(self, dataset: Any, streaming_spec: Dict[str, Any], mode: str):
		user_field = dataset.config["USER_ID_FIELD"]
		item_field = dataset.config["ITEM_ID_FIELD"]
		label_field = dataset.config["LABEL_FIELD"]
		time_field = dataset.config["TIME_FIELD"]

		inter_shards: List[str] = list(
			streaming_spec.get("train_inter_shards" if mode == "train" else "valid_inter_shards") or []
		)
		item_shards: List[str] = list(streaming_spec.get("items_shards") or [])
		item_feature_cols: List[Dict[str, str]] = list(streaming_spec.get("item_feature_cols") or [])
		inter_feature_cols: List[Dict[str, str]] = list(streaming_spec.get("inter_feature_cols") or [])
		batch_size: int = int(streaming_spec.get("batch_size", 1024))
		is_sequential = bool(streaming_spec.get("is_sequential", True))
		max_seqs_per_user: int = int(streaming_spec.get("MAX_SEQS_PER_USER", 5))

		# Register items view once
		duckdb_db_path: str | None = None
		item_cols: List[str] = [c.get("name") for c in item_feature_cols if c.get("name")]
		if item_cols:
			data_path = str(dataset.config["data_path"])  # recbole Config behaves like dict
			duckdb_dir = Path(data_path) / "duckdb"
			duckdb_dir.mkdir(parents=True, exist_ok=True)
			duckdb_db_path = str(duckdb_dir / "items.duckdb")
			items_files_sql = "[" + ",".join([f"'{p}'" for p in item_shards]) + "]" if item_shards else "[]"
			con_reg = duckdb.connect(duckdb_db_path)
			try:
				sel_items_cols = ", ".join([item_field] + item_cols)
				con_reg.execute(
					f"CREATE OR REPLACE VIEW items_all AS SELECT {sel_items_cols} FROM read_parquet({items_files_sql})"
				)
			finally:
				con_reg.close()

		# token maps from dataset
		def _ensure_token_to_id_map(ds: Any, field_name: str) -> Dict[str, int]:
			token_to_id = getattr(ds, "field2token_id", {}).get(field_name, None)
			if hasattr(token_to_id, "get"):
				return dict(token_to_id)  # type: ignore[arg-type]
			id_to_token = getattr(ds, "field2id_token", {}).get(field_name, None)
			mapping: Dict[str, int] = {}
			if isinstance(id_to_token, dict):
				for idx, tok in id_to_token.items():
					try:
						mapping[str(tok)] = int(idx)
					except Exception:
						continue
				return mapping
			try:
				for idx, tok in enumerate(id_to_token):  # type: ignore[assignment]
					mapping[str(tok)] = int(idx)
			except Exception:
				mapping = {}
			return mapping

		user_map: Dict[str, int] = _ensure_token_to_id_map(dataset, user_field)
		item_map: Dict[str, int] = _ensure_token_to_id_map(dataset, item_field)

		# capture outer adapter for use inside iterable
		outer = self
		class _SASRecIterable(IterableDataset):
			def __len__(self) -> int:  # rough estimate for progress
				try:
					con_tmp = duckdb.connect()
					cnt = 0
					for p in inter_shards:
						cnt += int(con_tmp.execute(
							f"SELECT COUNT(*) FROM read_parquet('{p}') WHERE {label_field} > 0"
						).fetchone()[0])
					con_tmp.close()
					from math import ceil
					return max(1, ceil(cnt / max(1, batch_size)))
				except Exception:
					return 1

			def __iter__(self) -> Iterator[Interaction]:
				con_main = duckdb.connect(duckdb_db_path, read_only=True) if duckdb_db_path else None
				try:
					user_seq_counter: Dict[int, int] = {}
					for inter_path in tqdm(inter_shards, desc="Processing inter shards"):
						# SQL: join items, filter positives, order by user/time
						if con_main is None:
							con = duckdb.connect()
						else:
							con = con_main
						it_sel = ", ".join([f"it.{c}" for c in item_cols]) if item_cols else ""
						in_sel = ", ".join([f"i.{c.get('name')}" for c in inter_feature_cols if c.get('name')])
						extra = (", " + in_sel if in_sel else "") + (", " + it_sel if it_sel else "")
						base = f"FROM read_parquet('{inter_path}') i " + ("LEFT JOIN items_all it USING (" + item_field + ") " if item_cols else "")
						sql = (
							f"SELECT i.{user_field} AS {user_field}, i.{item_field} AS {item_field}, i.{time_field} AS {time_field}, i.{label_field} AS {label_field}{extra} "
							+ base + f"WHERE i.{label_field} > 0 ORDER BY i.{user_field}, i.{time_field}"
						)
						tb = con.execute(sql).fetch_arrow_table()
						# reset per-shard user history to accumulate sequences across batches
						history: Dict[int, List[int]] = {}
						_total_batches = max(1, ceil(tb.num_rows / max(1, batch_size)))
						
						_batch_pbar = tqdm(range(0, tb.num_rows, batch_size), total=_total_batches, desc=f"batches:{Path(inter_path).name}", leave=False)
						for start in _batch_pbar:
							sb: pa.Table = tb.slice(start, min(start + batch_size, tb.num_rows) - start)
							if sb.num_rows == 0:
								continue
							u_raw = sb.column(user_field).to_pandas().astype(int)
							i_raw = sb.column(item_field).to_pandas().astype(int)
							um = [user_map.get(str(u)) for u in u_raw]
							im = [item_map.get(str(i)) for i in i_raw]
							keep = [j for j,(uu,ii) in enumerate(zip(um,im)) if uu is not None and ii is not None]
							if not keep:
								continue
							buf_u = [um[j] for j in keep]  # type: ignore[index]
							buf_i = [im[j] for j in keep]  # type: ignore[index]
							y_vals = sb.column(label_field).to_pandas().iloc[keep].astype(float).tolist()
							# features
							feat_cols = [c.get("name") for c in inter_feature_cols if c.get("name")]
							buf_feats: Dict[str, List[Any]] = {k: sb.column(k).to_pandas().iloc[keep].tolist() for k in feat_cols if k in sb.schema.names}
							feat_runtime = inter_feature_cols
							# sequential
							if is_sequential:
								list_suf = str(streaming_spec.get("LIST_SUFFIX", "_list"))
								seq_k = int(streaming_spec.get("MAX_ITEM_LIST_LENGTH", 50))
								len_field = str(streaming_spec.get("ITEM_LIST_LENGTH_FIELD", "item_length"))
								seq_field = f"{item_field}{list_suf}"
								seqs: List[List[int]] = []
								lens: List[int] = []
								u_vals = u_raw.iloc[keep].tolist()
								i_vals = i_raw.iloc[keep].tolist()
								kept_pos: List[int] = []
								for pos_idx, (u_ext, i_ext) in enumerate(zip(u_vals, i_vals)):
									hist = history.get(u_ext, [])
									mapped = [int(item_map.get(str(int(x)), 0)) for x in hist]
									L = min(len(mapped), seq_k)
									if L == 0:
										history[u_ext] = hist + [i_ext]
										continue
									# Limit sequences per user
									# if user_seq_counter.get(u_ext, 0) >= max_seqs_per_user:
									# 	# advance history but do not emit
									# 	history[u_ext] = hist + [i_ext]
									# 	continue
									seq = ([0] * (seq_k - L)) + mapped[-seq_k:]
									seqs.append(seq); lens.append(L); kept_pos.append(pos_idx)
									user_seq_counter[u_ext] = user_seq_counter.get(u_ext, 0) + 1
									history[u_ext] = hist + [i_ext]
								if not kept_pos:
									continue
								buf_u = [buf_u[i] for i in kept_pos]
								buf_i = [buf_i[i] for i in kept_pos]
								y_vals = [y_vals[i] for i in kept_pos]
								buf_feats[seq_field] = seqs
								buf_feats[len_field] = lens
								feat_runtime = feat_runtime + [{"name": seq_field, "type": "int_seq"}, {"name": len_field, "type": "int"}]
							yield outer._make_interaction(user_field, item_field, label_field, buf_u, buf_i, y_vals, buf_feats, feat_runtime)
				finally:
					if con_main is not None:
						con_main.close()

		return _SASRecIterable()

	def _infer_predictions(self, users: Iterable[int], N: int) -> Dict[int, List[Tuple[int, float]]]:
		from typing import Dict, List, Tuple, Iterable
		import torch
		res: Dict[int, List[Tuple[int, float]]] = {}
		user_list = list(users)
		if not user_list:
			return res
		# Map external user ids to internal indices if needed
		try:
			user_field = self.recbole_config["USER_ID_FIELD"] if hasattr(self, "recbole_config") else "user_id"
			dataset = getattr(self, "_dataset", None)
			if dataset is not None and hasattr(dataset, "field2token_id"):
				u_map = dict(dataset.field2token_id.get(user_field, {}))
				idxs: List[int] = []
				for u in user_list:
					idx = u_map.get(str(int(u)))
					idxs.append(int(idx) if idx is not None else int(u))
			else:
				idxs = [int(u) for u in user_list]
		except Exception:
			idxs = [int(u) for u in user_list]
		# Predict in no-grad
		with torch.no_grad():
			for u_ext, u_idx in zip(user_list, idxs):
				try:
					scores = self.model.full_sort_predict(torch.tensor([u_idx], dtype=torch.long)).squeeze(0)
					if scores.ndim == 0:
						scores = scores.unsqueeze(0)
					topk = min(N, scores.numel())
					values, indices = torch.topk(scores, k=topk)
					res[int(u_ext)] = [(int(indices[i].item()), float(values[i].item())) for i in range(topk)]
				except Exception:
					# fallback popular if available
					pairs = []
					if hasattr(self, "_popular_items") and self._popular_items:
						k = min(N, len(self._popular_items))
						pairs = [(iid, float(k - i)) for i, iid in enumerate(self._popular_items[:k])]
					res[int(u_ext)] = pairs
		return res



from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple, Iterator
from pathlib import Path
import json
import sys
import logging

import polars as pl
import duckdb
from tqdm import tqdm
from pathlib import Path

from ml.core.interfaces import BaseModel


class RecBoleBaseAdapter(BaseModel):
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.params = {**params}
        self.model = None
        self.config: Dict[str, Any] | None = None
        self._popular_items: List[int] = []

    # Labels are expected to be precomputed in inter_features parquet as LABEL_FIELD

    # -------- dataset preparation --------
    def prepare_dataset(self, cfg: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Prepare minimal atomic files and return config + streaming spec."""
        self.config = self._to_plain_dict(cfg)['model']
        artifacts_dir = Path(self.config.get("artifacts_dir", "artifacts"))
        recbole_root = artifacts_dir / "recbole"
        recbole_root.mkdir(parents=True, exist_ok=True)

        dataset_name = self.config.get("dataset_name", "recbole_dataset")
        cache_key = self._compute_cache_key(self.config)
        dataset_dir = recbole_root / f"{dataset_name}_{cache_key}"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Input sources (required)
        items_base = Path(self.config["items_dir"])  # required
        users_base = Path(self.config["users_dir"])  # required
        inter_features_base = Path(self.config["inter_features_dir"])  # required
        user_field = self.config.get("USER_ID_FIELD", "user_id")
        item_field = self.config.get("ITEM_ID_FIELD", "item_id")
        label_field = self.config.get("LABEL_FIELD", "label")
        time_field = self.config.get("TIME_FIELD", "date")
        # optional feature columns (used only in streaming, not written to atomic files)
        item_feature_cols: List[Dict[str, str]] = self.config.get("item_feature_cols")
        user_feature_cols: List[Dict[str, str]] = self.config.get("user_feature_cols")
        inter_feature_cols: List[Dict[str, str]] = self.config.get("inter_feature_cols")

        # Target atomic files live under a subfolder named as dataset
        dataset_subdir = dataset_dir / dataset_name
        dataset_subdir.mkdir(parents=True, exist_ok=True)
        inter_path = dataset_subdir / f"{dataset_name}.inter"
        item_path = dataset_subdir / f"{dataset_name}.item"
        user_path = dataset_subdir / f"{dataset_name}.user"

        # Build .item and .user catalogs streaming unique ids
        if not item_path.exists():
            self._build_item_file(items_base, item_path, item_field)
        if not user_path.exists():
            self._build_user_file(users_base, user_path, user_field)

        # Build .inter from inter_features (expects precomputed label field)
        # Optional mode to only write header to avoid heavy IO when fully streaming batches
        inter_write_mode = self.config.get("inter_write_mode", "full")
        if (not inter_path.exists()) or (not self._inter_has_time_column(inter_path, time_field)):
            if inter_write_mode == "header_only":
                self._build_inter_file(
                    inter_features_base,
                    inter_path,
                    user_col=user_field,
                    item_col=item_field,
                    label_col=label_field,
                    time_col=time_field,
                    limit=1
                )
            else:
                self._build_inter_file(
                    inter_features_base,
                    inter_path,
                    user_col=user_field,
                    item_col=item_field,
                    label_col=label_field,
                    time_col=time_field,
                )

        # Compute popular items from inter_features for fallback (stream-friendly)
        # self._popular_items = self._compute_popular_items(inter_features_base, item_field, top_k=200)

        # Compose recbole config dict
        recbole_cfg: Dict[str, Any] = {
            "data_path": str(dataset_dir),
            "dataset": dataset_name,
            "USER_ID_FIELD": user_field,
            "ITEM_ID_FIELD": item_field,
            "LABEL_FIELD": label_field,
            "TIME_FIELD": time_field,
            "load_col": {
                "inter": [user_field, item_field, label_field, time_field],
                "item": [item_field],
                "user": [user_field],
            },
            # You can add additional Recbole options via external config
        }

        # Expose streaming spec (hybrid approach) via configurable globs
        streaming_spec = {
            "train_inter_shards": [str(p) for p in sorted((inter_features_base / "train" ).glob("*.parquet"))],
            "valid_inter_shards": [str(p) for p in sorted((inter_features_base / "valid" ).glob("*.parquet"))],
            # recursively discover shards (supports nested raw/ structure)
            "items_shards": [str(p) for p in sorted(items_base.rglob("*.parquet"))],
            "users_shards": [str(p) for p in sorted(users_base.rglob("*.parquet"))],
            "item_feature_cols": item_feature_cols,
            "user_feature_cols": user_feature_cols,
            "inter_feature_cols": inter_feature_cols,
            "batch_size": int(self.config.get("stream_batch_size", 1)),
            "num_workers": int(self.config.get("stream_num_workers", 0)),
            "is_sequential": bool(self.config.get("is_sequential", False)),
            "MAX_ITEM_LIST_LENGTH": int(self.config.get("MAX_ITEM_LIST_LENGTH", 50)),
            "LIST_SUFFIX": str(self.config.get("LIST_SUFFIX", "_list")),
            "ITEM_LIST_LENGTH_FIELD": str(self.config.get("ITEM_LIST_LENGTH_FIELD", "item_length")),
        }

        self.recbole_config = recbole_cfg

        train_data = {
            "recbole_config": recbole_cfg,
            "dataset_dir": str(dataset_dir),
            "dataset_name": dataset_name,
            "streaming_spec": streaming_spec,
        }
        valid_data: Dict[str, Any] = {"streaming_spec": streaming_spec}
        return train_data, valid_data

    # -------- training (to be optionally overridden by subclasses) --------
    def fit(self, train_data: Dict[str, Any]) -> "RecBoleBaseAdapter":
        """Default training flow using Recbole's Trainer.

        Subclasses can override to customize the model or training details purely via config.
        """
        try:
            from recbole.config import Config
            from recbole.data import create_dataset
            from recbole.trainer import Trainer
        except Exception as e:
            raise ImportError("recbole is required for training. Please install recbole.") from e

        assert self.recbole_config is not None

        # Merge external config (if provided via params) with prepared config
        # Merge external recbole overrides from params (flattened at top-level)
        ext_cfg = dict(self.params.get("recbole_config", {}))
        cfg_dict = {**self.recbole_config, **ext_cfg}
        config = Config(model=self.params.get("model", None), config_dict=cfg_dict)
        if "seed" in config:
            try:
                reproducibility = bool(config.get("reproducibility", False))
            except Exception:
                reproducibility = False
            from recbole.utils import init_seed as _init_seed
            _init_seed(int(config["seed"]), reproducibility)

        # Dataset and loaders
        print(f"[train.fit] Creating dataset")
        dataset = create_dataset(config)
        streaming_spec = train_data.get("streaming_spec", {}) if isinstance(train_data, dict) else {}
        print(f"[train.fit] Building streaming loaders")
        train_loader = self._build_streaming_loader(dataset, streaming_spec, mode="train")
        valid_loader = self._build_streaming_loader(dataset, streaming_spec, mode="valid")

        # Build model first, then Trainer requires a non-None model
        print(f"[train.fit] Building model")
        try:
            from recbole.utils import get_model as _get_model  # type: ignore
            model_cls = _get_model(str(config["model"]))
            model = model_cls(config, dataset)
        except Exception:
            print(f"[train.fit] Fallback to Trainer's internal builder if available")
            tmp_trainer = Trainer(config, model=None)
            model = tmp_trainer._build_model(config, dataset)
        print(f"[train.fit] Building trainer")
        trainer = Trainer(config, model=model)
        print(f"[train.fit] Fitting model")
        self.model = trainer.fit(train_loader, valid_loader)
        return self

    # -------- inference --------
    def infer(self, users: Iterable[int], N: int = 100) -> Dict[int, List[Tuple[int, float]]]:
        if self.model is None:
            # Fallback: return popular items
            k = min(N, len(self._popular_items))
            pairs = [(iid, float(k - i)) for i, iid in enumerate(self._popular_items[:k])]
            return {int(u): pairs for u in users}

        try:
            import torch
        except Exception as e:
            raise ImportError("torch is required for inference.") from e

        # Recbole models typically implement full_sort_predict
        res: Dict[int, List[Tuple[int, float]]] = {}
        user_list = list(users)
        if not user_list:
            return res

        # Predict one-by-one to avoid coupling with recbole internals
        with torch.no_grad():
            for u in user_list:
                # Most recbole models expect internal user id (index). If external ids were tokens,
                # additional mapping may be required depending on model/dataset integration.
                try:
                    scores = self.model.full_sort_predict(torch.tensor([int(u)], dtype=torch.long)).squeeze(0)
                    if scores.ndim == 0:
                        scores = scores.unsqueeze(0)
                    topk = min(N, scores.numel())
                    values, indices = torch.topk(scores, k=topk)
                    res[int(u)] = [(int(indices[i].item()), float(values[i].item())) for i in range(topk)]
                except Exception:
                    # Fallback popular
                    k = min(N, len(self._popular_items))
                    pairs = [(iid, float(k - i)) for i, iid in enumerate(self._popular_items[:k])]
                    res[int(u)] = pairs
        return res

    # -------- helpers --------
    def _to_plain_dict(self, cfg: Any) -> Dict[str, Any]:
        try:
            from omegaconf import OmegaConf
            return dict(OmegaConf.to_container(cfg, resolve=True))
        except Exception:
            return dict(cfg) if isinstance(cfg, dict) else {}

    def _compute_cache_key(self, cfg_d: Dict[str, Any]) -> str:
        import hashlib
        payload = {
            "items_dir": cfg_d.get("items_dir", "data/raw_data/ml_ozon_recsys_items_data"),
            "orders_dir": cfg_d.get("orders_dir", "data/raw_data/ml_ozon_recsys_orders_data"),
            "label_type": self.config.get("label_type", "combined"),
        }
        key_str = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(key_str.encode("utf-8")).hexdigest()[:16]

    def _build_item_file(self, items_base: Path, item_path: Path, item_col: str) -> None:
        self._build_token_file(items_base, item_path, item_col)

    def _build_user_file(self, base_dir: Path, user_path: Path, user_col: str) -> None:
        self._build_token_file(base_dir, user_path, user_col)

    def _build_token_file(self, base_dir: Path, out_path: Path, col: str) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"{col}:token\n")
        seen: set[int] = set()
        # Search recursively to include nested directories (e.g., raw/)
        for shard in tqdm(sorted(base_dir.rglob("*.parquet")), desc=f"Building {out_path} file"):
            lf = pl.scan_parquet(str(shard)).select([col]).unique()
            df = lf.collect(streaming=True)
            if df.height == 0:
                continue
            ids = [int(x) for x in df[col].to_list() if x is not None]
            new_ids = [i for i in ids if i not in seen]
            if not new_ids:
                continue
            with open(out_path, "a", encoding="utf-8") as f:
                for vid in new_ids:
                    f.write(f"{vid}\n")
            seen.update(new_ids)

    def _build_inter_file(
        self,
        inter_base: Path,
        inter_path: Path,
        *,
        user_col: str,
        item_col: str,
        label_col: str,
        time_col: str,
        limit: int = -1,
    ) -> None:
        inter_path.parent.mkdir(parents=True, exist_ok=True)
        # Header for Recbole .inter
        with open(inter_path, "w", encoding="utf-8") as f_header:
            f_header.write(f"{user_col}:token\t{item_col}:token\t{label_col}:float\t{time_col}:float\n")

        # Build rows shard-by-shard using vectorized Polars ops and append TSV without header
        # Search recursively to include train/ and valid/ subfolders
        with open(inter_path, "a", encoding="utf-8") as out_f:
            shards = sorted(inter_base.rglob("*.parquet"))
            if limit and limit > 0:
                shards = shards[:limit]
            for shard in tqdm(shards, desc=f"Building {inter_path} file"):
                lf = (
                    pl.scan_parquet(str(shard))
                    .select([user_col, item_col, label_col, time_col])
                    .with_columns([
                        pl.col(user_col).cast(pl.Int64, strict=False).alias(user_col),
                        pl.col(item_col).cast(pl.Int64, strict=False).alias(item_col),
                        pl.col(label_col).cast(pl.Float64, strict=False).fill_null(0.0).alias(label_col),
                        pl.col(time_col).cast(pl.Float64, strict=False).fill_null(0.0).alias(time_col),
                    ])
                    .filter(
                        pl.col(user_col).is_not_null()
                        & pl.col(item_col).is_not_null()
                        & (pl.col(label_col) > 0)
                    )
                    .select([user_col, item_col, label_col, time_col])
                )

                df = lf.collect(streaming=True)
                if df.height == 0:
                    continue
                # Append without header in TSV format
                df.write_csv(out_f, separator="\t", include_header=False)

    def _compute_popular_items(self, inter_base: Path, item_col: str, top_k: int = 200) -> List[int]:
        # Build per-shard lazy aggregations, then aggregate globally in Polars
        shard_lfs: List[pl.LazyFrame] = []
        for shard in tqdm(sorted(inter_base.rglob("*.parquet")), desc="Computing popular items"):
            lf = (
                pl.scan_parquet(str(shard))
                .select([item_col])
                .with_columns([
                    pl.col(item_col).cast(pl.Int64, strict=False).alias(item_col),
                ])
                .drop_nulls([item_col])
                .group_by(item_col)
                .count()
            )
            shard_lfs.append(lf)

        if not shard_lfs:
            return []

        all_counts_lf = pl.concat(shard_lfs, how="vertical_relaxed")
        global_counts_lf = (
            all_counts_lf
            .group_by(item_col)
            .agg(pl.col("count").sum().alias("count"))
            .sort("count", descending=True)
            .head(top_k)
            .select([item_col])
        )

        df_top = global_counts_lf.collect(streaming=True)
        if df_top.height == 0:
            return []
        return [int(x) for x in df_top[item_col].to_list() if x is not None]

    def _inter_has_time_column(self, inter_path: Path, time_col: str) -> bool:
        try:
            if not inter_path.exists():
                return False
            with open(inter_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if not lines:
                return False
            header = lines[0].strip().split("\t")
            has_time = any(h.split(":")[0] == time_col for h in header)
            has_rows = len(lines) > 1
            return has_time and has_rows
        except Exception:
            return False

    def _precompute_final_features(
        self,
        *,
        inter_shards: List[str],
        item_shards: List[str],
        item_field: str,
        item_feature_cols: List[Dict[str, str]]
    ) -> List[str]:
        """Register DuckDB VIEW over item feature shards. No precompute or COPY is performed.

        Returns the original list of interaction shards unchanged.
        """
        item_cols = [c.get("name") for c in item_feature_cols if c.get("name")]
        items_files_sql = "[" + ",".join([f"'{p}'" for p in item_shards]) + "]" if item_shards else "[]"
        con = duckdb.connect()
        try:
            sel_items_cols = ", ".join([item_field] + item_cols) if item_cols else item_field
            con.execute(
                f"CREATE OR REPLACE VIEW items_all AS SELECT {sel_items_cols} FROM read_parquet({items_files_sql})"
            )
        finally:
            con.close()
        return inter_shards

    # -------- streaming DataLoader (hybrid) --------
    def _build_streaming_loader(self, dataset: Any, streaming_spec: Dict[str, Any], mode: str):
        try:
            import torch
            from torch.utils.data import IterableDataset, DataLoader
            from recbole.data.interaction import Interaction
        except Exception as e:
            raise ImportError("torch and recbole are required for streaming dataloader.") from e

        user_field = dataset.config["USER_ID_FIELD"]
        item_field = dataset.config["ITEM_ID_FIELD"]
        label_field = dataset.config["LABEL_FIELD"]
        time_field = dataset.config["TIME_FIELD"]

        # token maps from dataset (string token -> internal id)
        # Prefer field2token_id; if unavailable or array-like, build mapping safely.
        def _ensure_token_to_id_map(ds: Any, field_name: str) -> Dict[str, int]:
            # Try the correct mapping first
            token_to_id = getattr(ds, "field2token_id", {}).get(field_name, None)
            if hasattr(token_to_id, "get"):
                return dict(token_to_id)  # type: ignore[arg-type]
            # Fallback: invert id->token structure
            id_to_token = getattr(ds, "field2id_token", {}).get(field_name, None)
            mapping: Dict[str, int] = {}
            if isinstance(id_to_token, dict):
                for idx, tok in id_to_token.items():
                    try:
                        mapping[str(tok)] = int(idx)
                    except Exception:
                        continue
                return mapping
            # If it's a sequence/ndarray, enumerate
            try:
                for idx, tok in enumerate(id_to_token):  # type: ignore[assignment]
                    mapping[str(tok)] = int(idx)
            except Exception:
                mapping = {}
            return mapping

        user_map: Dict[str, int] = _ensure_token_to_id_map(dataset, user_field)
        item_map: Dict[str, int] = _ensure_token_to_id_map(dataset, item_field)

        inter_shards: List[str] = list(
            streaming_spec.get("train_inter_shards" if mode == "train" else "valid_inter_shards") or []
        )
        item_shards: List[str] = list(streaming_spec.get("items_shards") or [])
        user_shards: List[str] = list(streaming_spec.get("users_shards") or [])
        item_feature_cols: List[Dict[str, str]] = list(streaming_spec.get("item_feature_cols") or [])
        user_feature_cols: List[Dict[str, str]] = list(streaming_spec.get("user_feature_cols") or [])
        inter_feature_cols: List[Dict[str, str]] = list(streaming_spec.get("inter_feature_cols") or [])
        batch_size: int = int(streaming_spec.get("batch_size", 4096))
        num_workers: int = int(streaming_spec.get("num_workers", 0))
        # Optional: register DuckDB VIEW once (no precompute)
        is_sequential = bool(streaming_spec.get("is_sequential", False))
        if item_shards:
            self._precompute_final_features(
                inter_shards=inter_shards,
                item_shards=item_shards,
                item_field=item_field,
                item_feature_cols=item_feature_cols
            )
        print(f"[train.dataloader] Using DuckDB runtime join for item features")
        self_outer = self
        # ----- DuckDB one-time items registration (outside of __iter__) -----
        duckdb_db_path: str | None = None
        item_cols: List[str] = [c.get("name") for c in item_feature_cols if c.get("name")]
        if item_cols:
            try:
                import duckdb  # type: ignore
            except Exception as e:
                raise ImportError("duckdb is required for runtime item feature join") from e
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

        class _ParquetIterable(IterableDataset):
            def __iter__(self) -> Iterator[Interaction]:
                # Local imports
                import duckdb  # type: ignore
                import pyarrow as pa  # type: ignore
                feature_cols_all = [c.get("name") for c in (inter_feature_cols + item_feature_cols + user_feature_cols) if c.get("name")]
                # Open read-only connection to pre-registered DB if item features are present
                con = duckdb.connect(duckdb_db_path, read_only=True) if (duckdb_db_path is not None) else None
                try:
                    for inter_path in tqdm(inter_shards, desc="Processing inter shards"):
                        # Build join SQL per interactions shard
                        if item_cols and con is not None:
                            sel_item_feats = ", ".join([f"it.{c}" for c in item_cols])
                            select_list = f"i.*, {sel_item_feats}"
                            sql = (
                                f"SELECT {select_list} FROM read_parquet('{inter_path}') i "
                                f"LEFT JOIN items_all it USING ({item_field})"
                            )
                            tb = con.execute(sql).fetch_arrow_table()
                        else:
                            import pyarrow.parquet as pq  # type: ignore
                            tb = pq.read_table(inter_path)
                        if tb.num_rows == 0:
                            continue
                        # iterate in slices to control memory
                        for start in range(0, tb.num_rows, batch_size):
                            end = min(start + batch_size, tb.num_rows)
                            sb: pa.Table = tb.slice(start, end - start)
                            if sb.num_rows == 0:
                                continue
                            # map tokens to internal ids per row; drop unmapped
                            u_col = sb.column(user_field).to_pandas()
                            i_col = sb.column(item_field).to_pandas()
                            keep_idx: List[int] = []
                            buf_u: List[int] = []
                            buf_i: List[int] = []
                            for idx, (u, i) in enumerate(zip(u_col, i_col)):
                                um = user_map.get(str(int(u))) if u is not None else None
                                im = item_map.get(str(int(i))) if i is not None else None
                                if (um is not None) and (im is not None):
                                    keep_idx.append(idx)
                                    buf_u.append(um)
                                    buf_i.append(im)
                            if not keep_idx:
                                continue
                            # labels
                            if label_field in sb.schema.names:
                                y_vals = sb.column(label_field).to_pandas().iloc[keep_idx].astype(float).tolist()
                            else:
                                y_vals = [0.0] * len(keep_idx)
                            # gather features by kept indices
                            buf_feats: Dict[str, List[Any]] = {}
                            for name in feature_cols_all:
                                if name in sb.schema.names:
                                    buf_feats[name] = sb.column(name).to_pandas().iloc[keep_idx].tolist()
                            # sequential features runtime
                            feature_cols_runtime = inter_feature_cols + item_feature_cols + user_feature_cols
                            if is_sequential:
                                list_suffix = str(streaming_spec.get("LIST_SUFFIX", "_list"))
                                seq_max_len = int(streaming_spec.get("MAX_ITEM_LIST_LENGTH", 50))
                                seq_len_field = str(streaming_spec.get("ITEM_LIST_LENGTH_FIELD", "item_length"))
                                seq_field_name = f"{item_field}{list_suffix}"
                                lens_kept: List[int] = []
                                if seq_field_name in sb.schema.names:
                                    raw_seqs: List[Any] = sb.column(seq_field_name).to_pylist()
                                    seqs_kept: List[List[int]] = []
                                    for idx in keep_idx:
                                        seq_raw = raw_seqs[idx] or []
                                        mapped = [int(item_map.get(str(int(x)), 0)) for x in seq_raw if x is not None]
                                        length = min(len(mapped), seq_max_len)
                                        recent = mapped[-seq_max_len:]
                                        if len(recent) < seq_max_len:
                                            recent = [0] * (seq_max_len - len(recent)) + recent
                                        seqs_kept.append(recent)
                                        lens_kept.append(length)
                                    buf_feats[seq_field_name] = seqs_kept
                                    buf_feats[seq_len_field] = lens_kept
                                    feature_cols_runtime = feature_cols_runtime + [
                                        {"name": seq_field_name, "type": "int_seq"},
                                        {"name": seq_len_field, "type": "int"},
                                    ]
                                else:
                                    # build sequences on the fly within this slice
                                    u_vals = sb.column(user_field).to_pandas().iloc[keep_idx].astype(int).tolist()
                                    i_vals_ext = sb.column(item_field).to_pandas().iloc[keep_idx].astype(int).tolist()
                                    history: Dict[int, List[int]] = {}
                                    seqs_kept: List[List[int]] = []
                                    for u_ext, i_ext in zip(u_vals, i_vals_ext):
                                        hist = history.get(u_ext, [])
                                        mapped_hist = [int(item_map.get(str(int(x)), 0)) for x in hist]
                                        length = min(len(mapped_hist), seq_max_len)
                                        recent = mapped_hist[-seq_max_len:]
                                        if len(recent) < seq_max_len:
                                            recent = [0] * (seq_max_len - len(recent)) + recent
                                        seqs_kept.append(recent)
                                        lens_kept.append(length)
                                        history[u_ext] = hist + [i_ext]
                                    buf_feats[seq_field_name] = seqs_kept
                                    buf_feats[seq_len_field] = lens_kept
                                    feature_cols_runtime = feature_cols_runtime + [
                                        {"name": seq_field_name, "type": "int_seq"},
                                        {"name": seq_len_field, "type": "int"},
                                    ]
                                # drop rows with zero history length to avoid item_seq_len-1 = -1
                                sel_idx = [i for i, L in enumerate(lens_kept) if L > 0]
                                if not sel_idx:
                                    continue
                                # apply selection to core buffers
                                buf_u = [buf_u[i] for i in sel_idx]
                                buf_i = [buf_i[i] for i in sel_idx]
                                y_vals = [y_vals[i] for i in sel_idx]
                                # apply selection to all feature lists
                                for k, v in list(buf_feats.items()):
                                    if isinstance(v, list) and len(v) == len(lens_kept):
                                        buf_feats[k] = [v[i] for i in sel_idx]
                            yield self_outer._make_interaction(user_field, item_field, label_field, buf_u, buf_i, y_vals, buf_feats, feature_cols_runtime)
                finally:
                    if con is not None:
                        con.close()

        iterable = _ParquetIterable()
        print(f"[train.fit] Building dataloader")
        return DataLoader(iterable, batch_size=None, num_workers=num_workers)


    def _feature_to_python(self, val: Any, ftype: str) -> Any:
        if val is None:
            return None
        if ftype in ("float", "float32", "float64"):
            try:
                return float(val)
            except Exception:
                return None
        if ftype in ("token", "str", "string"):
            return str(val)
        if ftype in ("int", "i64"):
            try:
                return int(val)
            except Exception:
                return None
        if ftype in ("float_seq", "float32_seq", "float_list", "list_float", "list[f32]"):
            try:
                return list(val) if isinstance(val, list) else [float(x) for x in val]
            except Exception:
                return None
        return val

    def _make_interaction(
        self,
        user_field: str,
        item_field: str,
        label_field: str,
        buf_u: List[int],
        buf_i: List[int],
        buf_y: List[float],
        buf_feats: Dict[str, List[Any]],
        feature_cols: List[Dict[str, str]],
    ):
        import torch
        from recbole.data.interaction import Interaction
        batch: Dict[str, Any] = {
            user_field: torch.tensor(buf_u, dtype=torch.long),
            item_field: torch.tensor(buf_i, dtype=torch.long),
            label_field: torch.tensor(buf_y, dtype=torch.float32),
        }
        for c in feature_cols:
            name = c.get("name")
            ftype = c.get("type", "token")
            vals = buf_feats.get(name, [])
            if ftype in ("float_seq", "float32_seq", "float_list", "list_float", "list[f32]"):
                # Convert list-of-lists to dense tensor; ragged sequences require padding or masking (not handled here)
                # Simple approach: stack only if all equal lengths; else drop to object tensors (not supported), so skip
                if vals and all(isinstance(v, list) for v in vals):
                    max_len = max((len(v) for v in vals if isinstance(v, list)), default=0)
                    if max_len > 0 and all(len(v) == max_len for v in vals):
                        batch[name] = torch.tensor(vals, dtype=torch.float32)
                continue
            # Handle integer/token sequences
            if ftype in ("int_seq", "token_seq", "list_int", "int_list"):
                if vals and all(isinstance(v, list) for v in vals):
                    max_len = max((len(v) for v in vals if isinstance(v, list)), default=0)
                    if max_len > 0 and all(len(v) == max_len for v in vals):
                        batch[name] = torch.tensor(vals, dtype=torch.long)
                continue
            if ftype in ("float", "float32", "float64"):
                batch[name] = torch.tensor([float(x) if x is not None else 0.0 for x in vals], dtype=torch.float32)
            elif ftype in ("int", "i64"):
                batch[name] = torch.tensor([int(x) if x is not None else 0 for x in vals], dtype=torch.long)
            else:
                # token / string: keep as list of strings; if needed, a tokenizer can be applied in model-specific adapter
                # batch[name] = vals
                continue
        print(batch.keys())
        print(f"Tensor size: {batch['item_id_list'].size()}")
        return Interaction(batch)

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Tuple
from pathlib import Path
import json
import hashlib
import time
import logging
import numpy as np

import polars as pl
import torch
from torch.utils.data import IterableDataset, DataLoader
import duckdb as dd

# NumPy 2.0 compatibility for third-party checks inside RecBole
if not hasattr(np, "float"):  # numpy>=2.0 removed aliases
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]
if not hasattr(np, "object"):
    np.object = object  # type: ignore[attr-defined]

from recbole.data.interaction import Interaction

from ml.core.interfaces import BaseModel

# Настройка логгера
logger = logging.getLogger(__name__)


class _ParquetInteractionsDataset(IterableDataset):
    """Stream interactions from parquet into python-native samples.

    - External→internal id mapping via Polars joins
    - Optional label/time pre-cast in Polars
    - Only inter features (no item/user augmentation)
    - Values are native int/float or lists; tensors are formed in collate_fn
    """

    def __init__(
        self,
        files: List[Path],
        rb_fields: Dict[str, Any],
        user_id_to_idx: Dict[int, int],
        item_id_to_idx: Dict[int, int],
        items_dir: str | None,
        users_dir: str | None,
        float_seq_dtype: str = "float16",
    ) -> None:
        super().__init__()
        self.files = files
        self.rb_fields = rb_fields
        self.user_id_to_idx = user_id_to_idx
        self.item_id_to_idx = item_id_to_idx
        self.items_dir = items_dir
        self.users_dir = users_dir
        self.float_seq_dtype = float_seq_dtype
        # Names of additional feature columns produced by model-specific enrichment
        self._extra_feature_names: List[str] = []

        # Pre-build small mapping frames to vectorize joins in Polars
        if self.user_id_to_idx:
            self._user_map_lf = pl.DataFrame({
                "__u_ext__": list(self.user_id_to_idx.keys()),
                "__ui__": list(self.user_id_to_idx.values()),
            }).lazy()
        else:
            self._user_map_lf = None
        if self.item_id_to_idx:
            self._item_map_lf = pl.DataFrame({
                "__i_ext__": list(self.item_id_to_idx.keys()),
                "__ii__": list(self.item_id_to_idx.values()),
            }).lazy()
        else:
            self._item_map_lf = None
        self._init_duckdb_if_needed()

    def _init_duckdb_if_needed(self) -> None:
        # Prepare DuckDB views for fast joins (only if feature columns are configured)
        self._duck: dd.DuckDBPyConnection | None = None
        item_cols = [c.get("name") for c in (self.rb_fields.get("item_cols") or []) if c.get("name")]
        user_cols = [c.get("name") for c in (self.rb_fields.get("user_cols") or []) if c.get("name")]
        if (item_cols and self.items_dir) or (user_cols and self.users_dir):
            self._duck = dd.connect()
            self._duck.execute("PRAGMA enable_progress_bar=false;")
            if item_cols and self.items_dir:
                item_id_col = self.rb_fields.get("item_f")
                select_list = ", ".join([item_id_col] + item_cols)
                self._duck.execute(
                    f"CREATE OR REPLACE VIEW items AS SELECT {select_list} FROM read_parquet('{Path(self.items_dir) / 'raw' / '*.parquet'}');"
                )
            if user_cols and self.users_dir:
                user_id_col = self.rb_fields.get("user_f")
                select_list = ", ".join([user_id_col] + user_cols)
                self._duck.execute(
                    f"CREATE OR REPLACE VIEW users AS SELECT {select_list} FROM read_parquet('{Path(self.users_dir) / 'raw' / '*.parquet'}');"
                )

    def _duckdb_select_enriched(self, fp: Path, select_cols: List[str], user_f: str, item_f: str) -> pl.DataFrame:
        assert self._duck is not None
        item_cols_cfg = [c.get("name") for c in (self.rb_fields.get("item_cols") or [])]
        user_cols_cfg = [c.get("name") for c in (self.rb_fields.get("user_cols") or [])]
        proj_inter = ", ".join([f"i.{c}" for c in select_cols]) if select_cols else "i.*"
        proj_items = ", ".join([f"it.{c}" for c in item_cols_cfg]) if item_cols_cfg else ""
        proj_users = ", ".join([f"u.{c}" for c in user_cols_cfg]) if user_cols_cfg else ""
        proj_all = ", ".join([p for p in [proj_inter, proj_items, proj_users] if p])
        sql_parts: List[str] = [f"SELECT {proj_all} FROM read_parquet('{str(fp)}') i"]
        if item_cols_cfg and self.items_dir:
            sql_parts.append(f"LEFT JOIN items it ON i.{item_f} = it.{item_f}")
        if user_cols_cfg and self.users_dir:
            sql_parts.append(f"LEFT JOIN users u ON i.{user_f} = u.{user_f}")
        return self._duck.execute("\n".join(sql_parts)).pl()

    def _enrich_df(self, df: pl.DataFrame) -> pl.DataFrame:
        # Overridable by model-specific datasets to add new feature columns
        self._extra_feature_names = []
        return df

    def _convert_value_by_type(self, value: Any, field_type: str) -> Any:
        """Конвертирует значение в соответствии с типом из конфига.
        
        Поддерживаемые типы:
        - float: одиночное float значение
        - float_seq: последовательность float
        - token: категориальная переменная (int)
        - token_seq: последовательность токенов (int)
        """
        if value is None:
            # Возвращаем дефолтные значения для None
            if field_type == "float":
                return 0.0
            elif field_type == "float_seq":
                return []
            elif field_type == "token":
                return 0
            elif field_type == "token_seq":
                return []
            return value
        
        # Обработка последовательностей
        if field_type in ("float_seq", "token_seq"):
            if isinstance(value, (list, tuple, np.ndarray)):
                if field_type == "float_seq":
                    return [float(v) for v in value]
                else:  # token_seq
                    return [int(v) if v is not None else 0 for v in value]
            # Если не последовательность, оборачиваем в список
            if field_type == "float_seq":
                return [float(value)]
            else:  # token_seq
                return [int(value)]
        
        # Обработка скалярных значений
        if field_type == "float":
            return float(value) if not isinstance(value, (list, tuple, np.ndarray)) else float(value[0])
        elif field_type == "token":
            return int(value) if not isinstance(value, (list, tuple, np.ndarray)) else int(value[0])
        
        # Fallback: пытаемся определить тип автоматически
        if isinstance(value, (list, tuple, np.ndarray)):
            return list(value)
        elif isinstance(value, (float, np.floating)):
            return float(value)
        elif isinstance(value, (int, np.integer)):
            return int(value)
        
        return value

    def __iter__(self):
        user_f = self.rb_fields.get("user_f")
        item_f = self.rb_fields.get("item_f")
        label_f = self.rb_fields.get("label_f")
        time_f = self.rb_fields.get("time_f")
        base_cols: List[Dict[str, Any]] = self.rb_fields.get("base_cols", []) or []
        inter_cols: List[Dict[str, Any]] = self.rb_fields.get("inter_cols", []) or []
        item_cols: List[Dict[str, Any]] = self.rb_fields.get("item_cols", []) or []
        user_cols: List[Dict[str, Any]] = self.rb_fields.get("user_cols", []) or []
        auxiliary_cols: List[Dict[str, Any]] = self.rb_fields.get("auxiliary_feature_cols", []) or []

        select_cols = [c["name"] for c in (base_cols + inter_cols)]

        # Построение единой схемы типов для всех колонок
        column_types: Dict[str, str] = {}
        # Обязательные поля
        column_types[user_f] = "token"
        column_types[item_f] = "token"
        if label_f:
            column_types[label_f] = "float"
        if time_f:
            column_types[time_f] = "token"
        
        # Добавляем типы из конфига для всех feature колонок
        for spec in inter_cols + item_cols + user_cols + auxiliary_cols:
            name = spec.get("name")
            ftype = spec.get("type")
            if name and ftype:
                column_types[name] = ftype
        for fp in self.files:
            # read interactions and enrich with item/user features via DuckDB if available
            if hasattr(self, "_duck") and self._duck is not None:
                df = self._duckdb_select_enriched(fp, select_cols, user_f, item_f)
            else:
                df = pl.scan_parquet(str(fp)).select(select_cols).collect(streaming=True)
            # Map external ids to internal indices and pre-cast label/time
            lf = df.lazy()
            if self._user_map_lf is not None:
                lf = lf.join(self._user_map_lf, left_on=user_f, right_on="__u_ext__", how="left")
            if self._item_map_lf is not None:
                lf = lf.join(self._item_map_lf, left_on=item_f, right_on="__i_ext__", how="left")
            if label_f:
                lf = lf.with_columns(pl.col(label_f).cast(pl.Float32, strict=False))
            if time_f:
                lf = lf.with_columns(
                    pl.when(pl.col(time_f).is_not_null())
                      .then(pl.col(time_f).cast(pl.Date, strict=False).cast(pl.Int32, strict=False))
                      .otherwise(None)
                      .alias(time_f)
                )
            lf = lf.filter((pl.col("__ui__").is_not_null()) & (pl.col("__ii__").is_not_null()))
            df = lf.collect(streaming=True)
            if df.height == 0:
                continue
            # Allow subclass to add new features (e.g., sequences) efficiently
            df = self._enrich_df(df)
            
            # Унифицированная конвертация всех колонок в numpy массивы
            column_arrays: Dict[str, np.ndarray] = {}
            column_arrays["__ui__"] = df["__ui__"].to_numpy()
            column_arrays["__ii__"] = df["__ii__"].to_numpy()

            # Конвертируем все колонки в numpy за один проход
            for spec in base_cols + inter_cols + item_cols + user_cols + auxiliary_cols:
                col_name = spec["name"]
                column_arrays[col_name] = df[col_name].to_numpy()

            n = df.height
            for idx in range(n):
                sample: Dict[str, Any] = {
                    user_f: int(column_arrays["__ui__"][idx]),
                    item_f: int(column_arrays["__ii__"][idx]),
                }
                for col_name, arr in column_arrays.items():
                    if col_name in ("__ui__", "__ii__"):
                        continue  # Уже обработаны выше
                    # Используем типизированную конвертацию
                    sample[col_name] = self._convert_value_by_type(arr[idx], column_types[col_name])

                yield sample

class RecBoleBaseAdapter(BaseModel):
    """Adapter for RecBole over parquet-based streaming dataset.

    - Scans train parquet to build user/item id maps (external→internal)
    - Derives validation users and ground truth from valid parquet
    - Creates streaming DataLoaders that join ids and yield python-native samples
    - Collates batches into RecBole `Interaction`
    - Provides `fit`/`infer` wrappers around RecBole
    """

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.params = {**params}
        self.user_id_to_idx: Dict[int, int] = {}
        self.item_id_to_idx: Dict[int, int] = {}
        self.idx_to_user_id: List[int] = []
        self.idx_to_item_id: List[int] = []
        self._recbole_model: Any = None
        self._device: str = "cuda" if torch.cuda.is_available() and bool(self.params.get("use_gpu", True)) else "cpu"
        self._rb_fields: Dict[str, Any] = {}

    # --------- core API ---------
    def prepare_dataset(self, cfg: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        start_time = time.time()
        logger.info("🚀 Начинаем подготовку датасета...")

        cfg_d = self._cfg_to_dict(cfg)
        # extract recbole field names and paths from config (conf/model/recbole.yaml)
        self._extract_recbole_fields(cfg_d)
        artifacts_dir = Path(cfg_d.get("artifacts_dir", "artifacts"))
        cache_root = artifacts_dir / "cache" / "recbole_prepare"
        cache_root.mkdir(parents=True, exist_ok=True)

        cache_key = self._make_cache_key(cfg_d)
        cache_dir = cache_root / cache_key
        cache_dir.mkdir(parents=True, exist_ok=True)

        # paths
        user_map_path = cache_dir / "user_map.json"
        item_map_path = cache_dir / "item_map.json"
        meta_path = cache_dir / "meta.json"

        # Build id maps (cached if present)
        if user_map_path.exists() and item_map_path.exists():
            logger.info("📦 Найдены кэшированные map-файлы, загружаем...")
            self._load_id_maps(user_map_path, item_map_path)
        else:
            logger.info("🗺️ Строим ID маппинги из train parquet...")
            map_start = time.time()
            self._build_id_maps_from_train_split(cfg_d, user_map_path, item_map_path)
            logger.info(f"🗺️ Построение ID маппингов: {time.time() - map_start:.2f}s")
            logger.info(f"   - Пользователей: {len(self.user_id_to_idx)}")
            logger.info(f"   - Товаров: {len(self.item_id_to_idx)}")

        # Build validation users (cached in meta)
        if meta_path.exists():
            valid_users = self._load_valid_users(meta_path)
        else:
            users_start = time.time()
            valid_users = self.build_validation_users(cfg_d, cache_dir)
            logger.info(f"👥 Построение списка валидационных пользователей: {time.time() - users_start:.2f}s")
            with open(meta_path, "w") as f:
                json.dump({
                    "n_users": len(self.user_id_to_idx),
                    "n_items": len(self.item_id_to_idx),
                    "valid_users": valid_users,
                    "cache_key": cache_key,
                }, f)

        # STREAMING loaders directly from Parquet
        loader_start = time.time()
        paths = self._recbole_paths(cfg_d)
        inter_train_dir = Path(paths["inter_features_dir"]) / "train"
        inter_valid_dir = Path(paths["inter_features_dir"]) / "valid"

        train_loader = self._make_streaming_loader(
            files=sorted(inter_train_dir.glob("*.parquet")),
        )
        valid_loader = self._make_streaming_loader(
            files=sorted(inter_valid_dir.glob("*.parquet")),
        ) if inter_valid_dir.exists() and any(inter_valid_dir.glob("*.parquet")) else None
        logger.info(f"🔧 Создание Streaming DataLoader'ов: {time.time() - loader_start:.2f}s")

        gt_start = time.time()
        ground_truth = self._load_valid_ground_truth(cfg_d)
        logger.info(f"🎯 Загрузка ground truth: {time.time() - gt_start:.2f}s")
        logger.info(f"   - Ground truth пар: {len(ground_truth)}")

        train_data = {
            "train_loader": train_loader,
            "user_map": self.user_id_to_idx,
            "item_map": self.item_id_to_idx,
        }
        valid_data = {
            "users": valid_users,
            "valid_loader": valid_loader,
            "ground_truth": ground_truth,
        }
        
        total_time = time.time() - start_time
        logger.info(f"🎉 Подготовка датасета завершена! Общее время: {total_time:.2f}s")
        logger.info(f"📊 Итоговая статистика:")
        logger.info(f"   - Пользователей: {len(self.user_id_to_idx)}")
        logger.info(f"   - Товаров: {len(self.item_id_to_idx)}")
        logger.info(f"   - Валидационных пользователей: {len(valid_users)}")
        logger.info(f"   - Ground truth пар: {len(ground_truth)}")
        logger.info(f"   - Кэш директория: {cache_dir}")
        
        return train_data, valid_data

    def fit(self, train_data: Dict[str, Any]) -> "RecBoleBaseAdapter":
        # Lazy import to keep base light if group not installed
        from recbole.config import Config
        from recbole.data import create_dataset
        from recbole.trainer import Trainer
        from recbole.utils import init_seed
        from recbole.utils import get_model

        # Build a minimal RecBole Config from params; rely on external prepared data via our loaders
        rb_params = {**self.params}
        model_name: str = rb_params.pop("model", "DSSM")
        rb_params.setdefault("device", self._device)

        config = Config(model=model_name, dataset="custom", config_dict=rb_params)
        init_seed(config['seed']) if 'seed' in config else None

        # Build a minimal dataset for model init (no heavy IO); RecBole requires a dataset object
        dataset = create_dataset(config)
        model_class = get_model(model_name)
        model = model_class(config, dataset).to(self._device)

        # Train using our prebuilt DataLoader directly
        trainer = Trainer(config, model)
        trainer.fit(train_data=train_data.get("train_loader"), valid_data=None, saved=False, show_progress=True)

        self._recbole_model = model
        return self

    def infer(self, users: Iterable[int], N: int = 100) -> Dict[int, List[Tuple[int, float]]]:
        # Basic user-based scoring via model's full_sort_predict where available
        assert self._recbole_model is not None
        self._recbole_model.eval()
        device = self._device
        res: Dict[int, List[Tuple[int, float]]] = {}
        # default popular fallback from prepared train shards
        item_pop = self._estimate_item_popularity_from_train()
        default_items = [k for (k, _) in sorted(item_pop.items(), key=lambda x: -x[1])][:N]
        default_pairs = [(int(i), float(item_pop.get(i, 0.0))) for i in default_items]

        # Map users to internal if available
        batch: List[int] = []
        externals: List[int] = []
        def flush_batch():
            if not batch:
                return
            with torch.no_grad():
                user_ids = torch.tensor(batch, dtype=torch.long, device=device)
                scores = self._recbole_model.full_sort_predict(user_ids).detach().cpu()
            for i, u_ext in enumerate(externals):
                arr: torch.Tensor = scores[i]
                if arr.numel() == 0:
                    res[u_ext] = default_pairs
                    continue
                topk = min(N, arr.numel())
                vals, idx = torch.topk(arr, k=topk, largest=True, sorted=True)
                item_ids_ext = [self.idx_to_item_id[j] if j < len(self.idx_to_item_id) else j for j in idx.tolist()]
                res[u_ext] = [(int(item_ids_ext[k]), float(vals[k].item())) for k in range(len(idx))]
            batch.clear()
            externals.clear()

        bs = int(self.params.get("infer_batch_size", 1024))
        for u in users:
            ui = self.user_id_to_idx.get(int(u), -1)
            if ui < 0:
                res[int(u)] = default_pairs
                continue
            batch.append(int(ui))
            externals.append(int(u))
            if len(batch) >= bs:
                flush_batch()
        flush_batch()
        return res

    def save(self, base_path: str) -> None:
        root = Path(base_path)
        root.mkdir(parents=True, exist_ok=True)
        model_name = self.__class__.__name__.replace("Adapter", "").lower()
        dst = root / f"{model_name}"
        dst.mkdir(parents=True, exist_ok=True)
        # Persist maps for inference
        with open(dst / "user_map.json", "w") as f:
            json.dump(self.user_id_to_idx, f)
        with open(dst / "item_map.json", "w") as f:
            json.dump(self.item_id_to_idx, f)

    @classmethod
    def load(cls, base_path: str) -> "RecBoleBaseAdapter":
        # Note: Loading trained RecBole state is out of scope; maps are enough for inference via full_sort.
        ad = cls()
        p = Path(base_path)
        with open(p / "user_map.json") as f:
            ad.user_id_to_idx = {int(k): int(v) for k, v in json.load(f).items()}
        with open(p / "item_map.json") as f:
            ad.item_id_to_idx = {int(k): int(v) for k, v in json.load(f).items()}
        ad.idx_to_user_id = [uid for uid, _ in sorted(ad.user_id_to_idx.items(), key=lambda x: x[1])]
        ad.idx_to_item_id = [iid for iid, _ in sorted(ad.item_id_to_idx.items(), key=lambda x: x[1])]
        return ad

    def build_validation_users(self, cfg_d: Dict[str, Any], cache_dir: Path) -> List[int]:
        # Default: read unique users from processed inter_features valid split
        paths = self._recbole_paths(cfg_d)
        inter_valid = Path(paths["inter_features_dir"]) / "valid"
        user_f = self._rb_fields.get("user_f")
        users: set[int] = set()
        for shard in sorted(inter_valid.glob("*.parquet")):
            lf = pl.scan_parquet(str(shard)).select([user_f]).unique()
            df = lf.collect(streaming=True)
            users.update([int(u) for u in df[user_f].to_list()])
        limit = cfg_d.get("validation_user_limit", -1)
        users_list = sorted(int(u) for u in users)
        if isinstance(limit, int) and limit > 0 and len(users_list) > limit:
            return users_list[:limit]
        return users_list

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Interaction:
        if not batch:
            return Interaction({})
        out: Dict[str, Any] = {}
        keys = batch[0].keys()
        bs = len(batch)
        label_f = self._rb_fields.get("label_f")

        for k in keys:
            vals = [b[k] for b in batch]
            # Handle list features (e.g., sequences): pad to max length, left-pad with zeros
            if isinstance(vals[0], list):
                max_len = max((len(v) for v in vals), default=0)
                # infer inner dtype from first non-empty list
                first_elem = None
                for vv in vals:
                    if vv:
                        first_elem = vv[0]
                        break
                is_float_seq = isinstance(first_elem, (float, np.floating)) if first_elem is not None else False
                if is_float_seq:
                    float_dtype_str = str(getattr(self, "float_seq_dtype", getattr(self, "params", {}).get("float_seq_dtype", "float16")))
                    float_dtype = torch.float16 if float_dtype_str == "float16" else torch.float32
                    if max_len == 0:
                        out[k] = torch.zeros((bs, 0), dtype=float_dtype)
                    else:
                        t = torch.zeros((bs, max_len), dtype=float_dtype)
                        for i, v in enumerate(vals):
                            if not v:
                                continue
                            lv = len(v)
                            t[i, max_len - lv:max_len] = torch.tensor(v, dtype=float_dtype)
                        out[k] = t
                else:
                    if max_len == 0:
                        out[k] = torch.zeros((bs, 0), dtype=torch.long)
                    else:
                        t = torch.zeros((bs, max_len), dtype=torch.long)
                        for i, v in enumerate(vals):
                            if not v:
                                continue
                            lv = len(v)
                            t[i, max_len - lv:max_len] = torch.tensor(v, dtype=torch.long)
                        out[k] = t
                continue

            v0 = vals[0]
            if isinstance(v0, torch.Tensor):
                out[k] = torch.stack(vals)
            elif isinstance(v0, (int, np.integer)):
                out[k] = torch.tensor(vals, dtype=torch.long)
            elif isinstance(v0, (float, np.floating)) or k == label_f:
                out[k] = torch.tensor(vals, dtype=torch.float32)
            else:
                out[k] = vals
        return Interaction(out)

    # --------- private impl ---------
    def _cfg_to_dict(self, cfg: Any) -> Dict[str, Any]:
        try:
            from omegaconf import OmegaConf
            return dict(OmegaConf.to_container(cfg, resolve=True))
        except Exception:
            return dict(cfg) if isinstance(cfg, dict) else {}

    def _make_cache_key(self, cfg_d: Dict[str, Any]) -> str:
        payload = {
            "model": self.params.get("model"),
            "seed": cfg_d.get("seed"),
            "split": cfg_d.get("split", {}),
        }
        s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]

    def _load_id_maps(self, user_map_path: Path, item_map_path: Path) -> None:
        with open(user_map_path) as f:
            self.user_id_to_idx = {int(k): int(v) for k, v in json.load(f).items()}
        with open(item_map_path) as f:
            self.item_id_to_idx = {int(k): int(v) for k, v in json.load(f).items()}
        self.idx_to_user_id = [uid for uid, _ in sorted(self.user_id_to_idx.items(), key=lambda x: x[1])]
        self.idx_to_item_id = [iid for iid, _ in sorted(self.item_id_to_idx.items(), key=lambda x: x[1])]

    def _build_id_maps_from_train_split(self, cfg_d: Dict[str, Any], user_map_path: Path, item_map_path: Path) -> None:
        logger.info("🔍 Сканирование train данных для построения ID маппингов...")
        inter_dir = Path(self._recbole_paths(cfg_d)["inter_features_dir"]) / "train"
        users: set[int] = set()
        items: set[int] = set()
        user_f = self._rb_fields.get("user_f")
        item_f = self._rb_fields.get("item_f")
        
        shard_files = sorted(inter_dir.glob("*.parquet"))
        logger.info(f"   - Найдено {len(shard_files)} файлов parquet")
        
        for i, shard in enumerate(shard_files):
            shard_start = time.time()
            lf = pl.scan_parquet(str(shard)).select([user_f, item_f]).unique()
            df = lf.collect(streaming=True)
            if df.height:
                users.update([int(u) for u in df[user_f].to_list()])
                items.update([int(i) for i in df[item_f].to_list()])
            
            if (i + 1) % 10 == 0 or i == len(shard_files) - 1:
                logger.info(f"   - Обработано {i + 1}/{len(shard_files)} файлов, "
                          f"найдено {len(users)} пользователей, {len(items)} товаров "
                          f"(последний файл: {time.time() - shard_start:.2f}s)")
                
        self.idx_to_user_id = sorted(int(u) for u in users)
        self.idx_to_item_id = sorted(int(i) for i in items)
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(self.idx_to_user_id)}
        self.item_id_to_idx = {iid: idx for idx, iid in enumerate(self.idx_to_item_id)}
        
        save_start = time.time()
        with open(user_map_path, "w") as f:
            json.dump(self.user_id_to_idx, f)
        with open(item_map_path, "w") as f:
            json.dump(self.item_id_to_idx, f)
        logger.info(f"💾 Сохранение маппингов на диск: {time.time() - save_start:.2f}s")

    # legacy .pt shard loader removed; we use parquet streaming only

    def _make_streaming_loader(
        self,
        files: List[Path],
    ) -> DataLoader:
        # num_workers=0 for deterministic order and simplicity
        batch_size = int(self.params.get("batch_size", 1024))
        paths = self._recbole_paths({"model": self.params})
        ds = _ParquetInteractionsDataset(
            files=files,
            rb_fields=self._rb_fields,
            user_id_to_idx=self.user_id_to_idx,
            item_id_to_idx=self.item_id_to_idx,
            items_dir=paths.get("items_dir"),
            users_dir=paths.get("users_dir"),
            float_seq_dtype=str(self.params.get("float_seq_dtype", "float16")),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)

    def _estimate_item_popularity_from_train(self) -> Dict[int, float]:
        # Rough popularity from train shards
        pop: Dict[int, float] = {}
        # Prefer reading directly from parquet if available
        item_f = self._rb_fields.get("item_f") or "item_id"
        paths = self._recbole_paths({"model": {}})
        train_dir = Path(paths["inter_features_dir"]) / "train"
        if train_dir.exists():
            for fp in sorted(train_dir.glob("*.parquet")):
                try:
                    df = pl.scan_parquet(str(fp)).select([item_f]).collect(streaming=True)
                    for iid in df[item_f].to_list():
                        pop[int(iid)] = pop.get(int(iid), 0.0) + 1.0
                except Exception:
                    continue
        return pop

    def _load_valid_users(self, meta_path: Path) -> List[int]:
        if not meta_path.exists():
            return []
        with open(meta_path) as f:
            meta = json.load(f)
        users = meta.get("valid_users", [])
        return [int(u) for u in users]

    def _load_valid_ground_truth(self, cfg_d: Dict[str, Any]) -> set[Tuple[int, int]]:
        logger.info("🎯 Загрузка ground truth для валидации...")
        
        # Use processed valid inter_features target as ground truth (positive pairs)
        paths = self._recbole_paths(cfg_d)
        inter_valid = Path(paths["inter_features_dir"]) / "valid"
        user_f = self._rb_fields.get("user_f")
        item_f = self._rb_fields.get("item_f")
        label_f = self._rb_fields.get("label_f")
        gt: set[Tuple[int, int]] = set()
        
        shard_files = sorted(inter_valid.glob("*.parquet"))
        logger.info(f"   - Обрабатывается {len(shard_files)} файлов")
        
        for idx, shard in enumerate(shard_files):
            shard_start = time.time()
            lf = pl.scan_parquet(str(shard)).select([user_f, item_f, label_f])
            df = lf.collect(streaming=True)
            if df.height == 0:
                continue
            for u, i, y in zip(df[user_f].to_list(), df[item_f].to_list(), df[label_f].to_list()):
                if y and int(y) > 0:
                    gt.add((int(u), int(i)))
            
            if (idx + 1) % 10 == 0 or idx == len(shard_files) - 1:
                logger.info(f"GT: {idx + 1}/{len(shard_files)}, "
                          f"найдено пар: {len(gt)}, "
                          f"время: {time.time() - shard_start:.2f}s")
        
        logger.info(f"🎯 Ground truth загружен: {len(gt)} положительных пар")
        return gt

    # --------- recbole config helpers ---------
    def _recbole_paths(self, cfg_d: Dict[str, Any]) -> Dict[str, str]:
        p = {**(cfg_d.get("model", {}) or {})}
        return {
            "inter_features_dir": p.get("inter_features_dir", "data/processed_features/inter_features"),
            "items_dir": p.get("items_dir", "data/processed_features/items_features"),
            "users_dir": p.get("users_dir", "data/processed_features/users_features"),
        }
    
    def _extract_recbole_fields(self, cfg_d: Dict[str, Any]) -> None:
        m = cfg_d.get("model", {}) or {}
        # Prefer recbole.yaml keys under model, fallback to adapter params
        user_f = str(m.get("USER_ID_FIELD", self.params.get("USER_ID_FIELD", "user_id")))
        item_f = str(m.get("ITEM_ID_FIELD", self.params.get("ITEM_ID_FIELD", "item_id")))
        label_f = str(m.get("LABEL_FIELD", self.params.get("LABEL_FIELD", "target")))
        time_f = str(m.get("TIME_FIELD", self.params.get("TIME_FIELD", "date")))
        inter_cols = m.get("inter_feature_cols", self.params.get("inter_feature_cols", []))
        item_cols = m.get("item_feature_cols", self.params.get("item_feature_cols", []))
        user_cols = m.get("user_feature_cols", self.params.get("user_feature_cols", []))
        auxiliary_cols = m.get("auxiliary_feature_cols", self.params.get("auxiliary_feature_cols", []))
        base_cols = m.get("base_cols", self.params.get("base_cols", []))
        self._rb_fields = {
            "user_f": user_f,
            "item_f": item_f,
            "label_f": label_f,
            "time_f": time_f,
            "base_cols": base_cols,
            "inter_cols": inter_cols,
            "item_cols": item_cols,
            "user_cols": user_cols,
            "auxiliary_feature_cols": auxiliary_cols,
        }

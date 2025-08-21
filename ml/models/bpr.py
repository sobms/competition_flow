from __future__ import annotations
from typing import Any, Dict, Iterable, Tuple, List, Set
from pathlib import Path
import json
import numpy as np
import polars as pl
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse import save_npz, load_npz
from implicit.bpr import BayesianPersonalizedRanking
from tqdm import tqdm

from ml.core.interfaces import BaseModel


class BPRAdapter(BaseModel):
	def __init__(self, **params: Any) -> None:
		super().__init__(**params)
		self.params = {
			**params
		}
		self.model = None
		self.user_id_to_idx: Dict[int, int] = {}
		self.item_id_to_idx: Dict[int, int] = {}
		self.idx_to_user_id: List[int] = []
		self.idx_to_item_id: List[int] = []
		self.user_item_csr: csr_matrix | None = None

	# -------- label builders (polars expressions) --------
	@property
	def label_expr_to_cart_and_delivered(self) -> pl.Expr:
		return ((pl.col("action_type") == "to_cart") & (pl.col("last_status") == "delivered_orders")).cast(pl.Float32)

	@property
	def label_expr_to_cart(self) -> pl.Expr:
		return (pl.col("action_type") == "to_cart").cast(pl.Float32)
	
	@property
	def target_competition_expr(self) -> pl.Expr:
		return (pl.col("last_status") == "delivered_orders").cast(pl.Float32)
	
	@property
	def combined_label_expr(self) -> pl.Expr:
		return (
			pl.when(pl.col("last_status") == "delivered_orders")
			.then(5.0)
			.when(pl.col("action_type") == "to_cart")
			.then(3.0)
			.when(pl.col("action_type") == "favorite") 
			.then(3.0)
			.when(pl.col("action_type") == "view_description")
			.then(1.0)
			.when(pl.col("action_type") == "review_view")
			.then(1.0)
			.otherwise(0.0)
		).cast(pl.Float32)

	def _label_expr(self, label_type: str) -> pl.Expr:
		if label_type == "to_cart_and_delivered":
			return self.label_expr_to_cart_and_delivered
		if label_type == "to_cart":
			return self.label_expr_to_cart
		if label_type == "delivered":
			return self.target_competition_expr
		if label_type == "combined":
			return self.combined_label_expr
		raise ValueError(f"Invalid label type: {label_type}")

	# -------- dataset preparation --------
	def prepare_dataset(self, train_cfg: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
		"""Build train/valid data for BPR without loading all into memory.
		Returns:
		- train_data: {csr, user_map, item_map}
		- valid_data: {ground_truth: Set[(user_idx,item_idx)], users: List[user_idx]}
		"""
		# ---------- cache handling ----------
		def _cfg_to_dict(cfg: Any) -> Dict[str, Any]:
			try:
				from omegaconf import OmegaConf
				return dict(OmegaConf.to_container(cfg, resolve=True))
			except Exception:
				return dict(cfg) if isinstance(cfg, dict) else {}

		cfg_d = _cfg_to_dict(train_cfg)
		artifacts_dir = Path(cfg_d.get("artifacts_dir", "artifacts"))
		cache_root = artifacts_dir / "cache" / "als_prepare"
		cache_root.mkdir(parents=True, exist_ok=True)
		# cache key from parameters impacting dataset
		import hashlib, json as _json
		key_payload = {
			"label_type": self.params.get("label_type"),
			"split": cfg_d.get("split", {}),
			"seed": cfg_d.get("seed"),
		}
		key_str = _json.dumps(key_payload, sort_keys=True, ensure_ascii=False)
		cache_key = hashlib.md5(key_str.encode("utf-8")).hexdigest()[:16]
		cache_dir = cache_root / cache_key
		user_map_path = cache_dir / "user_map.json"
		item_map_path = cache_dir / "item_map.json"
		csr_path = cache_dir / "user_item_csr.npz"
		gt_path = cache_dir / "ground_truth.json"
		meta_path = cache_dir / "meta.json"

		# try to load cache
		if user_map_path.exists() and item_map_path.exists() and csr_path.exists() and gt_path.exists():
			with open(user_map_path) as f:
				self.user_id_to_idx = {int(k): int(v) for k, v in json.load(f).items()}
			with open(item_map_path) as f:
				self.item_id_to_idx = {int(k): int(v) for k, v in json.load(f).items()}
			self.idx_to_user_id = [uid for uid, _ in sorted(self.user_id_to_idx.items(), key=lambda x: x[1])]
			self.idx_to_item_id = [iid for iid, _ in sorted(self.item_id_to_idx.items(), key=lambda x: x[1])]
			self.user_item_csr = load_npz(csr_path)
			with open(gt_path) as f:
				gt_list = json.load(f)
			gt: Set[Tuple[int, int]] = set((int(u), int(i)) for u, i in gt_list)
			valid_users = sorted({u for (u, _) in gt})
			train_data = {"csr": self.user_item_csr, "user_map": self.user_id_to_idx, "item_map": self.item_id_to_idx}
			valid_data = {"ground_truth": gt, "users": valid_users}
			return train_data, valid_data

		# ---------- build from scratch ----------
		splits_base = Path("data/splits")
		tracker_dir_train = splits_base / "train" / "ml_ozon_recsys_tracker_data"
		orders_dir_train = splits_base / "train" / "ml_ozon_recsys_orders_data"
		orders_dir_valid = splits_base / "valid" / "ml_ozon_recsys_orders_data"

		label_type: str = self.params.get("label_type")
		label_expr = self._label_expr(label_type).alias("label")

		# 1) First pass train: collect unique users/items
		users: Set[int] = set()
		items: Set[int] = set()
		for shard in tqdm(sorted(tracker_dir_train.glob("shard=*.parquet")), desc="collect ids (train)"):
			lf_t = pl.scan_parquet(str(shard)).select(["user_id", "item_id"]).unique()
			df_ids = lf_t.collect(streaming=True)
			if df_ids.height:
				users.update(df_ids["user_id"].to_list())
				items.update(df_ids["item_id"].to_list())
		# Build maps
		self.idx_to_user_id = sorted(users)
		self.idx_to_item_id = sorted(items)
		self.user_id_to_idx = {uid: i for i, uid in enumerate(self.idx_to_user_id)}
		self.item_id_to_idx = {iid: i for i, iid in enumerate(self.idx_to_item_id)}

		# Helper to join tracker and orders shard by shard (train only)
		def join_tracker_orders(tracker_shard: Path, orders_base: Path) -> pl.LazyFrame:
			lf_tracker = pl.scan_parquet(str(tracker_shard)).with_columns(pl.col("timestamp").dt.date().alias("date_t"))
			orders_shard = orders_base / tracker_shard.name
			lf_orders = pl.scan_parquet(str(orders_shard)).with_columns(pl.col("last_status_timestamp").dt.date().alias("date_o"))
			lf_result = lf_tracker.join(
				lf_orders.select(["user_id", "item_id", "last_status", "date_o"]),
				left_on=["user_id", "item_id", "date_t"],
				right_on=["user_id", "item_id", "date_o"],
				how="left",
			)
			return lf_result

		rows: List[int] = []
		cols: List[int] = []
		data: List[float] = []
		# 2) Second pass train: build COO
		for shard in tqdm(sorted(tracker_dir_train.glob("shard=*.parquet")), desc="build COO (train)"):
			lf = join_tracker_orders(shard, orders_dir_train).select([
				"user_id",
				"item_id",
				"action_type",
				"last_status",
				label_expr,
			])
			df = lf.collect(streaming=True)
			if df.height == 0:
				continue
			# map ids → idx
			u_idx = pl.Series([self.user_id_to_idx.get(int(u), -1) for u in df["user_id"]])
			i_idx = pl.Series([self.item_id_to_idx.get(int(i), -1) for i in df["item_id"]])
			mask = (u_idx != -1) & (i_idx != -1) & (df["label"] > 0)
			if mask.sum() == 0:
				continue
			rows.extend(u_idx.filter(mask).to_list())
			cols.extend(i_idx.filter(mask).to_list())
			data.extend(df["label"].filter(mask).to_list())

		n_users = len(self.idx_to_user_id)
		n_items = len(self.idx_to_item_id)
		print(f"[train] BPR training with n_users: {n_users}, n_items: {n_items}, positives count: {(df['label'] > 0).sum()}")
		coo = coo_matrix((np.array(data, dtype=np.float32), (np.array(rows), np.array(cols))), shape=(n_users, n_items))
		self.user_item_csr = coo.tocsr()

		# 3) Valid ground truth только из orders valid по target_competition_expr
		gt: Set[Tuple[int, int]] = set()
		for shard in tqdm(sorted(orders_dir_valid.glob("shard=*.parquet")), desc="build ground truth (valid)"):
			lf = pl.scan_parquet(str(shard)).select([
				"user_id",
				"item_id",
				"last_status",
				self.target_competition_expr.alias("label"),
			])
			df = lf.collect(streaming=True)
			if df.height == 0:
				continue
			for u, i, y in zip(df["user_id"].to_list(), df["item_id"].to_list(), df["label"].to_list()):
				# u_idx = self.user_id_to_idx.get(int(u))
				# i_idx = self.item_id_to_idx.get(int(i))
				# if u_idx is None or i_idx is None:
				# 	continue
				if y and y > 0:
					gt.add((u, i))

		train_data = {
			"csr": self.user_item_csr,
			"user_map": self.user_id_to_idx,
			"item_map": self.item_id_to_idx,
		}
		valid_data = {
			"ground_truth": gt,
			"users": sorted({u for (u, _) in gt}),
		}

		# Extract limit from cfg.train.validation_user_limit if present
		limit = train_cfg.get("validation_user_limit", -1)

		if limit is not None and limit > 0:
			all_users = valid_data["users"]
			if len(all_users) > limit:
				valid_data["users"] = all_users[:limit]


		# ---------- save cache ----------
		cache_dir.mkdir(parents=True, exist_ok=True)
		with open(user_map_path, "w") as f:
			json.dump(self.user_id_to_idx, f)
		with open(item_map_path, "w") as f:
			json.dump(self.item_id_to_idx, f)
		save_npz(csr_path, self.user_item_csr)
		with open(gt_path, "w") as f:
			json.dump(list(gt), f)
		with open(meta_path, "w") as f:
			json.dump({"n_users": n_users, "n_items": n_items, "cache_key": cache_key}, f)

		return train_data, valid_data

	# -------- training --------
	def fit(self, train_data: Dict[str, Any]) -> "BPRAdapter":
		csr: csr_matrix = train_data["csr"]
		model = BayesianPersonalizedRanking(
			factors=int(self.params.get("factors", 128)),
			learning_rate=float(self.params.get("learning_rate", 0.01)),
			regularization=float(self.params.get("regularization", 0.01)),
			iterations=int(self.params.get("iterations", 30)),
			use_gpu=bool(self.params.get("use_gpu", False)),
			num_threads=int(self.params.get("num_threads", 0)),
			verify_negative_samples=bool(self.params.get("verify_negative_samples", True)),
		)
		# BPR игнорирует веса и рассматривает ненулевые как лайк → бинаризуем
		binary = csr.copy()
		binary.data = (binary.data > 0).astype(np.float32)
		model.fit(binary)
		self.model = model
		return self

	# -------- inference --------
	def infer(self, users: Iterable[int], N: int = 100) -> Dict[int, List[Tuple[int, float]]]:
		assert self.model is not None and self.user_item_csr is not None
		# Precompute default popular items (external ids)
		item_scores = np.asarray(self.user_item_csr.sum(axis=0)).ravel()
		if item_scores.size == 0:
			return {}
		k = min(N, item_scores.size)
		top_idx = np.argpartition(-item_scores, k - 1)[:k]
		top_idx = top_idx[np.argsort(-item_scores[top_idx])]
		default_pairs: List[Tuple[int, float]] = [
			(self.idx_to_item_id[i], float(item_scores[i])) for i in top_idx
		]

		# Build mapping external -> internal for known users, and list unknowns
		known_pairs: List[Tuple[int, int]] = []  # (u_ext, u_int)
		unknown_users: List[int] = []
		for u in users:
			ui = self.user_id_to_idx.get(u, -1)
			if ui is not None and ui >= 0:
				known_pairs.append((u, ui))
			else:
				unknown_users.append(u)

		res: Dict[int, List[Tuple[int, float]]] = {}
		# Fill defaults for unknown users
		for u in unknown_users:
			res[u] = default_pairs

		# Batched recommend for known users; keep keys as external user ids
		batch_size = int(self.params.get("infer_batch_size", 1024))
		for start in tqdm(range(0, len(known_pairs), batch_size), desc="infer users (batched)"):
			end = min(start + batch_size, len(known_pairs))
			batch_pairs = known_pairs[start:end]
			batch_user_idx = [ui for (_, ui) in batch_pairs]
			items, scores = self.model.recommend(
				np.array(batch_user_idx, dtype=np.int32),
				self.user_item_csr[batch_user_idx],
				N=N,
				filter_already_liked_items=False,
			)
			for i, (u_ext, _) in enumerate(batch_pairs):
				item_ids_ext = [self.idx_to_item_id[j] for j in items[i].tolist()]
				pairs = list(zip(item_ids_ext, scores[i].tolist()))
				pairs.sort(key=lambda x: x[1], reverse=True)
				res[u_ext] = pairs
		return res

	# -------- persistence --------
	def save(self, base_path: str) -> None:
		from datetime import datetime
		root = Path(base_path)
		root.mkdir(parents=True, exist_ok=True)
		model_name = self.__class__.__name__.replace("Adapter", "").lower()
		subdir = f"{model_name}_{datetime.now():%Y%m%d_%H%M}"
		dst = root / subdir
		dst.mkdir(parents=True, exist_ok=True)
		# save factors (if available)
		if self.model is not None:
			uf = getattr(self.model, "user_factors", None)
			if uf is not None:
				uf_np = uf.to_numpy() if hasattr(uf, "to_numpy") else uf
				np.save(dst / "user_factors.npy", uf_np)
			if getattr(self.model, "item_factors", None) is not None:
				if_np = self.model.item_factors
				if hasattr(if_np, "to_numpy"):
					if_np = if_np.to_numpy()
				np.save(dst / "item_factors.npy", if_np)
		# save maps
		with open(dst / "user_map.json", "w") as f:
			json.dump(self.user_id_to_idx, f)
		with open(dst / "item_map.json", "w") as f:
			json.dump(self.item_id_to_idx, f)
		# save meta (shapes)
		n_users = len(self.user_id_to_idx)
		n_items = len(self.item_id_to_idx)
		with open(dst / "meta.json", "w") as f:
			json.dump({"n_users": n_users, "n_items": n_items}, f)

	@classmethod
	def load(cls, base_path: str) -> "BPRAdapter":
		from implicit.bpr import BayesianPersonalizedRanking
		from scipy.sparse import csr_matrix as _csr
		ad = cls()
		root = Path(base_path)
		p = root
		# If base_path points to parent, pick the latest timestamped subdir
		if not (root / "user_map.json").exists():
			candidates = [d for d in root.iterdir() if d.is_dir()]
			if not candidates:
				raise FileNotFoundError(f"No saved BRP model directories under {base_path}")
			p = max(candidates, key=lambda d: d.stat().st_mtime)
		# maps
		with open(p / "user_map.json") as f:
			ad.user_id_to_idx = {int(k): int(v) for k, v in json.load(f).items()}
		with open(p / "item_map.json") as f:
			ad.item_id_to_idx = {int(k): int(v) for k, v in json.load(f).items()}
		ad.idx_to_user_id = [uid for uid, _ in sorted(ad.user_id_to_idx.items(), key=lambda x: x[1])]
		ad.idx_to_item_id = [iid for iid, _ in sorted(ad.item_id_to_idx.items(), key=lambda x: x[1])]
		# meta and factors
		with open(p / "meta.json") as f:
			meta = json.load(f)
		n_users = int(meta.get("n_users", len(ad.idx_to_user_id)))
		n_items = int(meta.get("n_items", len(ad.idx_to_item_id)))
		uf = np.load(p / "user_factors.npy")
		if_path_items = p / "item_factors.npy"
		if not if_path_items.exists():
			raise FileNotFoundError("item_factors.npy is missing")
		vf = np.load(if_path_items)
		# factors for BPR GPU include bias term (+1) → восстановим число факторов
		factors = int(uf.shape[1] - 1) if len(uf.shape) == 2 and uf.shape[1] > 0 else 128
		model = BayesianPersonalizedRanking(
			factors=factors,
			learning_rate=float(self.params.get("learning_rate", 0.01)) if hasattr(self, "params") else 0.01,
			regularization=0.01,
			iterations=1,
			use_gpu=True,
			num_threads=0,
			verify_negative_samples=True,
		)
		# attach factors directly
		import implicit.gpu as _igpu
		model.user_factors = _igpu.Matrix(uf)
		model.item_factors = _igpu.Matrix(vf)
		ad.model = model
		# create empty CSR with correct shape for recommend API
		ad.user_item_csr = _csr((n_users, n_items), dtype=np.float32)
		return ad

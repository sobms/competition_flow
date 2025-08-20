from __future__ import annotations
from typing import Any, Dict, Iterable, Tuple, List, Set
from pathlib import Path
import json
import numpy as np
import polars as pl
from scipy.sparse import coo_matrix, csr_matrix
from implicit.als import AlternatingLeastSquares
from tqdm import tqdm

from ml.core.interfaces import BaseModel


class ALSAdapter(BaseModel):
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
		"""Build train/valid data for ALS without loading all into memory.
		Returns:
		- train_data: {csr, user_map, item_map}
		- valid_data: {ground_truth: Set[(user_idx,item_idx)], users: List[user_idx]}
		"""
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
		print(f"[train] ALS training with n_users: {n_users}, n_items: {n_items}, positives count: {(df['label'] > 0).sum()}")
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

		return train_data, valid_data

	# -------- training --------
	def fit(self, train_data: Dict[str, Any]) -> "ALSAdapter":
		csr: csr_matrix = train_data["csr"]
		model = AlternatingLeastSquares(
			factors=int(self.params.get("factors", 128)),
			regularization=float(self.params.get("regularization", 0.01)),
			iterations=int(self.params.get("iterations", 30)),
			use_gpu=bool(self.params.get("use_gpu", False)),
			num_threads=int(self.params.get("num_threads", 0)),
		)
		alpha = float(self.params.get("alpha", 1.0))
		model.fit(csr * alpha)
		self.model = model
		return self

	# -------- inference --------
	def infer(self, users: Iterable[int], N: int = 100) -> Dict[int, List[Tuple[int, float]]]:
		assert self.model is not None and self.user_item_csr is not None
		users_idx = [self.user_id_to_idx.get(u, -1) for u in users]
		res: Dict[int, List[Tuple[int, float]]] = {}
		for u in tqdm(list(users_idx), desc="infer users"):
			items, scores = self.model.recommend(u, self.user_item_csr, N=N, filter_already_liked_items=False)
			pairs = list(zip(items.tolist(), scores.tolist()))
			pairs.sort(key=lambda x: x[1], reverse=True)
			res[u] = pairs
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
				np.save(dst / "user_factors.npy", uf)
			if getattr(self.model, "item_factors", None) is not None:
				np.save(dst / "item_factors.npy", self.model.item_factors)
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
	def load(cls, base_path: str) -> "ALSAdapter":
		from implicit.als import AlternatingLeastSquares
		from scipy.sparse import csr_matrix as _csr
		ad = cls()
		root = Path(base_path)
		p = root
		# If base_path points to parent, pick the latest timestamped subdir
		if not (root / "user_map.json").exists():
			candidates = [d for d in root.iterdir() if d.is_dir()]
			if not candidates:
				raise FileNotFoundError(f"No saved ALS model directories under {base_path}")
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
		model = AlternatingLeastSquares(factors=uf.shape[1], regularization=0.0, iterations=1, use_gpu=False, num_threads=0)
		# attach factors directly
		model.user_factors = uf
		model.item_factors = vf
		ad.model = model
		# create empty CSR with correct shape for recommend API
		ad.user_item_csr = _csr((n_users, n_items), dtype=np.float32)
		return ad

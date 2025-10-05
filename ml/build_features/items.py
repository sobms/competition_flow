from __future__ import annotations

from typing import Any, Dict
from pathlib import Path

import polars as pl
from tqdm import tqdm

from .base import BaseFeaturesFactory
from .io import iter_parquet_shards, scan_selected


class ItemsFeaturesFactory(BaseFeaturesFactory):
	def __init__(self, cfg: Dict[str, Any]) -> None:
		super().__init__(cfg)
		self.items_dir = Path(cfg["items_dir"])  # required
		self.num_shards = int(cfg.get("shards", 128))
		# Параметры для атрибутов
		self.min_freq_threshold = float(cfg.get("attributes_min_freq_pct", 0.01))  # 1% по умолчанию
		self.include_attributes = bool(cfg.get("include_attributes", True))

	def build(self, split: str) -> None:
		prefix = self.cfg.get("items_out", "items_features")
		tasks = [(self.cfg, str(s), prefix) for s in iter_parquet_shards(self.items_dir, "part-*.parquet")]
		# self._parallel_map(tasks, items_worker)
		for task in tqdm(tasks, desc=f"{self.__class__.__name__}"):
			items_worker(task)

	def _base_item_features(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
		return df.select([
			pl.col("item_id").cast(pl.Int64),
			pl.col("fclip_embed"),
			pl.col("catalogid").cast(pl.Utf8),
			pl.col("variant_id").cast(pl.Utf8),
			pl.col("model_id").cast(pl.Utf8),
		])

	def _attributes_item_features(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
		"""
		Извлекает признаки из поля attributes используя:
		- Категориальное кодирование для основных атрибутов (Type, Brand, etc.)
		- Мульти-hot кодирование для множественных значений (ColorBase, Material, etc.)
		- Фильтрация значений, встречающихся менее чем в 1% товаров
		"""
		# Если работаем с LazyFrame, сначала материализуем для анализа частот
		if isinstance(df, pl.LazyFrame):
			df_materialized = df.collect()
		else:
			df_materialized = df
		
		total_items = len(df_materialized)
		min_frequency = max(1, int(total_items * self.min_freq_threshold))  # Порог из конфига
		
		# Развернуть все атрибуты в отдельные строки
		attrs_expanded = df_materialized.explode("attributes").filter(
			pl.col("attributes").is_not_null()
		).select([
			pl.col("item_id"),
			pl.col("attributes").struct.field("attribute_name").alias("attr_name"),
			pl.col("attributes").struct.field("attribute_value").alias("attr_value"),
			pl.col("attributes").struct.field("attribute_isaspect").alias("is_aspect"),
			pl.col("attributes").struct.field("attribute_showascharacteristic").alias("show_char")
		])
		
		# Вычислить частоты для каждой пары (attribute_name, attribute_value)
		attr_frequencies = (
			attrs_expanded
			.group_by(["attr_name", "attr_value"])
			.agg(pl.len().alias("freq"))
			.filter(pl.col("freq") >= min_frequency)
		)
		
		# Получить словари частых значений для каждого типа атрибута
		frequent_values = {}
		for attr_name in ["Type", "Brand", "SexMaster", "Season", "StyleApparel", "ColorBase", "Material", "AgeO"]:
			values = (
				attr_frequencies
				.filter(pl.col("attr_name") == attr_name)
				.select("attr_value")
				.to_series()
				.to_list()
			)
			if values:
				frequent_values[attr_name] = values
		
		# Начинаем с базового DataFrame
		result = df_materialized.select(["item_id"])
		
		# Фильтруем расширенные атрибуты только до частых значений
		frequent_attrs = attrs_expanded.join(
			attr_frequencies.select(["attr_name", "attr_value"]),
			on=["attr_name", "attr_value"],
			how="inner"
		)
		
		# 1. Категориальное кодирование для основных атрибутов
		categorical_attrs = ["Type", "Brand", "SexMaster", "Season", "StyleApparel", "AgeO"]
		
		for attr_name in categorical_attrs:
			if attr_name in frequent_values:
				# Получить первое значение для каждого item_id
				attr_values = (
					frequent_attrs
					.filter(pl.col("attr_name") == attr_name)
					.group_by("item_id")
					.agg(pl.col("attr_value").first().alias(f"{attr_name.lower()}_main"))
				)
				result = result.join(attr_values, on="item_id", how="left")
		
		# 2. Мульти-hot кодирование для ColorBase
		if "ColorBase" in frequent_values:
			for color in frequent_values["ColorBase"]:
				safe_color = color.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
				color_items = (
					frequent_attrs
					.filter((pl.col("attr_name") == "ColorBase") & (pl.col("attr_value") == color))
					.select("item_id")
					.unique()
					.with_columns(pl.lit(True).alias(f"color_{safe_color}"))
				)
				result = result.join(color_items, on="item_id", how="left").with_columns(
					pl.col(f"color_{safe_color}").fill_null(False)
				)
		
		# 3. Мульти-hot кодирование для Material
		if "Material" in frequent_values:
			for material in frequent_values["Material"]:
				safe_material = material.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
				material_items = (
					frequent_attrs
					.filter((pl.col("attr_name") == "Material") & (pl.col("attr_value") == material))
					.select("item_id")
					.unique()
					.with_columns(pl.lit(True).alias(f"material_{safe_material}"))
				)
				result = result.join(material_items, on="item_id", how="left").with_columns(
					pl.col(f"material_{safe_material}").fill_null(False)
				)
		
		# 4. Числовые агрегаты через векторизованные операции
		attr_counts = (
			attrs_expanded
			.group_by("item_id")
			.agg([
				pl.len().alias("n_attributes"),
				pl.col("is_aspect").sum().alias("n_aspects"),
				pl.col("show_char").sum().alias("n_show_characteristics")
			])
		)
		result = result.join(attr_counts, on="item_id", how="left").with_columns([
			pl.col("n_attributes").fill_null(0),
			pl.col("n_aspects").fill_null(0),
			pl.col("n_show_characteristics").fill_null(0)
		])
		
		return result
	

	
	def _write_parquet(self, df: pl.DataFrame, prefix: str, src_path: Path) -> None:
		out_dir = self.out_root / prefix / "raw"
		out_dir.mkdir(parents=True, exist_ok=True)
		df.write_parquet(str(out_dir / src_path.name))

def items_worker(task: tuple) -> None:
	cfg, shard_path, prefix = task
	f = ItemsFeaturesFactory(cfg)
	lf = scan_selected(Path(shard_path), [
		"item_id","catalogid","variant_id","model_id","itemname","attributes","fclip_embed"
	])
	
	# Получить базовые фичи
	lf_base = f._base_item_features(lf)
	df_base = lf_base.collect(streaming=True)
	
	# Добавить фичи из атрибутов, если включено
	if f.include_attributes:
		lf_full = scan_selected(Path(shard_path), [
			"item_id","attributes"
		])
		df_attrs = f._attributes_item_features(lf_full)
		# Объединить все фичи
		df_feat = df_base.join(df_attrs, on="item_id", how="left")
	else:
		df_feat = df_base
	
	f._write_parquet(df_feat, prefix, Path(shard_path))

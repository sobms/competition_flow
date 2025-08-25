from __future__ import annotations

from typing import Any, Dict
from pathlib import Path
import shutil
import hydra
from omegaconf import DictConfig, OmegaConf

from .items import ItemsFeaturesFactory
from .inter import InteractionsFeaturesFactory
from .users import UsersFeaturesFactory


@hydra.main(config_path="../../conf", config_name="build_features", version_base=None)
def main(cfg: DictConfig) -> None:
	cfg_d: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
	print("[build_features] Starting pipeline...")
	# Clean output directories for a fresh run
	out_root = Path(cfg_d.get("out_root", "data/processed_features"))
	to_clean = [
		cfg_d.get("inter_out", "inter_features"),
		cfg_d.get("items_out", "items_features"),
		cfg_d.get("users_out", "users_features"),
	]
	for prefix in to_clean:
		p = out_root / prefix
		if p.exists():
			print(f"[build_features] Removing existing output: {p}")
			shutil.rmtree(p)

	items_f = ItemsFeaturesFactory(cfg_d)
	inter_f = InteractionsFeaturesFactory(cfg_d)
	users_f = UsersFeaturesFactory(cfg_d)
	# items are common for both splits
	items_f.build(split="common")
	users_f.build(split="common")
	# train/valid
	inter_f.build(split="train")
	inter_f.build(split="valid")
	print("[build_features] Pipeline completed.")


if __name__ == "__main__":
	main()


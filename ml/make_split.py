from __future__ import annotations
import os
from hydra import initialize, compose
from omegaconf import OmegaConf
from ml.core.data import load_raw, make_splits, DataPaths


def main() -> None:
	with initialize(config_path="conf", version_base=None):
		cfg = compose(config_name="config")
	paths = DataPaths(
		raw_dir=cfg.data.paths.raw_dir,
		splits_dir=cfg.data.paths.splits_dir,
		features_dir=cfg.data.paths.features_dir,
	)
	df = load_raw(OmegaConf.to_container(cfg, resolve=True))
	os.makedirs(paths.splits_dir, exist_ok=True)
	make_splits(df, OmegaConf.to_container(cfg, resolve=True), paths)
	print(f"Splits written to {paths.splits_dir}")


if __name__ == "__main__":
	main()

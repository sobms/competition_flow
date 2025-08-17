from __future__ import annotations
import os
from hydra import initialize, compose
from omegaconf import OmegaConf
from ml.core.data import build_features, DataPaths


def main() -> None:
	with initialize(config_path="conf", version_base=None):
		cfg = compose(config_name="config")
	paths = DataPaths(
		raw_dir=cfg.data.paths.raw_dir,
		splits_dir=cfg.data.paths.splits_dir,
		features_dir=cfg.data.paths.features_dir,
	)
	os.makedirs(paths.features_dir, exist_ok=True)
	build_features(OmegaConf.to_container(cfg, resolve=True), paths)
	print(f"Features written to {paths.features_dir}")


if __name__ == "__main__":
	main()

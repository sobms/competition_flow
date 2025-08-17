from __future__ import annotations
import os
from omegaconf import OmegaConf
from hydra import initialize, compose


def main() -> None:
	with initialize(config_path="../conf", version_base=None):
		cfg = compose(config_name="config")
	raw_dir = cfg.data.paths.raw_dir
	os.makedirs(raw_dir, exist_ok=True)
	# Optionally you could download/copy data from HF to local cache here
	print(f"Prepared raw dir at {raw_dir}")


if __name__ == "__main__":
	main()

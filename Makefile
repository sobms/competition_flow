.PHONY: train validate infer

train:
	python -m ml.train

validate:
	python -m ml.train pipeline.mode=validate

infer:
	python -m ml.infer split=test

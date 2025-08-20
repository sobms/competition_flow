from __future__ import annotations
from typing import Any, Dict, Iterable, List, Tuple
import abc


class BaseModel(abc.ABC):
	def __init__(self, **kwargs: Any) -> None:
		self._init_params: Dict[str, Any] = dict(kwargs)

	@abc.abstractmethod
	def prepare_dataset(self, cfg: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
		"""Return (train_data, valid_data) prepared for the model.
		Implement stream-friendly processing and avoid full in-memory loads.
		"""
		...

	@abc.abstractmethod
	def fit(self, train_data: Any) -> "BaseModel":
		"""Train the model on prepared train_data."""
		...

	@abc.abstractmethod
	def infer(self, users: Iterable[int], N: int = 100) -> Dict[int, List[Tuple[int, float]]]:
		"""Return top-N items with scores for each user, sorted by score desc."""
		...

	def save(self, path: str) -> None:
		raise NotImplementedError

	@classmethod
	def load(cls, path: str) -> "BaseModel":
		raise NotImplementedError

	def get_params(self) -> Dict[str, Any]:
		return dict(self._init_params)

	def set_params(self, **params: Any) -> None:
		self._init_params.update(params)

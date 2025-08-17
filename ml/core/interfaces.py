from __future__ import annotations
from typing import Any, Dict, Protocol, runtime_checkable
import abc


class BaseModel(abc.ABC):
	def __init__(self, **kwargs: Any) -> None:
		self._init_params: Dict[str, Any] = dict(kwargs)

	@abc.abstractmethod
	def fit(self, train_df: Any, valid_df: Any | None = None) -> "BaseModel":
		...

	@abc.abstractmethod
	def predict(self, df: Any) -> Any:
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

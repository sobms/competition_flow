from __future__ import annotations

from typing import Any

from .recbole_base import RecBoleBaseAdapter


class SASRecAdapter(RecBoleBaseAdapter):
	def __init__(self, **params: Any) -> None:
		# Ensure Recbole model is SASRec while keeping other params intact
		p = {**params}
		p.setdefault("model", "SASRec")
		super().__init__(**p)



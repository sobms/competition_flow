from __future__ import annotations

from typing import Any

from .recbole_base import RecBoleBaseAdapter


class DSSMAdapter(RecBoleBaseAdapter):
	def __init__(self, **params: Any) -> None:
		# Ensure Recbole model is DSSM while keeping other params intact
		p = {**params}
		p.setdefault("model", "DSSM")
		super().__init__(**p)




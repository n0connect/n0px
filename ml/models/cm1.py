# ml/models/cm1.py
from __future__ import annotations
from typing import Any, Dict
from ._mixture_vae_base import ComplexMixtureVAE

class CM1_ComplexAutoEncoder(ComplexMixtureVAE):
    """Complex uzay - DP stream (labels 3,4,5 ama train tarafında map edilip 0,1,2 yapılır)."""
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)

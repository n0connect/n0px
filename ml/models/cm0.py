# ml/models/cm0.py
from __future__ import annotations
from typing import Any, Dict
from ._mixture_vae_base import ComplexMixtureVAE

class CM0_ComplexAutoEncoder(ComplexMixtureVAE):
    """Complex uzay - RAW stream (labels 0,1,2)."""
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)

# ml/models/rm0.py
from __future__ import annotations
from typing import Any, Dict
from ._mixture_vae_base import RealMixtureVAE

class RM0_RealAutoEncoder(RealMixtureVAE):
    """Reel uzay - RAW stream (labels 0,1,2)."""
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)

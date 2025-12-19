# ml/integrity_guard.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple
import time
import numpy as np

@dataclass
class GuardStats:
    total_batches: int = 0
    dropped_batches: int = 0
    last_ok_ts: float = 0.0
    last_reason: str = ""

def _has_dp(snap_dict: Dict[str, float], dp_cfg: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Robust DP detection with config-driven thresholds.
    
    Returns dict with evidence scores:
        {
            'mid': float (0-1),
            'entropy': float (0-0.25),
            'mean': float (0-1),
            'std': float,
            'dp_score': float (0-1, higher = more DP-like),
            'decision': 'DP' | 'NO_DP' | 'INCONCLUSIVE'
        }
    
    Uses config thresholds:
      - mid_threshold: %values in (0.1, 0.9)
      - entropy_threshold: E[x(1-x)] proxy
      - mean_range: allowed mean for DP (centered near 0.5)
      - low_*: boundaries for NO_DP decision
    """
    # Load config with defaults
    if dp_cfg is None:
        dp_cfg = {}
    
    mid_th = dp_cfg.get("mid_threshold", 0.80)
    entropy_th = dp_cfg.get("entropy_threshold", 0.20)
    mean_min = dp_cfg.get("mean_range_min", 0.35)
    mean_max = dp_cfg.get("mean_range_max", 0.65)
    low_mid_th = dp_cfg.get("low_mid_threshold", 0.50)
    low_entropy_th = dp_cfg.get("low_entropy_threshold", 0.10)
    
    # Extract snapshot
    mid = snap_dict.get("mid(0.1-0.9)", 0.0)
    entropy = snap_dict.get("entropy_proxy_E[x(1-x)]", 0.0)
    mean_x = snap_dict.get("mean", 0.0)
    std_x = snap_dict.get("std", 0.0)
    
    # DP signals (multiple lines of evidence)
    has_high_mid = mid > mid_th
    has_high_entropy = entropy > entropy_th
    has_mid_mean = mean_min <= mean_x <= mean_max
    
    # Compute evidence score (0-1)
    mid_score = min(1.0, mid / max(mid_th, 0.01))
    entropy_score = min(1.0, entropy / max(entropy_th, 0.01))
    mean_score = 1.0 - abs(mean_x - 0.5) / 0.5  # Peak at 0.5, drop to 0 at edges
    
    # Combined score (weighted average of evidence)
    dp_score = 0.4 * mid_score + 0.4 * entropy_score + 0.2 * mean_score
    dp_score = min(1.0, max(0.0, dp_score))
    
    # Decision logic
    if (has_high_mid and has_high_entropy and has_mid_mean):
        decision = "DP"
    elif mid < low_mid_th and entropy < low_entropy_th:
        decision = "NO_DP"
    else:
        decision = "INCONCLUSIVE"
    
    return {
        "mid": float(mid),
        "entropy": float(entropy),
        "mean": float(mean_x),
        "std": float(std_x),
        "dp_score": float(dp_score),
        "decision": decision,
    }


def _feature_snapshot(x: np.ndarray) -> Dict[str, float]:
    # x: (B,D) - NumPy array
    # Optimized: Single pass calculations where possible
    x_float = x.astype(np.float32) if x.dtype != np.float32 else x
    
    # Single pass for multiple calculations
    near0 = float(np.mean(np.abs(x_float - 0.0) < 0.10))
    near1 = float(np.mean(np.abs(x_float - 1.0) < 0.10))
    mid = float(np.mean((x_float > 0.10) & (x_float < 0.90)))
    
    # Entropy proxy: E[x(1-x)] 
    # Measures distance from 0/1 (higher = more entropy in Bernoulli sense)
    entropy_proxy = float(np.mean(x_float * (1.0 - x_float)))
    
    return {
        "min": float(x_float.min()),
        "max": float(x_float.max()),
        "mean": float(x_float.mean()),
        "std": float(x_float.std()),
        "near0": near0,
        "near1": near1,
        "mid(0.1-0.9)": mid,
        "entropy_proxy_E[x(1-x)]": entropy_proxy,  # DP fingerprint
    }

def _extract_labels(batch: Any) -> Optional[np.ndarray]:
    for k in ("labels", "y", "label"):
        if hasattr(batch, k):
            v = getattr(batch, k)
            if v is None:
                continue
            arr = np.asarray(v)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            return arr
    return None

def _extract_features(batch: Any) -> Optional[np.ndarray]:
    for k in ("features", "x", "input", "input_vector"):
        if hasattr(batch, k):
            v = getattr(batch, k)
            if v is None:
                continue
            arr = np.asarray(v)
            return arr
    return None

class VerifiedStreamDataIterator:
    """
    StreamDataIterator wrapper:
      - allowed_labels enforcement (RAW vs DP)
      - DP label remap (optional)
      - shape/finite checks
      - feature distribution snapshot
      - DP detection with config-driven thresholds
    """
    def __init__(
        self,
        base_iter: Any,
        cfg: Dict[str, Any],
        *,
        allowed_labels: Sequence[int],
        label_remap: Optional[Dict[int, int]] = None,
        expected_dim: Optional[int] = None,
        mode_tag: str = "RM0",
        max_wait_s: float = 2.0,
        strict: bool = True,
        dp_cfg: Optional[Dict[str, float]] = None,
    ):
        self.it = base_iter
        self.cfg = cfg
        self.dp_cfg = dp_cfg or {}  # DP detection config
        self.allowed = set(int(x) for x in allowed_labels)
        self.remap = {int(k): int(v) for k, v in (label_remap or {}).items()}
        self.expected_dim = int(expected_dim) if expected_dim is not None else None
        self.mode_tag = mode_tag
        self.max_wait_s = float(max_wait_s)
        self.strict = bool(strict)
        self.stats = GuardStats()

    def start(self) -> None:
        self.it.start()

    def stop(self) -> None:
        self.it.stop()

    def health_snapshot(self) -> Dict[str, Any]:
        return {
            "mode": self.mode_tag,
            "total_batches": self.stats.total_batches,
            "dropped_batches": self.stats.dropped_batches,
            "last_ok_ts": self.stats.last_ok_ts,
            "last_reason": self.stats.last_reason,
        }

    def get_batch(self) -> Any:
        t0 = time.time()
        consecutive_rejects = 0
        max_consecutive_rejects = 100000  # Very high: let time-based timeout dominate
        
        while True:
            if time.time() - t0 > self.max_wait_s:
                raise RuntimeError(f"[{self.mode_tag}] No valid batch within {self.max_wait_s}s (rejected {consecutive_rejects} batches). Last={self.stats.last_reason}")
            
            # Only raise if we're rejecting for OTHER reasons (not labels)
            if consecutive_rejects > max_consecutive_rejects and "not subset" not in self.stats.last_reason:
                raise RuntimeError(f"[{self.mode_tag}] Too many consecutive rejects ({consecutive_rejects}). Last={self.stats.last_reason}")

            batch = self.it.get_batch() if hasattr(self.it, "get_batch") else self.it.get_next_batch()
            if batch is None:
                self._drop("batch=None")
                consecutive_rejects += 1
                continue

            x = _extract_features(batch)
            y = _extract_labels(batch)

            if x is None:
                self._drop("features missing")
                consecutive_rejects += 1
                continue

            if x.ndim != 2:
                self._drop(f"features shape {x.shape} != (B,D)")
                consecutive_rejects += 1
                continue

            if self.expected_dim is not None and int(x.shape[1]) != self.expected_dim:
                self._drop(f"dim mismatch D={x.shape[1]} expected={self.expected_dim}")
                consecutive_rejects += 1
                continue

            if not np.isfinite(x).all():
                self._drop("non-finite features")
                consecutive_rejects += 1
                continue

            if y is not None:
                y = y.astype(np.int32, copy=False)
                uniq = set(int(v) for v in np.unique(y))
                if not uniq.issubset(self.allowed):
                    # AUTO-FILTER: Keep only allowed labels instead of rejecting
                    mask = np.isin(y, list(self.allowed))
                    if mask.sum() > 0:
                        # Filter batch to keep only allowed rows
                        x = x[mask]
                        y = y[mask]
                        if hasattr(batch, "features"):
                            batch.features = x
                        if hasattr(batch, "labels"):
                            batch.labels = y
                        elif hasattr(batch, "y"):
                            batch.y = y
                        elif hasattr(batch, "label"):
                            batch.label = y
                    else:
                        # No valid rows in this batch, reject it
                        self._drop(f"labels {sorted(list(uniq))} not subset of {sorted(list(self.allowed))} (and no valid rows after filter)")
                        consecutive_rejects += 1
                        continue
                # remap if requested
                if self.remap:
                    y2 = y.copy()
                    for src, dst in self.remap.items():
                        y2[y == src] = dst
                    # write back to batch if possible
                    if hasattr(batch, "labels"):
                        batch.labels = y2
                    elif hasattr(batch, "y"):
                        batch.y = y2
                    elif hasattr(batch, "label"):
                        batch.label = y2

            # feature snapshot (attached to batch for training inspection)
            snap = _feature_snapshot(x.astype(np.float32, copy=False))
            setattr(batch, "_guard_snapshot", snap)
            
            # DP detection (attached for optional monitoring)
            dp_result = _has_dp(snap, dp_cfg=self.dp_cfg)
            setattr(batch, "_dp_detection", dp_result)

            self.stats.total_batches += 1
            self.stats.last_ok_ts = time.time()
            self.stats.last_reason = "ok"
            return batch

    def _drop(self, reason: str) -> None:
        self.stats.dropped_batches += 1
        self.stats.last_reason = reason
        if self.strict:
            # strict=false yaparsan sadece drop’lar, strict=true ise ağır hata koşullarını sen script’te fail edebilirsin
            pass

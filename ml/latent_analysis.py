# -*- coding: utf-8 -*-
"""
Latent Space Separation Analysis
==================================
Window-based latent statistics tracker for per-class separation measurement.
- Collects last `window` latents per class
- Computes Fisher separation ratio + Mahalanobis distances
- Analysis-only: model training does NOT use these metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


def _to_1d_long(labels) -> torch.Tensor:
    """Convert various label formats to 1D long tensor."""
    if labels is None:
        raise ValueError("labels is None (need labels for analysis only)")
    
    # Convert numpy to torch if needed
    if not isinstance(labels, torch.Tensor):
        labels = torch.from_numpy(labels) if hasattr(labels, 'shape') else torch.tensor(labels)
    
    # Handle shape variations
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    
    return labels.long()


@dataclass
class SepMetrics:
    """Latent separation metrics snapshot."""
    fisher: float
    md_01: float
    md_02: float
    md_12: float
    counts: Tuple[int, int, int]


class LatentSeparationTracker:
    """
    Window-based latent stats:
      - collects last `window` latents per class
      - computes mean/cov per class
      - outputs Fisher separation + Mahalanobis distances

    Notes:
      - This is analysis only; model training does NOT use labels.
      - Uses float32 for stability; adds eps*I regularization to cov.
    """

    def __init__(
        self,
        num_classes: int = 3,
        window: int = 4096,
        eps: float = 1e-4,
        device: Optional[torch.device] = None,
    ):
        self.C = int(num_classes)
        self.window = int(window)
        self.eps = float(eps)
        self.device = device

        self._buf = [None for _ in range(self.C)]  # list[Tensor | None]
        self._counts = [0 for _ in range(self.C)]
        self._latent_dim: Optional[int] = None

    @property
    def latent_dim(self) -> Optional[int]:
        return self._latent_dim

    def reset(self) -> None:
        """Reset buffers and counts."""
        self._buf = [None for _ in range(self.C)]
        self._counts = [0 for _ in range(self.C)]
        self._latent_dim = None

    def update(self, z, labels) -> None:
        """
        Update tracker with latent vectors and labels.
        
        Args:
            z: (B, Z) latent vectors - torch tensor or numpy array
            labels: (B,) - torch tensor or numpy array
        """
        # Convert to torch if needed
        if not isinstance(z, torch.Tensor):
            z = torch.from_numpy(z)
        
        if z.ndim != 2:
            raise ValueError(f"Expected z shape (B,Z), got {tuple(z.shape)}")

        labels = _to_1d_long(labels)
        if labels.shape[0] != z.shape[0]:
            raise ValueError(f"labels len {labels.shape[0]} != batch {z.shape[0]}")

        z = z.detach().float()
        if self.device is None:
            self.device = z.device
        z = z.to(self.device, dtype=torch.float32)
        labels = labels.to(self.device)

        if self._latent_dim is None:
            self._latent_dim = int(z.shape[1])

        for c in range(self.C):
            idx = (labels == c).nonzero(as_tuple=False).reshape(-1)
            if idx.numel() == 0:
                continue
            zc = z.index_select(0, idx)  # (Nc, Z)
            self._append_class(c, zc)

    def snapshot(self) -> Optional[SepMetrics]:
        """
        Compute separation metrics snapshot.
        
        Returns:
            SepMetrics if at least 2 classes have enough samples, else None.
        """
        stats = self._class_stats()
        if stats is None:
            return None
        means, covs, counts = stats

        # Fisher: sum_{i<j} ||μi-μj||^2 / (tr(Σi)+tr(Σj)+eps)
        fisher = 0.0
        pairs = [(0, 1), (0, 2), (1, 2)]
        for i, j in pairs:
            if counts[i] < 2 or counts[j] < 2:
                continue
            d = means[i] - means[j]
            num = float((d @ d).item())
            denom = float(torch.trace(covs[i]).item() + torch.trace(covs[j]).item() + self.eps)
            fisher += num / denom

        # Mahalanobis: sqrt((μi-μj)^T Σp^{-1} (μi-μj))
        md_01 = self._mahalanobis(means[0], means[1], covs[0], covs[1], counts[0], counts[1])
        md_02 = self._mahalanobis(means[0], means[2], covs[0], covs[2], counts[0], counts[2])
        md_12 = self._mahalanobis(means[1], means[2], covs[1], covs[2], counts[1], counts[2])

        return SepMetrics(
            fisher=float(fisher),
            md_01=float(md_01),
            md_02=float(md_02),
            md_12=float(md_12),
            counts=(int(counts[0]), int(counts[1]), int(counts[2])),
        )

    # ============================================================
    #  INTERNALS
    # ============================================================

    def _append_class(self, c: int, zc: torch.Tensor) -> None:
        """Append latents to class buffer, keeping only last `window` rows."""
        if self._buf[c] is None:
            self._buf[c] = zc
        else:
            self._buf[c] = torch.cat([self._buf[c], zc], dim=0)
        if self._buf[c].shape[0] > self.window:
            self._buf[c] = self._buf[c][-self.window :]
        self._counts[c] = int(self._buf[c].shape[0])

    def _class_stats(
        self,
    ) -> Optional[Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...], Tuple[int, ...]]]:
        """Compute per-class mean and covariance."""
        if self._latent_dim is None:
            return None

        means = []
        covs = []
        counts = []
        Z = self._latent_dim
        I = torch.eye(Z, device=self.device, dtype=torch.float32) * self.eps

        have = 0
        for c in range(self.C):
            buf = self._buf[c]
            n = 0 if buf is None else int(buf.shape[0])
            counts.append(n)
            if buf is None or n < 2:
                means.append(torch.zeros(Z, device=self.device, dtype=torch.float32))
                covs.append(I.clone())
                continue

            have += 1
            mu = buf.mean(dim=0)
            xc = buf - mu
            cov = (xc.t() @ xc) / float(n - 1)
            cov = cov + I
            means.append(mu)
            covs.append(cov)

        if have < 2:
            return None

        return tuple(means), tuple(covs), tuple(counts)

    def _mahalanobis(
        self,
        mu_i: torch.Tensor,
        mu_j: torch.Tensor,
        cov_i: torch.Tensor,
        cov_j: torch.Tensor,
        n_i: int,
        n_j: int,
    ) -> float:
        """Compute Mahalanobis distance between two classes."""
        if n_i < 2 or n_j < 2:
            return 0.0
        # pooled covariance (weighted)
        w_i = max(1, n_i - 1)
        w_j = max(1, n_j - 1)
        cov_p = (cov_i * w_i + cov_j * w_j) / float(w_i + w_j)
        d = (mu_i - mu_j).unsqueeze(1)  # (Z,1)
        # solve cov_p^{-1} d without explicit inverse
        try:
            sol = torch.linalg.solve(cov_p, d)  # (Z,1)
            md2 = float((d.t() @ sol).item())
            return float(max(md2, 0.0) ** 0.5)
        except Exception:
            return 0.0

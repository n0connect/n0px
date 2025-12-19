# ml/sep_metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch


def _safe_pinv(a: torch.Tensor) -> torch.Tensor:
    """CPU/float64 stabil pseudo-inverse."""
    return torch.linalg.pinv(a)


@dataclass
class SepConfig:
    labels: Tuple[int, int, int] = (0, 1, 2)  # 0=easy, 1=prime, 2=hard
    report_every: int = 200
    cov_eps: float = 1e-2
    max_per_class: int = 4096  # ring buffer (per class)


class LatentSeparationMonitor:
    """
    z-uzayında separation metrikleri:
      - Fisher = Tr(Sb) / Tr(Sw)
      - Mahalanobis(mean_i, mean_j) with pooled cov + eps*I
    Not: yalnızca min örnek eşiği sağlanınca raporlar.
    """

    def __init__(self, cfg: SepConfig):
        self.cfg = cfg
        self._buf: Dict[int, torch.Tensor] = {c: torch.empty(0) for c in cfg.labels}
        self._dim: Optional[int] = None

    def update(self, z: torch.Tensor, y: torch.Tensor) -> None:
        """z: (B, Z)  y: (B,)"""
        # Move to CPU first, THEN convert dtype (MPS doesn't support float64)
        z = z.detach().to("cpu").to(dtype=torch.float64)
        y = y.detach().to("cpu")

        if z.ndim != 2:
            return
        if self._dim is None:
            self._dim = int(z.shape[1])

        for c in self.cfg.labels:
            idx = (y == c).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            zc = z.index_select(0, idx)

            if self._buf[c].numel() == 0:
                self._buf[c] = zc
            else:
                self._buf[c] = torch.cat([self._buf[c], zc], dim=0)

            # ring buffer
            if self._buf[c].shape[0] > self.cfg.max_per_class:
                self._buf[c] = self._buf[c][-self.cfg.max_per_class :]

    def min_required(self) -> int:
        """Her sınıfta istenen min örnek."""
        zdim = int(self._dim or 0)
        return max(32, zdim + 8)

    def ready(self) -> bool:
        nmin = self.min_required()
        return all(self._buf[c].shape[0] >= nmin for c in self.cfg.labels)

    def _mean_cov(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (N, Z) CPU/float64"""
        mu = x.mean(dim=0, keepdim=True)  # (1,Z)
        xc = x - mu
        # unbiased cov: (Z,Z)
        denom = max(1, x.shape[0] - 1)
        cov = (xc.T @ xc) / float(denom)
        # regularize
        eps = float(self.cfg.cov_eps)
        cov = cov + eps * torch.eye(cov.shape[0], dtype=cov.dtype)
        return mu.squeeze(0), cov

    def compute(self) -> Optional[Dict[str, float]]:
        if not self.ready():
            return None

        # collect
        stats = {}
        mus = {}
        covs = {}

        for c in self.cfg.labels:
            mu_c, cov_c = self._mean_cov(self._buf[c])
            mus[c] = mu_c
            covs[c] = cov_c
            stats[f"n_c{c}"] = float(self._buf[c].shape[0])

        # Fisher: Tr(Sb) / Tr(Sw)
        means = torch.stack([mus[c] for c in self.cfg.labels], dim=0)  # (K,Z)
        grand = means.mean(dim=0, keepdim=True)  # (1,Z)

        # Between-class scatter (trace only)
        sb = ((means - grand) ** 2).sum(dim=1).mean()  # scalar proxy
        # Within-class scatter (trace only)
        sw = torch.stack([torch.trace(covs[c]) for c in self.cfg.labels]).mean()
        fisher = (sb / (sw + 1e-12)).item()
        stats["latent_fisher"] = float(fisher)

        # Pairwise Mahalanobis (pooled)
        def mahal(
            a: torch.Tensor,
            b: torch.Tensor,
            cov_a: torch.Tensor,
            cov_b: torch.Tensor,
        ) -> float:
            d = (a - b).view(-1, 1)  # (Z,1)
            pooled = 0.5 * (cov_a + cov_b)
            inv = _safe_pinv(pooled)
            val = torch.sqrt((d.T @ inv @ d).clamp_min(0.0)).item()
            return float(val)

        c0, c1, c2 = self.cfg.labels
        stats["mahal_01"] = mahal(mus[c0], mus[c1], covs[c0], covs[c1])
        stats["mahal_02"] = mahal(mus[c0], mus[c2], covs[c0], covs[c2])
        stats["mahal_12"] = mahal(mus[c1], mus[c2], covs[c1], covs[c2])

        return stats


class ClasswiseReconMonitor:
    """
    Class-wise reconstruction quality metrics:
      - Reconstruction MSE (recon_mse)
      - Loglik + NLL (both shown for clarity)
      - KL divergence (now truly per-class)
      - Latent stats: ||z|| norm, covariance trace
    """

    def __init__(self, labels: Tuple[int, int, int] = (0, 1, 2)):
        self.labels = labels

    @torch.no_grad()
    def compute(
        self, model, x: torch.Tensor, y: torch.Tensor, step: int, z: Optional[torch.Tensor] = None,
        model_out: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive per-class metrics.
        
        Args:
            model: VAE model with compute_loss() and forward()
            x: Input batch (B, D)
            y: Labels (B,)
            step: Training step
            z: Optional pre-computed latents (B, Z). If None, extracted from model.forward()
            model_out: Optional pre-computed model output dict. If provided, uses pi_logits + mu/logvar from here.
        """
        out: Dict[str, float] = {}
        
        # Extract latents if not provided
        if z is None:
            if model_out is None:
                model_out = model.forward(x)
            z = model_out.get("z", None)
        
        # Pre-compute pi_logits and mu/logvar if model_out provided (avoid per-class forward)
        pi_logits_full = None
        mu_full = None
        logvar_full = None
        
        if model_out is not None:
            pi_logits_full = model_out.get("pi_logits", None)
            mu_full = model_out.get("mu", None)
            logvar_full = model_out.get("logvar", None)
        
        for c in self.labels:
            idx = (y == c).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            
            xc = x.index_select(0, idx)
            
            # 1) Reconstruction loss (NLL, loglik, and MSE)
            _, m = model.compute_loss(xc, step=step, return_per_sample=True)
            nll = float(m.get("nll", 0.0))
            loglik = -nll  # loglik = -NLL (for clarity)
            
            # Per-class reconstruction MSE (for compression analysis)
            # MSE must be in same space as input: [0,1] normalized
            recon_mse = 0.0
            try:
                # Use pre-computed pi_logits if available (avoid forward pass per class)
                if pi_logits_full is not None:
                    pi_logits_c = pi_logits_full.index_select(0, idx)
                    x_recon_prob = torch.sigmoid(pi_logits_c)
                    recon_mse = float(torch.mean((xc - x_recon_prob) ** 2).item())
                else:
                    # Fallback: compute forward per class (original logic)
                    model_out_c = model.forward(xc)
                    if "pi_logits" in model_out_c:
                        pi_logits = model_out_c["pi_logits"]
                        x_recon_prob = torch.sigmoid(pi_logits)
                        recon_mse = float(torch.mean((xc - x_recon_prob) ** 2).item())
            except Exception as e:
                recon_mse = 0.0
            
            out[f"nll_c{c}"] = nll
            out[f"loglik_c{c}"] = loglik
            out[f"recon_mse_c{c}"] = recon_mse
            out[f"kl_c{c}"] = float(m.get("kl", 0.0))
            out[f"n_c{c}"] = float(idx.numel())
            
            # 2) Latent stats (if z available)
            if z is not None:
                zc = z.index_select(0, idx).detach().cpu()
                z_norm = float(torch.norm(zc, dim=1).mean().item())
                z_cov = torch.cov(zc.T)
                z_cov_trace = float(z_cov.trace().item())
                
                # Anisotropy detection: var_max / (var_min + eps)
                # If condition number >> 1, latent has "collapsed" modes
                z_var = torch.diag(z_cov)  # eigenvalues on diagonal (approx)
                z_var_min = float(z_var.min().item())
                z_var_max = float(z_var.max().item())
                z_cond_proxy = z_var_max / (z_var_min + 1e-8)
                
                out[f"z_norm_c{c}"] = z_norm
                out[f"z_cov_trace_c{c}"] = z_cov_trace
                out[f"z_var_min_c{c}"] = z_var_min
                out[f"z_var_max_c{c}"] = z_var_max
                out[f"z_cond_proxy_c{c}"] = z_cond_proxy  # Anisotropy metric
        
        # ΔNLL'ler (varsa)
        def get(key: str) -> Optional[float]:
            return out.get(key, None)

        if get("nll_c1") is not None and get("nll_c2") is not None:
            out["dnll_12"] = float(out["nll_c1"] - out["nll_c2"])
        if get("nll_c1") is not None and get("nll_c0") is not None:
            out["dnll_10"] = float(out["nll_c1"] - out["nll_c0"])
        if get("nll_c2") is not None and get("nll_c0") is not None:
            out["dnll_20"] = float(out["nll_c2"] - out["nll_c0"])
        
        return out

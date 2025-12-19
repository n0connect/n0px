# -*- coding: utf-8 -*-
"""
RealNoiseMixtureVAE (ℝ-uzayı)
- Input: x ∈ R^D
- Decoder: her boyut için π_i logits üretir (bit=1 olasılığı)
- Likelihood: π N(1,σ^2) + (1-π) N(0,σ^2)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class VAEConfig:
    noise_sigma: float
    num_layers: int
    hidden_dimension: int
    dropout: float

    # KL schedule
    beta_max: float = 1.0
    beta_warmup_steps: int = 20_000  # <-- daha gerçekçi (senin batch sayın yüksek)
    free_bits: float = 0.05          # <-- KL collapse'a karşı "free-bits" (dim başına)

    activation: str = "silu"
    layernorm: bool = True
    expected_input_dim: Optional[int] = None

    @staticmethod
    def from_cfg(cfg: Dict[str, Any]) -> "VAEConfig":
        sigma = float(cfg["crypto"]["noise_sigma"])
        model = cfg["model"]
        # Optional overrides (config.json'a ekleyebilirsin)
        vae = cfg.get("vae", {})
        return VAEConfig(
            noise_sigma=sigma,
            num_layers=int(model["num_layers"]),
            hidden_dimension=int(model["hidden_dimension"]),
            dropout=float(model["dropout"]),
            expected_input_dim=int(model.get("input_dimension")) if "input_dimension" in model else None,
            beta_max=float(vae.get("beta_max", 1.0)),
            beta_warmup_steps=int(vae.get("beta_warmup_steps", 20_000)),
            free_bits=float(vae.get("free_bits", 0.05)),
            activation=str(vae.get("activation", "silu")),
            layernorm=bool(vae.get("layernorm", True)),
        )


def _get_activation(name: str) -> nn.Module:
    n = name.strip().lower()
    if n == "silu":
        return nn.SiLU()
    if n == "gelu":
        return nn.GELU()
    if n == "softplus":
        return nn.Softplus()
    raise ValueError(f"Unsupported activation: {name}")


def _beta_schedule(step: int, beta_max: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return float(beta_max)
    s = max(0, int(step))
    return float(beta_max) * min(1.0, s / float(warmup_steps))


def _gaussian_logpdf(x: torch.Tensor, mean: float, sigma: float) -> torch.Tensor:
    var = sigma * sigma
    return -0.5 * ((x - mean) ** 2) / var - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)


def _mixture_nll_bits_per_sample(x: torch.Tensor, pi_logits: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    returns: (B,)  per-sample NLL (sum over D)
    """
    log_pi = -F.softplus(-pi_logits)
    log_1m_pi = -F.softplus(pi_logits)

    logN1 = _gaussian_logpdf(x, mean=1.0, sigma=sigma)
    logN0 = _gaussian_logpdf(x, mean=0.0, sigma=sigma)

    log_mix = torch.logsumexp(
        torch.stack([log_pi + logN1, log_1m_pi + logN0], dim=0),
        dim=0
    )
    return (-log_mix.sum(dim=1))  # (B,)


def _kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # (B,Z) KL per dim
    return -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())


def _compute_latent_dim(input_dim: int, depth: int) -> int:
    d = max(1, int(input_dim))
    L = max(1, int(depth))
    return max(1, d // (L + 1))


def _make_widths(input_dim: int, latent_dim: int, depth: int, hidden_cap: int) -> Tuple[int, ...]:
    D = max(1, int(input_dim))
    Z = max(1, int(latent_dim))
    L = max(1, int(depth))
    cap = max(1, int(hidden_cap))

    widths = []
    for i in range(1, L + 1):
        t = i / float(L + 1)
        w = int(round(D * (1.0 - t) + Z * t))
        w = max(1, min(cap, w))
        widths.append(w)

    cleaned = []
    for w in widths:
        if not cleaned or w != cleaned[-1]:
            cleaned.append(w)

    if not cleaned or cleaned[-1] != Z:
        cleaned.append(Z)

    return tuple(cleaned)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, widths: Tuple[int, ...], dropout: float, act: nn.Module, use_ln: bool):
        super().__init__()
        layers = []
        prev = in_dim
        for w in widths:
            layers.append(nn.Linear(prev, w))
            if use_ln:
                layers.append(nn.LayerNorm(w))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = w
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RealNoiseMixtureVAE(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = VAEConfig.from_cfg(cfg)
        self._built = False
        self._input_dim: Optional[int] = None
        self._latent_dim: Optional[int] = None

        self.encoder: Optional[nn.Module] = None
        self.decoder: Optional[nn.Module] = None
        self._enc_head: Optional[nn.Linear] = None
        self._dec_head: Optional[nn.Linear] = None

        self._act = _get_activation(self.cfg.activation)

    def _build(self, input_dim: int) -> None:
        if self.cfg.expected_input_dim is not None and int(self.cfg.expected_input_dim) != int(input_dim):
            raise ValueError(
                f"Input dimension mismatch: runtime D={input_dim} vs cfg.model.input_dimension={self.cfg.expected_input_dim}"
            )

        latent_dim = _compute_latent_dim(input_dim, self.cfg.num_layers)
        enc_widths = _make_widths(input_dim, latent_dim, self.cfg.num_layers, self.cfg.hidden_dimension)
        dec_widths = tuple(reversed(enc_widths))[:-1]

        self.encoder = _MLP(
            in_dim=input_dim,
            widths=enc_widths[:-1],
            dropout=self.cfg.dropout,
            act=self._act,
            use_ln=self.cfg.layernorm,
        )
        last_enc = enc_widths[-2] if len(enc_widths) >= 2 else input_dim
        self._enc_head = nn.Linear(last_enc, 2 * latent_dim)

        self.decoder = _MLP(
            in_dim=latent_dim,
            widths=dec_widths,
            dropout=self.cfg.dropout,
            act=self._act,
            use_ln=self.cfg.layernorm,
        )
        last_dec = dec_widths[-1] if len(dec_widths) > 0 else latent_dim
        self._dec_head = nn.Linear(last_dec, input_dim)

        self._input_dim = int(input_dim)
        self._latent_dim = int(latent_dim)
        self._built = True

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError(f"Expected x shape (B,D), got {tuple(x.shape)}")
        if not self._built:
            self._build(int(x.shape[1]))

        assert self.encoder is not None
        assert self.decoder is not None
        assert self._enc_head is not None
        assert self._dec_head is not None
        assert self._latent_dim is not None

        h = self.encoder(x)
        stats = self._enc_head(h)
        mu, logvar = stats[:, : self._latent_dim], stats[:, self._latent_dim :]
        z = self._reparameterize(mu, logvar)
        d = self.decoder(z)
        pi_logits = self._dec_head(d)
        return {"pi_logits": pi_logits, "mu": mu, "logvar": logvar, "z": z}

    def compute_loss(
        self,
        x: torch.Tensor,
        step: int,
        return_per_sample: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Union[float, torch.Tensor]]]:
        out = self.forward(x)

        nll_ps = _mixture_nll_bits_per_sample(x, out["pi_logits"], sigma=self.cfg.noise_sigma)  # (B,)
        nll = nll_ps.mean()

        kl_dim = _kl_per_dim(out["mu"], out["logvar"])  # (B,Z)

        # FREE-BITS: KL küçülürken “daha da küçült” baskısını kır
        fb = float(self.cfg.free_bits)
        if fb > 0:
            kl_dim_fb = torch.clamp(kl_dim, min=fb)  # clamp => alt bölgede gradient = 0
        else:
            kl_dim_fb = kl_dim

        kl_ps = kl_dim_fb.sum(dim=1)  # (B,)
        kl = kl_ps.mean()

        beta = _beta_schedule(step, beta_max=self.cfg.beta_max, warmup_steps=self.cfg.beta_warmup_steps)
        loss = nll + beta * kl

        metrics: Dict[str, Union[float, torch.Tensor]] = {
            "loss": float(loss.detach().cpu().item()),
            "nll": float(nll.detach().cpu().item()),
            "kl": float(kl.detach().cpu().item()),
            "beta": float(beta),
            "input_dim": float(x.shape[1]),
            "latent_dim": float(self._latent_dim or 0),
        }
        if return_per_sample:
            metrics["nll_ps"] = nll_ps.detach()
            metrics["kl_ps"] = kl_ps.detach()
        return loss, metrics

    def save(self, path: str) -> None:
        if not self._built:
            raise RuntimeError("Model not built yet. Run one forward pass before saving.")
        torch.save({"state_dict": self.state_dict(), "cfg": self.cfg.__dict__}, path)

    @classmethod
    def load(cls, cfg: Dict[str, Any], path: str, map_location: Optional[str] = None) -> "RealNoiseMixtureVAE":
        obj = cls(cfg)
        payload = torch.load(path, map_location=map_location)
        obj.load_state_dict(payload["state_dict"])
        return obj

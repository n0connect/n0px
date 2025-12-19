# ml/models/_mixture_vae_base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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

    beta_max: float = 1.0
    beta_warmup_steps: int = 20000
    activation: str = "silu"   # silu | gelu | softplus
    layernorm: bool = True
    expected_input_dim: Optional[int] = None

    @staticmethod
    def from_cfg(cfg: Dict[str, Any]) -> "VAEConfig":
        sigma = float(cfg["crypto"]["noise_sigma"])
        m = cfg["model"]
        return VAEConfig(
            noise_sigma=sigma,
            num_layers=int(m["num_layers"]),
            hidden_dimension=int(m["hidden_dimension"]),
            dropout=float(m["dropout"]),
            beta_max=float(m.get("beta_max", 1.0)),
            beta_warmup_steps=int(m.get("beta_warmup_steps", 20000)),
            activation=str(m.get("activation", "silu")),
            layernorm=bool(m.get("layernorm", True)),
            expected_input_dim=int(m["input_dimension"]) if "input_dimension" in m else None,
        )

def _get_act(name: str) -> nn.Module:
    n = name.strip().lower()
    if n == "silu":
        return nn.SiLU()
    if n == "gelu":
        return nn.GELU()
    if n == "softplus":
        return nn.Softplus()
    raise ValueError(f"Unsupported activation: {name}")

def beta_schedule(step: int, beta_max: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return float(beta_max)
    s = max(0, int(step))
    return float(beta_max) * min(1.0, s / float(warmup_steps))

def gaussian_logpdf(x: torch.Tensor, mean: float, sigma: float) -> torch.Tensor:
    var = sigma * sigma
    return -0.5 * ((x - mean) ** 2) / var - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)

def mixture_nll_bits(x: torch.Tensor, pi_logits: torch.Tensor, sigma: float) -> torch.Tensor:
    # Not: sigma çok küçükse logpdf pozitif olabilir; bu yüzden “NLL” negatif görünebilir.
    log_pi = -F.softplus(-pi_logits)
    log_1m_pi = -F.softplus(pi_logits)

    logN1 = gaussian_logpdf(x, mean=1.0, sigma=sigma)
    logN0 = gaussian_logpdf(x, mean=0.0, sigma=sigma)

    log_mix = torch.logsumexp(torch.stack([log_pi + logN1, log_1m_pi + logN0], dim=0), dim=0)
    return (-log_mix.sum(dim=1)).mean()

def kl_std_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()

def compute_latent_dim(input_dim: int, depth: int) -> int:
    d = max(1, int(input_dim))
    L = max(1, int(depth))
    return max(2, d // (L + 1))

def make_widths(input_dim: int, latent_dim: int, depth: int, cap: int) -> Tuple[int, ...]:
    D = max(1, int(input_dim))
    Z = max(2, int(latent_dim))
    L = max(1, int(depth))
    cap = max(2, int(cap))

    widths = []
    for i in range(1, L + 1):
        t = i / float(L + 1)
        w = int(round(D * (1.0 - t) + Z * t))
        w = max(2, min(cap, w))
        widths.append(w)

    cleaned = []
    for w in widths:
        if not cleaned or w != cleaned[-1]:
            cleaned.append(w)

    if not cleaned or cleaned[-1] != Z:
        cleaned.append(Z)

    return tuple(cleaned)

class MLP(nn.Module):
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

class RealMixtureVAE(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = VAEConfig.from_cfg(cfg)
        self._built = False
        self._input_dim: Optional[int] = None
        self._latent_dim: Optional[int] = None

        self._act = _get_act(self.cfg.activation)

        self.encoder: Optional[nn.Module] = None
        self.decoder: Optional[nn.Module] = None
        self.enc_head: Optional[nn.Linear] = None
        self.dec_head: Optional[nn.Linear] = None

    def _build(self, D: int) -> None:
        if self.cfg.expected_input_dim is not None and int(self.cfg.expected_input_dim) != int(D):
            raise ValueError(f"Input dim mismatch runtime D={D} vs cfg={self.cfg.expected_input_dim}")

        Z = compute_latent_dim(D, self.cfg.num_layers)
        enc_widths = make_widths(D, Z, self.cfg.num_layers, self.cfg.hidden_dimension)
        dec_widths = tuple(reversed(enc_widths))[:-1]

        trunk_widths = enc_widths[:-1]
        last_enc = trunk_widths[-1] if trunk_widths else D

        self.encoder = MLP(D, trunk_widths, self.cfg.dropout, self._act, self.cfg.layernorm)
        self.enc_head = nn.Linear(last_enc, 2 * Z)

        self.decoder = MLP(Z, dec_widths, self.cfg.dropout, self._act, self.cfg.layernorm)
        last_dec = dec_widths[-1] if dec_widths else Z
        self.dec_head = nn.Linear(last_dec, D)

        self._input_dim = D
        self._latent_dim = Z
        self._built = True

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._built:
            self._build(int(x.shape[1]))
        assert self.encoder and self.enc_head and self._latent_dim is not None
        h = self.encoder(x)
        stats = self.enc_head(h)
        mu = stats[:, : self._latent_dim]
        logvar = stats[:, self._latent_dim :]
        z = self.reparam(mu, logvar)
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        assert self.decoder and self.dec_head
        h = self.decoder(z)
        return self.dec_head(h)  # pi_logits

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z, mu, logvar = self.encode(x)
        pi_logits = self.decode(z)
        return {"z": z, "mu": mu, "logvar": logvar, "pi_logits": pi_logits}

    def compute_loss(self, x: torch.Tensor, step: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.forward(x)
        nll = mixture_nll_bits(x, out["pi_logits"], sigma=self.cfg.noise_sigma)
        kl = kl_std_normal(out["mu"], out["logvar"])
        beta = beta_schedule(step, self.cfg.beta_max, self.cfg.beta_warmup_steps)
        loss = nll + beta * kl
        # Return tensors directly - let caller extract on CPU if needed
        # This avoids GPU sync stalls in hot training loop
        return loss, {
            "loss": loss.detach(),  # Return tensor, not float
            "nll": nll.detach(),    # Return tensor, not float
            "kl": kl.detach(),      # Return tensor, not float
            "beta": beta,           # Python float is OK (scalar)
            "latent_dim": float(self._latent_dim or 0),
        }

    def save(self, path: str) -> None:
        if not self._built:
            raise RuntimeError("Model not built yet.")
        torch.save({"state_dict": self.state_dict(), "cfg": self.cfg.__dict__}, path)

class ComplexLinear(nn.Module):
    """Complex linear using real tensors: input (B,2in) -> output (B,2out)."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_f = in_features
        self.out_f = out_features
        # W = A + iB, b = c + id
        self.A = nn.Linear(in_features, out_features, bias=True)
        self.B = nn.Linear(in_features, out_features, bias=False)
        self.c = nn.Parameter(torch.zeros(out_features))
        self.d = nn.Parameter(torch.zeros(out_features))

    def forward(self, x2: torch.Tensor) -> torch.Tensor:
        xr, xi = x2[:, : self.in_f], x2[:, self.in_f :]
        yr = self.A(xr) - self.B(xi) + self.c
        yi = self.A(xi) + self.B(xr) + self.d
        return torch.cat([yr, yi], dim=1)

class ModActivation(nn.Module):
    """Magnitude activation without mentioning forbidden stuff."""
    def __init__(self, act: nn.Module, eps: float = 1e-6):
        super().__init__()
        self.act = act
        self.eps = eps

    def forward(self, x2: torch.Tensor) -> torch.Tensor:
        half = x2.shape[1] // 2
        xr, xi = x2[:, :half], x2[:, half:]
        r = torch.sqrt(xr * xr + xi * xi + self.eps)
        scale = self.act(r) / (r + self.eps)
        return torch.cat([xr * scale, xi * scale], dim=1)

class ComplexMixtureVAE(nn.Module):
    """
    Complex latent: z = z_r + i z_i represented as (B,2Z).
    q(z|x) factorized over real/imag.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = VAEConfig.from_cfg(cfg)
        self._built = False
        self._input_dim: Optional[int] = None
        self._latent_dim: Optional[int] = None

        act = _get_act(self.cfg.activation)
        self._modact = ModActivation(act)

        self.enc_layers: nn.ModuleList = nn.ModuleList()
        self.dec_layers: nn.ModuleList = nn.ModuleList()
        self.enc_head: Optional[nn.Linear] = None
        self.dec_head: Optional[nn.Linear] = None

    def _build(self, D: int) -> None:
        if self.cfg.expected_input_dim is not None and int(self.cfg.expected_input_dim) != int(D):
            raise ValueError(f"Input dim mismatch runtime D={D} vs cfg={self.cfg.expected_input_dim}")

        Z = compute_latent_dim(D, self.cfg.num_layers)
        widths = make_widths(D, Z, self.cfg.num_layers, self.cfg.hidden_dimension)

        # Encode: real x -> complex trunk (2*width)
        prev = D
        for w in widths[:-1]:
            self.enc_layers.append(ComplexLinear(prev, w))
            self.enc_layers.append(self._modact)
            if self.cfg.dropout > 0:
                self.enc_layers.append(nn.Dropout(self.cfg.dropout))
            prev = w

        last = widths[-2] if len(widths) >= 2 else D
        self.enc_head = nn.Linear(2 * last, 4 * Z)  # mu_r, mu_i, logvar_r, logvar_i

        # Decode: complex z -> complex trunk -> pi_logits (real)
        rev = list(reversed(widths))[:-1]
        prevz = Z
        for w in rev:
            self.dec_layers.append(ComplexLinear(prevz, w))
            self.dec_layers.append(self._modact)
            if self.cfg.dropout > 0:
                self.dec_layers.append(nn.Dropout(self.cfg.dropout))
            prevz = w

        last_dec = rev[-1] if len(rev) else Z
        self.dec_head = nn.Linear(2 * last_dec, D)

        self._input_dim = D
        self._latent_dim = Z
        self._built = True

    @staticmethod
    def _reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._built:
            self._build(int(x.shape[1]))
        assert self.enc_head and self._latent_dim is not None

        h = x
        for layer in self.enc_layers:
            h = layer(h if isinstance(layer, nn.Dropout) else h)  # keep explicit
        # h is still real; first ComplexLinear expects (B,2in). We need lift at start.
        # Fix: If enc_layers exists, first is ComplexLinear expecting 2*D input.
        # So lift x -> (x,0) before loop:
        # We'll handle below by reconstructing properly.
        raise RuntimeError("encode() should be overridden by wrapper class (see cm0/cm1 model files).")

    def _encode_impl(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self._built:
            self._build(int(x.shape[1]))
        assert self.enc_head and self._latent_dim is not None

        # lift real x -> complex (x + i0)
        h = torch.cat([x, torch.zeros_like(x)], dim=1)  # (B,2D)
        for layer in self.enc_layers:
            h = layer(h)

        stats = self.enc_head(h)  # (B,4Z)
        Z = self._latent_dim
        mu_r = stats[:, 0:Z]
        mu_i = stats[:, Z:2 * Z]
        lv_r = stats[:, 2 * Z:3 * Z]
        lv_i = stats[:, 3 * Z:4 * Z]

        z_r = self._reparam(mu_r, lv_r)
        z_i = self._reparam(mu_i, lv_i)
        z2 = torch.cat([z_r, z_i], dim=1)
        mu2 = torch.cat([mu_r, mu_i], dim=1)
        lv2 = torch.cat([lv_r, lv_i], dim=1)
        return {"z2": z2, "mu2": mu2, "logvar2": lv2}

    def _decode_impl(self, z2: torch.Tensor) -> torch.Tensor:
        assert self.dec_head and self._latent_dim is not None
        h = z2  # (B,2Z)
        # first ComplexLinear expects 2*Z input, OK
        for layer in self.dec_layers:
            h = layer(h)
        pi_logits = self.dec_head(h)
        return pi_logits

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = self._encode_impl(x)
        pi_logits = self._decode_impl(enc["z2"])
        return {"z": enc["z2"], "mu": enc["mu2"], "logvar": enc["logvar2"], "pi_logits": pi_logits}

    def compute_loss(self, x: torch.Tensor, step: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.forward(x)
        nll = mixture_nll_bits(x, out["pi_logits"], sigma=self.cfg.noise_sigma)
        kl = kl_std_normal(out["mu"], out["logvar"])  # works on concatenated real+imag
        beta = beta_schedule(step, self.cfg.beta_max, self.cfg.beta_warmup_steps)
        loss = nll + beta * kl
        # Return tensors directly - let caller extract on CPU if needed
        return loss, {
            "loss": loss.detach(),   # Return tensor, not float
            "nll": nll.detach(),     # Return tensor, not float
            "kl": kl.detach(),       # Return tensor, not float
            "beta": float(beta),
            "latent_dim": float(self._latent_dim or 0),
        }

    def save(self, path: str) -> None:
        if not self._built:
            raise RuntimeError("Model not built yet.")
        torch.save({"state_dict": self.state_dict(), "cfg": self.cfg.__dict__}, path)

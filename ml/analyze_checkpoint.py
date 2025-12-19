#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Checkpoint Analyzer (n0px)
============================

Amaç:
- Kaydedilmiş modeli (RM0/RM1/CM0/CM1) yükle
- Go Bridge üzerinden N batch örnekle
- Per-class compression metrikleri:
    LL, NLL, KL, Recon_MSE
- Latent istatistikleri:
    z_norm, z_cov_trace, z_cond_proxy (anisotropy/collapse alarmı)
- Separation metrikleri (ready olunca):
    Fisher, Mahalanobis (01/02/12)
- DP fingerprint (batch-level):
    mean/std, mid% (0.1-0.9), entropy_proxy=E[x(1-x)]

Çalıştırma:
  python3 -m ml.analyze_checkpoint \
    --checkpoint checkpoints/checkpoint_epoch_010.pt \
    --model real \
    --batches 2000 --batch-size 64 \
    --device auto \
    --out reports/rm0_analysis.json
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# --------------------- Imports (local package safe) ---------------------

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent))

try:
    from core import create_and_connect_bridge
    from data import StreamDataIterator
except ImportError:
    from ml.core import create_and_connect_bridge
    from ml.data import StreamDataIterator

# Models (real / complex)
try:
    from models.real_noise_mixture_vae import RealNoiseMixtureVAE
except ImportError:
    from ml.models.real_noise_mixture_vae import RealNoiseMixtureVAE

try:
    from models.complex_noise_mixture_vae import ComplexNoiseMixtureVAE
except Exception:
    ComplexNoiseMixtureVAE = None  # optional


# --------------------- Utils ---------------------

def resolve_device(name: str) -> torch.device:
    n = (name or "auto").lower().strip()
    if n == "auto":
        # Apple MPS > CUDA > CPU (sen macOS’ta olduğun için)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if n in ("cpu", "mps", "cuda"):
        if n == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS requested but not available")
        if n == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available")
        return torch.device(n)
    raise ValueError(f"Unsupported device: {name}")


def to_cpu64(x: torch.Tensor) -> torch.Tensor:
    """Convert to CPU. Use float32 on MPS (not supported float64), else float64."""
    device = x.device
    if device.type == "mps":
        # MPS doesn't support float64, use float32
        return x.detach().to("cpu", dtype=torch.float32)
    else:
        # CPU and CUDA can use float64
        return x.detach().to("cpu", dtype=torch.float64)


def safe_pinv(a: torch.Tensor) -> torch.Tensor:
    # CPU/float64 pinv
    return torch.linalg.pinv(a)


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # mean over batch
    return (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()


def gaussian_logpdf(x: torch.Tensor, mean: float, sigma: float) -> torch.Tensor:
    var = sigma * sigma
    return -0.5 * ((x - mean) ** 2) / var - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)


def mixture_loglik_bits(x: torch.Tensor, pi_logits: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    returns: loglik per sample (B,), summed over dims
    """
    # log π and log(1-π)
    log_pi = -F.softplus(-pi_logits)
    log_1m_pi = -F.softplus(pi_logits)

    logN1 = gaussian_logpdf(x, mean=1.0, sigma=sigma)
    logN0 = gaussian_logpdf(x, mean=0.0, sigma=sigma)

    log_mix = torch.logsumexp(
        torch.stack([log_pi + logN1, log_1m_pi + logN0], dim=0),
        dim=0
    )
    return log_mix.sum(dim=1)


def recon_mean_from_output(out: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Recon mean (expected x) için:
    - RealNoiseMixtureVAE: pi_logits -> sigmoid(pi_logits) = E[x]
    - Eğer model farklı isimle dönüyorsa: x_hat / recon vb.
    """
    if "pi_logits" in out:
        return torch.sigmoid(out["pi_logits"])
    for key in ("x_hat", "recon", "reconstruction"):
        if key in out:
            return out[key]
    return None


def extract_latent(out: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Latent z’yi yakala:
    - real: out["z"] float
    - complex: out["z"] complex -> view_as_real -> flatten
    """
    z = out.get("z", None)
    if z is None:
        return None
    if torch.is_complex(z):
        zr = torch.view_as_real(z)  # (...,2)
        return zr.reshape(zr.shape[0], -1)
    return z


def dp_fingerprint(x: torch.Tensor) -> Dict[str, float]:
    """
    DP fingerprint (batch-level):
    - mean/std
    - mid%: 0.1<x<0.9
    - entropy_proxy: E[x(1-x)]
    """
    # x: (B,D) float
    xf = x.detach()
    mean = float(xf.mean().item())
    std = float(xf.std(unbiased=False).item())
    mid = float(((xf > 0.1) & (xf < 0.9)).float().mean().item())
    ent = float((xf * (1.0 - xf)).mean().item())
    return {"x_mean": mean, "x_std": std, "x_mid_0p1_0p9": mid, "x_entropy_proxy": ent}


# --------------------- Metrics accumulators ---------------------

@dataclass
class ClassAgg:
    n: int = 0
    ll_sum: float = 0.0
    nll_sum: float = 0.0
    kl_sum: float = 0.0
    mse_sum: float = 0.0

    z_norm_sum: float = 0.0
    z_cov_trace_sum: float = 0.0
    z_cond_proxy_sum: float = 0.0

    def add(self, n: int, ll: float, nll: float, kl: float, mse: float,
            z_norm: float, z_cov_trace: float, z_cond_proxy: float) -> None:
        self.n += int(n)
        self.ll_sum += float(ll) * n
        self.nll_sum += float(nll) * n
        self.kl_sum += float(kl) * n
        self.mse_sum += float(mse) * n
        self.z_norm_sum += float(z_norm) * n
        self.z_cov_trace_sum += float(z_cov_trace) * n
        self.z_cond_proxy_sum += float(z_cond_proxy) * n

    def mean(self) -> Dict[str, float]:
        if self.n <= 0:
            return {"n": 0}
        inv = 1.0 / float(self.n)
        return {
            "n": int(self.n),
            "ll": self.ll_sum * inv,
            "nll": self.nll_sum * inv,
            "kl": self.kl_sum * inv,
            "recon_mse": self.mse_sum * inv,
            "z_norm": self.z_norm_sum * inv,
            "z_cov_trace": self.z_cov_trace_sum * inv,
            "z_cond_proxy": self.z_cond_proxy_sum * inv,
        }


class LatentBuffer:
    """
    Separation için per-class latent buffer (CPU/float64).
    """
    def __init__(self, labels: Tuple[int, int, int], max_per_class: int = 4096):
        self.labels = labels
        self.max = int(max_per_class)
        self.buf: Dict[int, torch.Tensor] = {c: torch.empty((0, 1), dtype=torch.float64) for c in labels}
        self.zdim: Optional[int] = None

    def update(self, z: torch.Tensor, y: torch.Tensor) -> None:
        z = to_cpu64(z)
        y = y.detach().to("cpu")
        if z.ndim != 2:
            return
        if self.zdim is None:
            self.zdim = int(z.shape[1])
            for c in self.labels:
                self.buf[c] = torch.empty((0, self.zdim), dtype=torch.float64)

        for c in self.labels:
            idx = (y == c).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            zc = z.index_select(0, idx)
            self.buf[c] = torch.cat([self.buf[c], zc], dim=0)
            if self.buf[c].shape[0] > self.max:
                self.buf[c] = self.buf[c][-self.max:]

    def min_required(self) -> int:
        zdim = int(self.zdim or 0)
        return max(32, zdim + 8)

    def ready(self) -> bool:
        nmin = self.min_required()
        return all(self.buf[c].shape[0] >= nmin for c in self.labels)

    def compute_sep(self, cov_eps: float = 1e-2) -> Optional[Dict[str, float]]:
        if not self.ready():
            return None

        labels = self.labels
        mus: Dict[int, torch.Tensor] = {}
        covs: Dict[int, torch.Tensor] = {}

        for c in labels:
            x = self.buf[c]
            mu = x.mean(dim=0, keepdim=True)
            xc = x - mu
            denom = max(1, x.shape[0] - 1)
            cov = (xc.T @ xc) / float(denom)
            cov = cov + float(cov_eps) * torch.eye(cov.shape[0], dtype=cov.dtype)
            mus[c] = mu.squeeze(0)
            covs[c] = cov

        # Fisher proxy
        means = torch.stack([mus[c] for c in labels], dim=0)
        grand = means.mean(dim=0, keepdim=True)
        sb = ((means - grand) ** 2).sum(dim=1).mean()
        sw = torch.stack([torch.trace(covs[c]) for c in labels]).mean()
        fisher = float((sb / (sw + 1e-12)).item())

        def mahal(a, b, ca, cb) -> float:
            d = (a - b).view(-1, 1)
            pooled = 0.5 * (ca + cb)
            inv = safe_pinv(pooled)
            v = torch.sqrt((d.T @ inv @ d).clamp_min(0.0)).item()
            return float(v)

        c0, c1, c2 = labels
        return {
            "latent_fisher": fisher,
            "mahal_01": mahal(mus[c0], mus[c1], covs[c0], covs[c1]),
            "mahal_02": mahal(mus[c0], mus[c2], covs[c0], covs[c2]),
            "mahal_12": mahal(mus[c1], mus[c2], covs[c1], covs[c2]),
            "min_required": int(self.min_required()),
            "n_c0": int(self.buf[c0].shape[0]),
            "n_c1": int(self.buf[c1].shape[0]),
            "n_c2": int(self.buf[c2].shape[0]),
        }


def latent_stats(z: torch.Tensor, eps: float = 1e-12) -> Tuple[float, float, float]:
    """
    z_norm: mean ||z||
    z_cov_trace: trace(cov(z))
    z_cond_proxy: var_max / (var_min + eps)
    """
    zc = to_cpu64(z)
    mu = zc.mean(dim=0, keepdim=True)
    xc = zc - mu
    denom = max(1, zc.shape[0] - 1)
    cov = (xc.T @ xc) / float(denom)

    z_norm = float(torch.linalg.vector_norm(zc, dim=1).mean().item())
    z_cov_trace = float(torch.trace(cov).item())

    var = torch.diag(cov).clamp_min(0.0)
    vmin = float(var.min().item())
    vmax = float(var.max().item())
    z_cond = float(vmax / (vmin + eps))
    return z_norm, z_cov_trace, z_cond


# --------------------- Main ---------------------

_STOP = False
def _sigint_handler(signum, frame):
    global _STOP
    _STOP = True

def load_cfg(config_path: str, checkpoint_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # Önce config/config.json, yoksa checkpoint içinden cfg
    p = Path(config_path)
    if p.exists():
        return json.loads(p.read_text())
    if checkpoint_payload and "cfg" in checkpoint_payload:
        # checkpoint cfg: dataclass __dict__ olabilir; ama model için orijinal config gerekebilir
        # Yine de fallback olarak kullan.
        return {"model": {}, "crypto": {"noise_sigma": float(checkpoint_payload["cfg"].get("noise_sigma", 0.05))}}
    raise FileNotFoundError(f"Config not found: {config_path}")

def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load checkpoint from path. Supports relative and absolute paths."""
    checkpoint_path = Path(path)
    
    # If not absolute, make it relative to project root (parent of ml/)
    if not checkpoint_path.is_absolute():
        project_root = Path(__file__).parent.parent
        checkpoint_path = project_root / path
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Full path checked: {checkpoint_path}\n"
            f"Available checkpoints:\n" +
            "\n".join(str(p) for p in (Path(__file__).parent.parent / "checkpoints_rm0").glob("*.pt")[:5]) if (Path(__file__).parent.parent / "checkpoints_rm0").exists() else "  (no checkpoints directory)"
        )
    
    return torch.load(checkpoint_path, map_location="cpu")

def instantiate_model(kind: str, cfg: Dict[str, Any]):
    k = kind.lower().strip()
    if k in ("real", "rm", "rm0", "rm1"):
        return RealNoiseMixtureVAE(cfg)
    if k in ("complex", "cm", "cm0", "cm1"):
        if ComplexNoiseMixtureVAE is None:
            raise RuntimeError("ComplexNoiseMixtureVAE not importable in this environment.")
        return ComplexNoiseMixtureVAE(cfg)
    raise ValueError(f"Unknown model kind: {kind}")

def main():
    parser = argparse.ArgumentParser(description="Analyze a saved checkpoint on live stream data.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--model", required=True, choices=["real", "complex"], help="Model family")
    parser.add_argument("--config", default="config/config.json", help="Config path")
    parser.add_argument("--bridge-host", default="127.0.0.1")
    parser.add_argument("--bridge-port", type=int, default=50051)
    parser.add_argument("--batches", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--labels", default="0,1,2", help="Comma separated labels to analyze (e.g. 0,1,2)")
    parser.add_argument("--cov-eps", type=float, default=1e-2)
    parser.add_argument("--latent-max-per-class", type=int, default=4096)
    parser.add_argument("--report-every", type=int, default=200)
    parser.add_argument("--out", default="analysis_report.json")
    parser.add_argument("--save-latents", default="", help="Optional .npz path to save per-class latents")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    labels = tuple(int(x.strip()) for x in args.labels.split(","))
    if len(labels) != 3:
        raise ValueError("labels must have exactly 3 integers, e.g. 0,1,2")

    ckpt = load_checkpoint(args.checkpoint)
    cfg = load_cfg(args.config, checkpoint_payload=ckpt)

    # Connect bridge
    bridge = create_and_connect_bridge(host=args.bridge_host, port=args.bridge_port)
    if not bridge:
        raise RuntimeError(f"Go Bridge connect failed: {args.bridge_host}:{args.bridge_port}")

    device = resolve_device(args.device)

    # Model
    model = instantiate_model(args.model, cfg)
    model.eval()

    # Build lazy layers on CPU then move
    # input dimension: runtime batch decides; still we can warmup with cfg if exists
    input_dim = int(cfg.get("model", {}).get("input_dimension", 256))
    with torch.no_grad():
        _ = model(torch.randn(1, input_dim))
    model.load_state_dict(ckpt.get("state_dict", ckpt))  # support direct state_dict
    model = model.to(device)
    model.eval()

    # Iter
    data_iter = StreamDataIterator(bridge=bridge, batch_size=args.batch_size, balance_labels=False)
    data_iter.start()

    # Aggregators
    agg: Dict[int, ClassAgg] = {c: ClassAgg() for c in labels}
    zbuf = LatentBuffer(labels=labels, max_per_class=args.latent_max_per_class)

    # DP fingerprints collected over time
    dp_snaps = []

    sigma = float(cfg.get("crypto", {}).get("noise_sigma", 0.05))

    try:
        for step in range(1, args.batches + 1):
            if _STOP:
                break

            batch = data_iter.get_batch()
            if batch is None:
                continue

            x_np = batch.features  # (B,D)
            y_np = getattr(batch, "labels", None)
            if y_np is None:
                # some implementations use .label per sample; adapt if needed
                y_np = getattr(batch, "label", None)
            if y_np is None:
                # last fallback: maybe DataBatch has .labels inside; if not available, can't do per-class
                raise RuntimeError("Batch labels not found (expected batch.labels).")

            x = torch.from_numpy(x_np).float().to(device)
            y = torch.from_numpy(np.asarray(y_np)).long().to("cpu")

            with torch.no_grad():
                out = model.forward(x) if hasattr(model, "forward") else model(x)
                z = extract_latent(out)
                if z is None:
                    raise RuntimeError("Model output has no latent 'z' (needed for analysis).")

                # recon mean and mse
                recon_mean = recon_mean_from_output(out)
                if recon_mean is None:
                    raise RuntimeError("Cannot infer reconstruction mean from model output.")
                mse_per_sample = ((x - recon_mean) ** 2).mean(dim=1)  # (B,)

                # loglik / nll
                # prefer compute from mixture if pi_logits exists; else try compute_loss metrics
                if "pi_logits" in out:
                    ll_per_sample = mixture_loglik_bits(x, out["pi_logits"], sigma=sigma)  # (B,)
                    nll_per_sample = -ll_per_sample
                else:
                    # fallback: compute_loss
                    loss_t, m = model.compute_loss(x, step=0)
                    # try loglik key
                    if "loglik" in m:
                        ll_per_sample = torch.full((x.shape[0],), float(m["loglik"]), device=device)
                        nll_per_sample = -ll_per_sample
                    elif "nll" in m:
                        nll_per_sample = torch.full((x.shape[0],), float(m["nll"]), device=device)
                        ll_per_sample = -nll_per_sample
                    else:
                        raise RuntimeError("Neither pi_logits nor (loglik/nll) available for likelihood analysis.")

                # KL (prefer per-sample if available; else from mu/logvar)
                if "mu" in out and "logvar" in out:
                    # per-sample KL
                    mu = out["mu"]
                    logvar = out["logvar"]
                    kl_ps = (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1))  # (B,)
                else:
                    # fallback to compute_loss scalar
                    _, m = model.compute_loss(x, step=0)
                    kl_ps = torch.full((x.shape[0],), float(m.get("kl", 0.0)), device=device)

                # latent stats (batch)
                z_norm, z_cov_trace, z_cond = latent_stats(z)

            # update latent buffer for separation
            zbuf.update(z, y)

            # DP snapshot every report interval (or tune)
            if step % max(1, args.report_every) == 0:
                dp_snaps.append({"step": step, **dp_fingerprint(x)})

            # aggregate per class
            # do everything on CPU scalars
            ll_ps_cpu = ll_per_sample.detach().to("cpu")
            nll_ps_cpu = nll_per_sample.detach().to("cpu")
            mse_ps_cpu = mse_per_sample.detach().to("cpu")
            kl_ps_cpu = kl_ps.detach().to("cpu")

            for c in labels:
                idx = (y == c).nonzero(as_tuple=False).view(-1)
                if idx.numel() == 0:
                    continue
                n = int(idx.numel())
                ll = float(ll_ps_cpu.index_select(0, idx).mean().item())
                nll = float(nll_ps_cpu.index_select(0, idx).mean().item())
                mse = float(mse_ps_cpu.index_select(0, idx).mean().item())
                klv = float(kl_ps_cpu.index_select(0, idx).mean().item())
                agg[c].add(n=n, ll=ll, nll=nll, kl=klv, mse=mse,
                           z_norm=z_norm, z_cov_trace=z_cov_trace, z_cond_proxy=z_cond)

            # console report
            if step % max(1, args.report_every) == 0:
                sep = zbuf.compute_sep(cov_eps=args.cov_eps)
                m0 = agg[labels[0]].mean()
                m1 = agg[labels[1]].mean()
                m2 = agg[labels[2]].mean()

                print(f"\n[ANALYZE] step={step}/{args.batches} device={device} sigma={sigma}")
                print(f"  c{labels[0]} n={m0.get('n',0)} LL={m0.get('ll',0):.3f} NLL={m0.get('nll',0):.3f} KL={m0.get('kl',0):.4f} MSE={m0.get('recon_mse',0):.6f}")
                print(f"  c{labels[1]} n={m1.get('n',0)} LL={m1.get('ll',0):.3f} NLL={m1.get('nll',0):.3f} KL={m1.get('kl',0):.4f} MSE={m1.get('recon_mse',0):.6f}")
                print(f"  c{labels[2]} n={m2.get('n',0)} LL={m2.get('ll',0):.3f} NLL={m2.get('nll',0):.3f} KL={m2.get('kl',0):.4f} MSE={m2.get('recon_mse',0):.6f}")
                print(f"  z: ||z||={m1.get('z_norm',0):.3f} cov_tr={m1.get('z_cov_trace',0):.3f} cond~={m1.get('z_cond_proxy',0):.3f}")

                last_dp = dp_snaps[-1] if dp_snaps else {}
                if last_dp:
                    print(f"  DP fingerprint: mean={last_dp['x_mean']:.3f} std={last_dp['x_std']:.3f} mid={last_dp['x_mid_0p1_0p9']*100:.1f}% ent={last_dp['x_entropy_proxy']:.3f}")

                if sep is not None:
                    print(f"  SEP: Fisher={sep['latent_fisher']:.4f} M01={sep['mahal_01']:.3f} M02={sep['mahal_02']:.3f} M12={sep['mahal_12']:.3f} (nmin={sep['min_required']})")
                else:
                    print(f"  SEP: waiting (need >= {zbuf.min_required()} per class)")

    finally:
        data_iter.stop()
        try:
            bridge.disconnect()
        except Exception:
            pass

    # Final report
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_family": args.model,
        "device": str(device),
        "sigma": sigma,
        "labels": labels,
        "per_class": {f"c{c}": agg[c].mean() for c in labels},
        "sep": zbuf.compute_sep(cov_eps=args.cov_eps),
        "dp_snapshots": dp_snaps,
        "notes": {
            "interpretation": {
                "compression": "Lower NLL and/or lower MSE => better reconstruction/compressibility (compare within same model & sigma).",
                "latent": "z_cond_proxy >> 1 suggests anisotropy/collapse risk; check KL and separation.",
            }
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n✅ Saved analysis report: {out_path.resolve()}")

    # Optionally save latents
    if args.save_latents:
        lat_path = Path(args.save_latents)
        lat_path.parent.mkdir(parents=True, exist_ok=True)
        npz = {}
        for c in labels:
            npz[f"z_c{c}"] = zbuf.buf[c].cpu().numpy()
        np.savez_compressed(lat_path, **npz)
        print(f"✅ Saved latents: {lat_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

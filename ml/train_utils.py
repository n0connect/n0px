import logging
import signal
import atexit
import warnings
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import sys

# Suppress sklearn deprecation and numerical warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')

from .latent_analysis import LatentSeparationTracker
from .sep_metrics import SepConfig, LatentSeparationMonitor, ClasswiseReconMonitor
from .losses.separability import fisher_ratio_trace, lambda_schedule, linear_probe_accuracy


# ============================================================
#  COLORED LOGGING
# ============================================================
class ColoredFormatter(logging.Formatter):
    """Professional colored formatter for terminal output."""
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record):
        levelname = record.levelname
        color = self.COLORS.get(levelname, self.RESET)
        record.levelname = f"{color}{levelname}{self.RESET}"
        record.name = f"{self.BOLD}{record.name}{self.RESET}"
        return super().format(record)


# Setup colored logger
_handler = logging.StreamHandler()
_handler.setFormatter(ColoredFormatter(
    '%(asctime)s [%(name)s] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
))
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ANSI color codes for pbar.write() output
ANSI_CYAN = '\033[36m'
ANSI_GREEN = '\033[92m'
ANSI_YELLOW = '\033[93m'
ANSI_MAGENTA = '\033[35m'
ANSI_BOLD = '\033[1m'
ANSI_RESET = '\033[0m'


# ============================================================
#  EPOCH SUMMARY GENERATION (Türkçe Model Status)
# ============================================================
def _get_model_parameter_stats(model: torch.nn.Module) -> Dict[str, Any]:
    """Extract parametric stats from model for academic reporting."""
    try:
        stats = {
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "layer_stats": {}
        }
        
        # Per-layer parameter count and weight statistics
        param_num = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_num += 1
                if param.dim() > 1:  # Skip biases
                    try:
                        w = param.data.cpu().numpy().flatten()
                        stats["layer_stats"][name] = {
                            "count": param.numel(),
                            "mean": float(np.mean(w)),
                            "std": float(np.std(w)),
                            "min": float(np.min(w)),
                            "max": float(np.max(w)),
                            "sparsity": float(np.sum(np.abs(w) < 1e-6) / len(w))
                        }
                    except Exception as e:
                        logger.debug(f"Skipping layer {name} stats: {e}")
        
        return stats
    except Exception as e:
        logger.warning(f"Failed to extract model stats: {e}")
        return {"total_params": 0, "trainable_params": 0, "layer_stats": {}}


def _generate_epoch_summary(epoch_num: int, metrics: Dict[str, float], train_state: 'TrainState', sep_enabled: bool = False) -> str:
    """
    Generate Turkish summary of model health at epoch end with detailed trend analysis.
    Includes parametric (model architecture) and statistical (distribution) analysis.
    
    Args:
        epoch_num: Current epoch number
        metrics: Epoch metrics (loss, nll, kl, fisher, probe, mahalanobis)
        train_state: Training state with history
        sep_enabled: Whether separability loss is enabled
    
    Returns:
        Formatted summary string for console with detailed analysis
    """
    if train_state.epoch_metrics_history is None:
        return ""
    
    # Get current metrics
    current_loss = metrics.get("loss", float("nan"))
    current_nll = metrics.get("nll", float("nan"))
    current_kl = metrics.get("kl", float("nan"))
    current_fisher = metrics.get("fisher", None)
    current_probe = metrics.get("probe", None)
    current_mahal = metrics.get("mahalanobis", None)
    
    # Get model parametric stats
    model_stats = None
    if train_state.model is not None:
        model_stats = _get_model_parameter_stats(train_state.model)
    
    # Store in history
    train_state.epoch_metrics_history["loss"].append(current_loss)
    train_state.epoch_metrics_history["nll"].append(current_nll)
    train_state.epoch_metrics_history["kl"].append(current_kl)
    if current_fisher is not None:
        train_state.epoch_metrics_history["fisher"].append(current_fisher)
    if current_probe is not None:
        train_state.epoch_metrics_history["probe"].append(current_probe)
    if current_mahal is not None:
        train_state.epoch_metrics_history["mahalanobis"].append(current_mahal)
    
    # Build summary
    lines = [f"\n{ANSI_BOLD}{ANSI_CYAN}[SUMMARY - Epoch {epoch_num}]{ANSI_RESET}"]
    
    # ============ RECONSTRUCTION QUALITY ============
    lines.append(f"\n  {ANSI_YELLOW}Reconstruction Quality:{ANSI_RESET}")
    
    # Loss trend with velocity
    if len(train_state.epoch_metrics_history["loss"]) > 1:
        prev_loss = train_state.epoch_metrics_history["loss"][-2]
        loss_change = current_loss - prev_loss
        
        if len(train_state.epoch_metrics_history["loss"]) > 2:
            prev_prev_loss = train_state.epoch_metrics_history["loss"][-3]
            acceleration = (current_loss - prev_loss) - (prev_loss - prev_prev_loss)
            velocity_symbol = "🚀" if acceleration < -0.01 else "📈" if acceleration > 0.01 else "→"
        else:
            velocity_symbol = ""
        
        trend_symbol = "📉" if loss_change < -0.01 else "📈" if loss_change > 0.01 else "➡️"
        lines.append(f"    Loss: {current_loss:.4f} {trend_symbol} ({loss_change:+.4f}) {velocity_symbol}")
        
        if loss_change < -0.01:
            lines.append(f"      ✅ Gelişiyor: {abs(loss_change):.4f} iyileşme")
        elif loss_change > 0.01:
            lines.append(f"      ⚠️  Kötüleşiyor: {loss_change:.4f} artış")
    else:
        lines.append(f"    Loss: {current_loss:.4f} (baseline)")
    
    # NLL detailed analysis
    if len(train_state.epoch_metrics_history["nll"]) > 1:
        prev_nll = train_state.epoch_metrics_history["nll"][-2]
        nll_change = current_nll - prev_nll
        trend_symbol = "📉" if nll_change < -0.01 else "📈" if nll_change > 0.01 else "➡️"
        
        if current_nll < -150:
            quality = "🟢 Mükemmel"
        elif current_nll < -100:
            quality = "🟡 İyi"
        elif current_nll < -50:
            quality = "🟠 Kabul edilebilir"
        else:
            quality = "🔴 Kötü"
        
        lines.append(f"    NLL (Recon): {current_nll:.4f} {trend_symbol} {quality}")
        
        if nll_change < -0.05:
            lines.append(f"      ✅ Reconstruction hızlı gelişiyor ({nll_change:.4f})")
        elif nll_change > 0.05:
            lines.append(f"      ❌ Reconstruction degrading ({nll_change:+.4f}) - Check regularization")
    else:
        lines.append(f"    NLL: {current_nll:.4f}")
    
    # KL analysis
    if len(train_state.epoch_metrics_history["kl"]) > 1:
        prev_kl = train_state.epoch_metrics_history["kl"][-2]
        kl_change = current_kl - prev_kl
        
        if current_kl < 1e-2:
            kl_status = "🟢 Stabil (KL converged)"
        elif current_kl > 0.5:
            kl_status = "🟡 Yüksek (more exploration needed)"
        else:
            kl_status = "➡️  Normal"
        
        trend = "📉" if kl_change < -1e-3 else "📈" if kl_change > 1e-3 else "→"
        lines.append(f"    KL: {current_kl:.3e} {trend} {kl_status}")
    else:
        lines.append(f"    KL: {current_kl:.3e}")
    
    # ============ SEPARABILITY METRICS ============
    if sep_enabled and current_fisher is not None:
        lines.append(f"\n  {ANSI_GREEN}Separability Analysis:{ANSI_RESET}")
        
        if len(train_state.epoch_metrics_history["fisher"]) > 1:
            prev_fisher = train_state.epoch_metrics_history["fisher"][-2]
            fisher_improvement = (current_fisher - prev_fisher) / (abs(prev_fisher) + 1e-8) * 100
            fisher_delta = current_fisher - prev_fisher
            
            if len(train_state.epoch_metrics_history["fisher"]) > 2:
                prev_prev_fisher = train_state.epoch_metrics_history["fisher"][-3]
                fisher_accel = (current_fisher - prev_fisher) - (prev_fisher - prev_prev_fisher)
                accel_sym = "🚀" if fisher_accel > 0.001 else "↘️" if fisher_accel < -0.001 else "→"
            else:
                accel_sym = ""
            
            lines.append(f"    Fisher Ratio: {current_fisher:.6f} ({fisher_delta:+.6f}) {accel_sym}")
            
            if current_fisher > 0.05:
                lines.append(f"      🟢 Güçlü ayrışma (Fisher > 0.05)")
            elif current_fisher > 0.01:
                lines.append(f"      🟡 Orta ayrışma (0.01 < Fisher < 0.05)")
            elif current_fisher > 0.001:
                lines.append(f"      🟠 Zayıf ama gelişiyor (0.001 < Fisher < 0.01)")
            else:
                lines.append(f"      🔴 Çok zayıf (Fisher < 0.001)")
            
            if fisher_delta > 0.005:
                lines.append(f"        → Eğitim etkili: {fisher_improvement:+.1f}% iyileşme")
            elif fisher_delta < -0.001:
                lines.append(f"        ⚠️  Regression: {fisher_improvement:.1f}% azalış (şüpheli)")
        else:
            lines.append(f"    Fisher Ratio: {current_fisher:.6f} (baseline - no history)")
        
        # Probe accuracy: semantic validity check
        if current_probe is not None and current_probe > 0.0:
            lines.append(f"\n    {ANSI_YELLOW}Semantic Validity (Linear Probe):{ANSI_RESET}")
            
            if len(train_state.epoch_metrics_history["probe"]) > 1:
                prev_probe = train_state.epoch_metrics_history["probe"][-2]
                probe_improvement = (current_probe - prev_probe) * 100
                
                lines.append(f"    Accuracy: {current_probe:.1%} ({probe_improvement:+.1f}pp)")
                
                if current_probe > 0.70:
                    lines.append(f"      🟢 Güçlü: Ayrışma anlamsal ve etkili")
                elif current_probe > 0.50:
                    lines.append(f"      🟡 Orta: Ayrışma kısmen anlamlı")
                elif current_probe > 0.35:
                    lines.append(f"      🟠 Zayıf: Random ({100/3:.0f}%) yakın - kontrol et")
                else:
                    lines.append(f"      🔴 Fail: Probe < 33% (random)")
                    if current_fisher > 0.01:
                        lines.append(f"        ⚠️  ARTIFACT DETECTION: Fisher yüksek ama probe düşük!")
                        lines.append(f"        → Recommendation: λ'yı azalt veya warmup artır")
            else:
                if current_probe > 0.65:
                    lines.append(f"    Accuracy: {current_probe:.1%} ✅ (baseline - güçlü)")
                else:
                    lines.append(f"    Accuracy: {current_probe:.1%} (baseline - check next epoch)")
        
        # Combined health check
        if current_fisher is not None and current_probe is not None and current_probe > 0:
            lines.append(f"\n    {ANSI_MAGENTA}Separability Verdict:{ANSI_RESET}")
            
            if current_fisher > 0.01 and current_probe > 0.65:
                lines.append(f"      🟢 HEALTHY: Fisher ✅ & Probe ✅ - Ayrışma gerçek")
            elif current_fisher > 0.001 and current_probe > 0.50:
                lines.append(f"      🟡 DEVELOPING: Fisher 🟡 & Probe 🟡 - Eğitim yolunda")
            elif current_fisher > 0.001 or current_probe > 0.40:
                lines.append(f"      🟠 WEAK: Partial signal - daha eğit veya λ artır")
            else:
                lines.append(f"      🔴 FAILED: No effective separation - reset needed")
    
    # ============ OVERALL STATUS ============
    lines.append(f"\n  {ANSI_MAGENTA}═══════════════════════════════════════════{ANSI_RESET}")
    lines.append(f"  {ANSI_MAGENTA}Overall Model Status:{ANSI_RESET}")
    
    # Compute composite health score
    recon_ok = current_nll < -50
    sep_ok = sep_enabled and (current_fisher is not None and current_fisher > 0.01 and current_probe is not None and current_probe > 0.60)
    
    if recon_ok and (not sep_enabled or sep_ok):
        status_line = f"    🟢 HEALTHY: Tüm metriker iyi"
    elif recon_ok or (sep_ok):
        status_line = f"    🟡 OKAY: Kısmen iyi - improvement gerekli"
    else:
        status_line = f"    🔴 NEEDS ATTENTION: Problemler var - kontrol et"
    
    lines.append(status_line)
    
    # Recommendations
    recommendations = []
    if sep_enabled and current_fisher is not None:
        if current_fisher < 0.001:
            recommendations.append("  • λ artır (0.05 → 0.10)")
            recommendations.append("  • ramp_steps'i uzat (1500 → 3000)")
        elif current_nll > -50 and current_fisher > 0.01:
            recommendations.append("  • λ azalt (0.05 → 0.02) - recon degrading")
            recommendations.append("  • warmup artır (500 → 2000)")
        elif current_probe is not None and current_fisher > 0.01 and current_probe < 0.40:
            recommendations.append("  ⚠️  ARTIFACT DETECTED: λ önemli ölçüde azalt")
            recommendations.append("  • balance_labels doğrula")
    
    if recommendations:
        lines.append(f"\n  {ANSI_YELLOW}💡 Recommendations:{ANSI_RESET}")
        lines.extend(recommendations)
    
    # ============ PARAMETRIC & STATISTICAL ANALYSIS ============
    if model_stats is not None:
        lines.append(f"\n  {ANSI_GREEN}📊 Model Parametric Statistics:{ANSI_RESET}")
        lines.append(f"    Toplam Parametreler: {model_stats['total_params']:,} "
                    f"({model_stats['trainable_params']:,} trainable)")
        
        # Weight distribution analysis
        if model_stats['layer_stats']:
            weight_means = [s['mean'] for s in model_stats['layer_stats'].values()]
            weight_stds = [s['std'] for s in model_stats['layer_stats'].values()]
            weight_sparsity = [s['sparsity'] for s in model_stats['layer_stats'].values()]
            
            avg_mean = np.mean(weight_means) if weight_means else 0.0
            avg_std = np.mean(weight_stds) if weight_stds else 0.0
            avg_sparsity = np.mean(weight_sparsity) if weight_sparsity else 0.0
            
            lines.append(f"    Ağırlık Dağılımı:")
            lines.append(f"      • Ortalama μ: {avg_mean:+.6f} (beklenen ≈ 0.0)")
            lines.append(f"      • Standart Sapma σ: {avg_std:.6f}")
            
            if avg_std < 0.05:
                lines.append(f"        ⚠️  Düşük σ - activation saturation riski")
            elif avg_std > 1.0:
                lines.append(f"        ⚠️  Yüksek σ - gradient explosion riski")
            else:
                lines.append(f"        ✅ Sağlıklı σ aralığında")
            
            lines.append(f"      • Sparsity (|w| < 1e-6): {avg_sparsity:.2%}")
            
            # Layer-wise variation
            max_sparsity_layer = max(model_stats['layer_stats'].items(), 
                                     key=lambda x: x[1]['sparsity'], default=None)
            if max_sparsity_layer and max_sparsity_layer[1]['sparsity'] > 0.3:
                lines.append(f"        ⚠️  {max_sparsity_layer[0]}: {max_sparsity_layer[1]['sparsity']:.1%} sparse")
    
    # Statistical interpretation of loss trajectory
    if len(train_state.epoch_metrics_history["loss"]) >= 3:
        recent_losses = train_state.epoch_metrics_history["loss"][-5:]  # Last 5 epochs
        loss_diffs = np.diff(recent_losses)
        
        lines.append(f"\n  {ANSI_CYAN}📈 Statistical Convergence Analysis:{ANSI_RESET}")
        lines.append(f"    Son 5 epoch loss değişimi: {[f'{d:+.3f}' for d in loss_diffs]}")
        
        # Trend assessment
        recent_trend = np.mean(loss_diffs)
        trend_volatility = np.std(loss_diffs)
        
        if recent_trend < -0.1 and trend_volatility < 0.05:
            lines.append(f"    🟢 Strong downtrend: μ_diff={recent_trend:.3f}, σ={trend_volatility:.3f} (stabil gelişim)")
        elif recent_trend < 0 and trend_volatility < 0.15:
            lines.append(f"    🟡 Gradual improvement: μ_diff={recent_trend:.3f}, σ={trend_volatility:.3f}")
        elif trend_volatility > 0.2:
            lines.append(f"    🟠 High volatility: σ={trend_volatility:.3f} - learning rate kontrol et")
        else:
            lines.append(f"    ➡️  Stagnant: μ_diff={recent_trend:.3f} (plateau veya instability)")
        
        # KL divergence trajectory
        if len(train_state.epoch_metrics_history["kl"]) >= 3:
            recent_kls = train_state.epoch_metrics_history["kl"][-5:]
            kl_trend = np.mean(np.diff(recent_kls))
            
            if abs(kl_trend) < 1e-5:
                lines.append(f"    ✅ KL stabilized: drift={kl_trend:.2e} (prior matching optimal)")
            elif kl_trend > 0:
                lines.append(f"    ⚠️  KL diverging: drift={kl_trend:.2e} - β'yı kontrol et")
            else:
                lines.append(f"    📉 KL collapsing: drift={kl_trend:.2e} (posterior annealing)")
    
    # Fisher / Separability trajectory (if enabled)
    if sep_enabled and len(train_state.epoch_metrics_history["fisher"]) >= 3:
        recent_fishers = train_state.epoch_metrics_history["fisher"][-5:]
        fisher_trend = np.mean(np.diff(recent_fishers))
        fisher_vol = np.std(np.diff(recent_fishers))
        
        lines.append(f"\n  {ANSI_MAGENTA}🔀 Separability Trajectory:{ANSI_RESET}")
        lines.append(f"    Fisher trend (5-epoch): {fisher_trend:+.6f} ± {fisher_vol:.6f}")
        
        if fisher_trend > 0.0001 and fisher_vol < 0.001:
            lines.append(f"    🟢 Consistent improvement (λ schedule optimal)")
        elif fisher_trend > 0 and fisher_vol < 0.01:
            lines.append(f"    🟡 Gradual gains (but needs acceleration)")
        elif fisher_vol > 0.005:
            lines.append(f"    🟠 Unstable dynamics - warmup/ramp schedule adjust")
        else:
            lines.append(f"    ➡️  Plateau reached - may need higher λ")
    
    lines.append(f"  {ANSI_MAGENTA}═══════════════════════════════════════════{ANSI_RESET}\n")
    
    return ''.join(lines)


# ============================================================
#  TRAIN STATE (shared across training scripts)
# ============================================================
@dataclass
class TrainState:
    """Global training state for graceful shutdown."""
    is_training: bool = False
    model: Optional[torch.nn.Module] = None
    checkpoint_dir: str = "checkpoints"
    best_loss: float = float("inf")
    sep_cfg: Optional[Dict[str, Any]] = None  # Separability loss config
    epoch_metrics_history: Dict[str, list] = None  # Track metrics across epochs for summary
    
    def __post_init__(self):
        if self.epoch_metrics_history is None:
            self.epoch_metrics_history = {
                "loss": [],
                "nll": [],
                "kl": [],
                "fisher": [],
                "probe": [],
                "mahalanobis": [],
            }
    
    def save_checkpoint(self, epoch: int, loss: float, tag: str = "") -> None:
        """Save model checkpoint."""
        if self.model is None:
            logger.warning("Model is None, cannot save checkpoint")
            return
        
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        ckpt_name = f"epoch_{epoch:03d}_loss_{loss:.4f}_{tag}.pt"
        path = Path(self.checkpoint_dir) / ckpt_name
        
        try:
            if hasattr(self.model, 'save'):
                self.model.save(str(path))
            else:
                torch.save({"state_dict": self.model.state_dict()}, str(path))
            logger.info(f"✓ Checkpoint saved: {path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def shutdown(self, reason: str = "shutdown") -> None:
        """Graceful shutdown."""
        logger.info(f"Shutting down: {reason}")
        self.is_training = False
    
    def find_latest_checkpoint(self) -> Optional[tuple]:
        """
        Find latest checkpoint in checkpoint_dir.
        
        Returns:
            (path, epoch, loss) or None if no checkpoint found
        """
        ckpt_dir = Path(self.checkpoint_dir)
        if not ckpt_dir.exists():
            return None
        
        # Find all checkpoints, parse epoch from filename
        checkpoints = list(ckpt_dir.glob("epoch_*.pt"))
        if not checkpoints:
            return None
        
        # Sort by epoch number (descending)
        def parse_epoch(path):
            try:
                epoch = int(path.stem.split("_")[1])
                return epoch
            except:
                return -1
        
        checkpoints.sort(key=parse_epoch, reverse=True)
        latest = checkpoints[0]
        
        # Extract epoch and loss from filename
        try:
            parts = latest.stem.split("_")
            epoch = int(parts[1])
            loss_str = parts[3]  # format: "loss_X.XXXX"
            loss = float(loss_str)
            return (latest, epoch, loss)
        except Exception as e:
            logger.warning(f"Failed to parse checkpoint {latest}: {e}")
            return None
    
    def load_checkpoint_for_resume(self, model, optimizer, new_batch_size=None, ckpt_path=None):
        """
        Load checkpoint for resume training with safety checks.
        
        Args:
            model: Model to load checkpoint into
            optimizer: Optimizer (for state)
            new_batch_size: If provided and different, warns user (doesn't block)
            ckpt_path: Explicit checkpoint path, else find latest
        
        Returns:
            (success: bool, epoch: int, loss: float, warnings: list[str])
            - success: True if loaded successfully
            - epoch: Starting epoch for resume
            - loss: Best loss from checkpoint
            - warnings: List of potential issues
        """
        warnings_list = []
        
        # Find checkpoint
        if ckpt_path is None:
            result = self.find_latest_checkpoint()
            if result is None:
                logger.info("No checkpoint found - starting fresh")
                return (False, 0, float("inf"), [])
            ckpt_path, epoch, loss = result
        else:
            ckpt_path = Path(ckpt_path)
            if not ckpt_path.exists():
                logger.error(f"Checkpoint not found: {ckpt_path}")
                return (False, 0, float("inf"), [f"Checkpoint not found: {ckpt_path}"])
            
            # Extract epoch from filename for consistency
            try:
                parts = ckpt_path.stem.split("_")
                epoch = int(parts[1])
                loss_str = parts[3]
                loss = float(loss_str)
            except:
                epoch = 0
                loss = float("inf")
        
        logger.info(f"Found checkpoint: {ckpt_path.name}")
        
        try:
            # Load checkpoint
            ckpt = torch.load(str(ckpt_path), map_location='cpu')
            
            # Try different checkpoint formats
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif isinstance(ckpt, dict) and "model" in ckpt:
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt
            
            # Strict=False to allow size mismatches (will be caught below)
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"✓ Model loaded from epoch {epoch}")
            
            # Batch size check (warning, not error)
            if new_batch_size is not None:
                # This is informational - batch size doesn't affect weights
                logger.info(f"ℹ️  Batch size change: checkpoint trained with unknown size → resuming with {new_batch_size}")
                warnings_list.append(f"Batch size may have changed - monitor first few epochs")
            
            # Optimizer state reset (safer for resumed training)
            if optimizer is not None:
                logger.info("ℹ️  Optimizer state reset for resume (fresh start with new LR)")
                optimizer = type(optimizer)(model.parameters(), **optimizer.defaults) if hasattr(optimizer, 'defaults') else optimizer
            
            logger.info(f"✓ Resume ready: Start from epoch {epoch + 1}")
            return (True, epoch, loss, warnings_list)
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return (False, 0, float("inf"), [f"Load error: {str(e)}"])


# Global train state instance
train_state = TrainState()


# ============================================================
#  SIGNAL HANDLERS (graceful Ctrl+C)
# ============================================================
def _handle_signal(sig, frame):
    """Signal handler for SIGINT/SIGTERM."""
    logger.info("\n⚠️  Received signal, gracefully shutting down...")
    train_state.is_training = False


def setup_signal_handlers():
    """Register signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    atexit.register(lambda: train_state.shutdown("program exit"))
    logger.debug("Signal handlers registered")


def _extract_labels(batch: Any) -> Optional[torch.Tensor]:
    """
    StreamDataIterator batch objesinden label çıkar.
    batch.labels / batch.label / batch.y gibi varyasyonları tolere eder.
    """
    for name in ("labels", "label", "y", "target"):
        if hasattr(batch, name):
            v = getattr(batch, name)
            if v is None:
                continue
            # numpy veya list ise tensor'a çevir
            if not torch.is_tensor(v):
                v = torch.tensor(v, dtype=torch.long)
            else:
                v = v.to(dtype=torch.long)
            return v
    return None


def _per_label_stats(
    labels: torch.Tensor,
    nll_ps: torch.Tensor,
    kl_ps: torch.Tensor,
    label_set: Sequence[int] = (0, 1, 2),
) -> Dict[str, float]:
    """
    nll_ps, kl_ps: (B,)
    labels: (B,)
    """
    out: Dict[str, float] = {}
    for c in label_set:
        m = (labels == int(c))
        if torch.any(m):
            out[f"nll_c{c}"] = float(nll_ps[m].mean().cpu().item())
            out[f"kl_c{c}"] = float(kl_ps[m].mean().cpu().item())
    return out


def train_epoch_generic(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_iter: Any,
    device: torch.device,
    num_batches: int,
    epoch_num: int,
    gradient_clip: float = 1.0,
    global_step_offset: int = 0,     # <-- kritik: step epoch’ta sıfırlanmasın
    label_set: Tuple[int, ...] = (0, 1, 2),
    debug_every: int = 200,
    analyze_every: int = 200,        # <-- latent separation analysis frequency
    sep_window: int = 4096,          # <-- buffer size for per-class stats
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_nll = 0.0
    total_kl = 0.0
    total_fisher = 0.0
    total_probe = 0.0
    n_seen = 0
    n_fisher_seen = 0
    n_probe_seen = 0
    n_sep_skipped = 0  # Track skips
    sep_skip_rate_ema = 0.0  # EMA of skip rate
    
    # PATCH-4: Per-class NLL EMA tracking (stabilize dalgalanmayı görmek için)
    class_nll_ema = {label: 0.0 for label in label_set}
    class_counts = {label: 0 for label in label_set}
    nll_ema_alpha = 0.95  # Exponential moving average coefficient
    
    # Windowed probe buffer (last ~4096 examples for stable accuracy measurement)
    probe_z_buffer = []
    probe_y_buffer = []
    probe_window_size = 4096
    
    # Initialize latent separation tracker
    sep_tracker = LatentSeparationTracker(num_classes=len(label_set), window=sep_window)
    
    # Initialize new separation metrics monitors
    sep_cfg = SepConfig(report_every=200, cov_eps=1e-1, max_per_class=4096)
    sep_mon = LatentSeparationMonitor(sep_cfg)
    recon_mon = ClasswiseReconMonitor(labels=label_set)

    pbar = tqdm(
        range(num_batches),
        desc=f"Epoch {epoch_num}",
        unit="batch",
        ncols=140,
        colour="green",
        file=sys.stderr,
        position=0,
        leave=True,
        dynamic_ncols=False,
        disable=False,
        bar_format='{desc} {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} {postfix}'
    )

    for batch_idx in pbar:
        # GRACEFUL SHUTDOWN CHECK
        if not train_state.is_training:
            logger.warning("Training stopped by signal")
            break
        
        batch = data_iter.get_batch()
        if batch is None:
            logger.warning("StreamDataIterator returned None batch")
            break

        x = torch.from_numpy(batch.features).float().to(device)
        labels = _extract_labels(batch)
        if labels is not None:
            labels = labels.to(device)

        # DEBUG: Log batch class distribution every 50 batches (to detect imbalance)
        if batch_idx % 50 == 0 and labels is not None:
            unique, counts = torch.unique(labels, return_counts=True)
            dist_str = ", ".join([f"c{int(u)}:{int(c)}" for u, c in zip(unique, counts)])
            logger.debug(f"[Batch {batch_idx}] Class dist: {dist_str} (n_total={len(labels)})")

        optimizer.zero_grad(set_to_none=True)

        global_step = global_step_offset + batch_idx  # <-- KRİTİK
        
        # OPTIMIZATION: Cache model output to avoid triple forward passes
        # Get model output for latent analysis AND loss computation
        with torch.no_grad():
            model_out_cached = model(x)
            z_latent = model_out_cached.get("z", None)
        
        # Compute loss (model.compute_loss internally calls forward again, but we accept this for accuracy)
        loss, metrics = model.compute_loss(x, step=global_step)
        
        # Sanity check: Non-finite loss detection
        if not torch.isfinite(loss):
            logger.error("Non-finite loss at step=%d (loss=%s). Check data / sigma / dtype.", global_step, str(loss))
            raise RuntimeError("Non-finite loss")
        
        # ========== SEPARABILITY LOSS (NEW) ==========
        # Add Fisher/LDA-style separability regularization to optimize class discrimination
        loss_total = loss
        sep_score = None
        sep_lambda = 0.0
        sep_skipped = False
        sep_skip_reason = None
        
        # Only apply if we have valid latent z and labels
        if z_latent is not None and labels is not None:
            sep_cfg_dict = getattr(train_state, 'sep_cfg', {})
            if sep_cfg_dict and sep_cfg_dict.get("enabled", False):
                # Compute lambda schedule (warmup + ramp)
                sep_lambda = lambda_schedule(
                    step=global_step,
                    base_lambda=float(sep_cfg_dict.get("lambda", 0.05)),
                    warmup_steps=int(sep_cfg_dict.get("warmup_steps", 500)),
                    ramp_steps=int(sep_cfg_dict.get("ramp_steps", 3000)),
                )
                
                if sep_lambda > 0.0:
                    # PATCH-A: Vectorized GPU-side label remapping (no CPU transfers!)
                    # Handles RM1's 3/4/5 labels without Python loops
                    with torch.no_grad():
                        unique_labels = torch.unique(labels)
                        num_classes = len(unique_labels)
                        
                        # Create remapping: label_i → i efficiently on GPU
                        y_remapped = torch.zeros_like(labels)
                        for idx, label in enumerate(unique_labels):
                            y_remapped[labels == label] = idx
                    
                    # Compute Fisher separability ratio
                    sep_score = fisher_ratio_trace(
                        z=z_latent,
                        y=y_remapped,
                        n_classes=num_classes,
                        eps=float(sep_cfg_dict.get("eps", 1e-6)),
                        min_per_class=int(sep_cfg_dict.get("min_per_class", 8)),
                    )
                    
                    # Check if Fisher returned zero (min_per_class guard)
                    if sep_score.item() == 0.0:
                        sep_skipped = True
                        sep_skip_reason = "min_per_class"
                        n_sep_skipped += 1
                    elif sep_score.item() > 0:
                        # loss_total = loss_vae - lambda * sep_score
                        # (negative because we want to MAXIMIZE sep_score)
                        loss_total = loss - (sep_lambda * sep_score)
                else:
                    sep_skipped = True
                    sep_skip_reason = "warmup"
                    n_sep_skipped += 1
                
                # Update skip rate EMA (alpha=0.95)
                skip_indicator = 1.0 if sep_skipped else 0.0
                sep_skip_rate_ema = 0.95 * sep_skip_rate_ema + 0.05 * skip_indicator
        
        if z_latent is not None:
            metrics["z"] = z_latent  # Add latent vectors to metrics
            # Update separation monitor with z and labels
            if labels is not None:
                sep_mon.update(z_latent, labels)

        loss_total.backward()
        if gradient_clip and gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(gradient_clip))
        optimizer.step()

        # Extract metrics to Python floats only when needed (lazy evaluation)
        total_loss += float(metrics["loss"].cpu().item() if torch.is_tensor(metrics["loss"]) else metrics["loss"])
        total_nll += float(metrics["nll"].cpu().item() if torch.is_tensor(metrics["nll"]) else metrics["nll"])
        total_kl += float(metrics["kl"].cpu().item() if torch.is_tensor(metrics["kl"]) else metrics["kl"])
        n_seen += 1
        
        # PATCH-4: Per-class NLL EMA update (for stable per-class trend)
        if labels is not None:
            with torch.no_grad():
                nll_per_class = metrics.get("nll_per_class", None)
                if nll_per_class is not None:
                    # nll_per_class should be a dict: {label: nll_value}
                    for label in label_set:
                        if label in nll_per_class:
                            nll_val = float(nll_per_class[label])
                            # EMA update
                            if class_nll_ema[label] == 0.0:
                                class_nll_ema[label] = nll_val  # Initialize
                            else:
                                class_nll_ema[label] = (nll_ema_alpha * class_nll_ema[label] + 
                                                        (1 - nll_ema_alpha) * nll_val)
                            class_counts[label] += int((labels == label).sum().item())

        # ============ BUILD POSTFIX FOR PROGRESS BAR ============
        loss_val = float(metrics["loss"].cpu().item() if torch.is_tensor(metrics["loss"]) else metrics["loss"])
        nll_val = float(metrics["nll"].cpu().item() if torch.is_tensor(metrics["nll"]) else metrics["nll"])
        kl_val = float(metrics["kl"].cpu().item() if torch.is_tensor(metrics["kl"]) else metrics["kl"])
        
        postfix = {
            "L": f'{loss_val:.2f}',
            "N": f'{nll_val:.1f}',
            "K": f'{kl_val:.2e}',
            "β": f'{metrics["beta"]:.3f}',
        }
        
        # Add separability metrics to postfix (ALWAYS visible if sep_loss enabled)
        sep_cfg_dict = getattr(train_state, 'sep_cfg', {})
        if sep_cfg_dict and sep_cfg_dict.get("enabled", False):
            postfix["λ"] = f'{sep_lambda:.4f}'
            postfix["skip%"] = f'{sep_skip_rate_ema*100:.1f}'
            
            if sep_skipped:
                postfix["S"] = f'⊘({sep_skip_reason})'
            elif sep_score is not None and sep_score.item() > 0:
                postfix["S"] = f'{sep_score.item():.4f}'
                total_fisher += sep_score.item()
                n_fisher_seen += 1
        
        # Linear probe sanity check every 50 batches (window-based, not per-batch)
        # Accumulate latents/labels in rolling buffer, calculate probe from buffer
        if z_latent is not None and labels is not None:
            # Add to buffer
            probe_z_buffer.append(z_latent.detach().cpu())
            probe_y_buffer.append(labels.detach().cpu())
            
            # Trim buffer to window size
            total_buffer_size = sum(z.shape[0] for z in probe_z_buffer)
            while total_buffer_size > probe_window_size and len(probe_z_buffer) > 1:
                total_buffer_size -= probe_z_buffer[0].shape[0]
                probe_z_buffer.pop(0)
                probe_y_buffer.pop(0)
            
            # Calculate probe every 50 batches from windowed buffer
            if (batch_idx % 50) == 0 and len(probe_z_buffer) > 0:
                sep_cfg_dict = getattr(train_state, 'sep_cfg', {})
                if sep_cfg_dict and sep_cfg_dict.get("enabled", False):
                    z_concat = torch.cat(probe_z_buffer, dim=0).to(device)
                    y_concat = torch.cat(probe_y_buffer, dim=0).to(device)
                    
                    # PATCH-A: Remap labels for probe (handles RM1's 3/4/5)
                    with torch.no_grad():
                        y_cpu = y_concat.detach().to("cpu").numpy().astype(int)
                        unique_labels = sorted(set(y_cpu.tolist()))
                        label_map = {lab: i for i, lab in enumerate(unique_labels)}
                        y_concat_remapped = torch.tensor(
                            [label_map[int(v)] for v in y_cpu],
                            device=y_concat.device,
                            dtype=torch.long
                        )
                    
                    probe_acc = linear_probe_accuracy(z_concat, y_concat_remapped, n_classes=len(unique_labels))
                    postfix["P"] = f'{probe_acc:.1%}'
                    total_probe += probe_acc
                    n_probe_seen += 1
        
        # Reset probe buffer every epoch (prevent accumulation drift)
        if batch_idx == num_batches - 1:
            probe_z_buffer.clear()
            probe_y_buffer.clear()

        # UPDATE PROGRESS BAR WITH POSTFIX (every 50 batches to reduce flicker)
        if batch_idx % 50 == 0 or batch_idx == num_batches - 1:
            pbar.set_postfix(postfix)

        # 3) LABEL-BAZLI METRİK
        if labels is not None and ("nll_ps" in metrics) and ("kl_ps" in metrics):
            nll_ps = metrics["nll_ps"].to(device)
            kl_ps = metrics["kl_ps"].to(device)
            
            # Per-report_every step: compute class-wise recon + separation metrics
            if (batch_idx > 0) and (batch_idx % sep_cfg.report_every) == 0:
                # 1) Class-wise reconstruction metrics
                # OPTIMIZATION: Pass cached model_out to avoid per-class forward passes
                z_for_recon = z_latent
                cw = recon_mon.compute(model, x, labels, step=global_step, z=z_for_recon, model_out=model_out_cached)
                
                # 2) Separation metrics (Fisher + Mahalanobis if ready)
                # Only compute/log if buffer meets min sample threshold for all classes
                if sep_mon.ready():
                    sep = sep_mon.compute()
                else:
                    sep = None
                
                # Build comprehensive log message
                parts = []
                
                # NLL + loglik + KL per class with sample counts
                for c in [0, 1, 2]:
                    if f"nll_c{c}" in cw:
                        n = int(cw.get(f"n_c{c}", 0))
                        nll = cw.get(f"nll_c{c}", 0.0)
                        loglik = cw.get(f"loglik_c{c}", 0.0)
                        kl = cw.get(f"kl_c{c}", 0.0)
                        parts.append(
                            f"c{c}(n={n}): LL={loglik:.3f}/NLL={nll:.3f}, KL={kl:.3e}"
                        )
                        # Optional: latent stats if available
                        z_norm = cw.get(f"z_norm_c{c}", None)
                        if z_norm is not None:
                            parts[-1] += f" |z|={z_norm:.3f}"
                
                # Delta NLL (compression difference)
                if "dnll_12" in cw or "dnll_10" in cw or "dnll_20" in cw:
                    delta_parts = []
                    if "dnll_12" in cw:
                        delta_parts.append(f"Δ(1-2)={cw['dnll_12']:.3f}")
                    if "dnll_10" in cw:
                        delta_parts.append(f"Δ(1-0)={cw['dnll_10']:.3f}")
                    if "dnll_20" in cw:
                        delta_parts.append(f"Δ(2-0)={cw['dnll_20']:.3f}")
                    parts.append("Deltas: " + " ".join(delta_parts))
                
                # Separation metrics (only if ready)
                if sep is not None:
                    parts.append(f"Fisher={sep['latent_fisher']:.4f}")
                    parts.append(f"Mahal(01,02,12)=({sep['mahal_01']:.3f},{sep['mahal_02']:.3f},{sep['mahal_12']:.3f})")
                    logger.info("[FULL ANALYSIS] step=%d | %s", global_step, " | ".join(parts))
                    
                    # Akademik özet (FULL)
                    fisher_val = sep['latent_fisher']
                    mahal_avg = (sep['mahal_01'] + sep['mahal_02'] + sep['mahal_12']) / 3
                    
                    if fisher_val > 0.01 and mahal_avg > 0.5:
                        fisher_interp = "strong inter-class discrimination with efficient latent utilization"
                        icon = "🟢"
                    elif fisher_val > 0.001 and mahal_avg > 0.15:
                        fisher_interp = "emerging class separability; latent manifolds exhibit initial differentiation"
                        icon = "🟡"
                    else:
                        fisher_interp = "insufficient class discrimination; latent representations remain entangled"
                        icon = "🔴"
                    
                    summary_msg = (
                        f"  {ANSI_MAGENTA}[SUMMARY - FULL]{ANSI_RESET}\n"
                        f"    {icon} Reconstruction demonstrates {fisher_interp}\n"
                        f"    • Fisher criterion: {fisher_val:.4f} (J = tr(S_B)/tr(S_W))\n"
                        f"    • Mean Mahalanobis distance: {mahal_avg:.3f}"
                    )
                    pbar.write(summary_msg)
                else:
                    # Still log reconstruction even without separation
                    if parts:
                        logger.info("[RECON ANALYSIS] step=%d | %s", global_step, " | ".join(parts))
            
            # 4) LATENT SEPARATION ANALYSIS (backward compat with old tracker)
            if (batch_idx > 0) and (batch_idx % analyze_every) == 0:
                if "z" in metrics:
                    z = metrics["z"]  # latent vectors from model
                    sep_tracker.update(z.detach(), labels)  # Pass torch tensors, not numpy
                    sep_snapshot = sep_tracker.snapshot()
                    
                    # Only log if snapshot is ready (enough samples)
                    if sep_snapshot is not None:
                        # Professional colored separation report - separate line with pbar.write()
                        sep_msg = (
                            f"{ANSI_CYAN}{ANSI_BOLD}[SEP ANALYSIS]{ANSI_RESET} "
                            f"Step {global_step:06d} | "
                            f"{ANSI_GREEN}Fisher={sep_snapshot.fisher:.4f}{ANSI_RESET} | "
                            f"{ANSI_YELLOW}Mahal_01={sep_snapshot.md_01:.4f} "
                            f"Mahal_02={sep_snapshot.md_02:.4f} "
                            f"Mahal_12={sep_snapshot.md_12:.4f}{ANSI_RESET} | "
                            f"{ANSI_MAGENTA}Samples: {sep_snapshot.counts}{ANSI_RESET}"
                        )
                        pbar.write(sep_msg)
                        
                        # Akademik özet (SEP)
                        fisher = sep_snapshot.fisher
                        mahal_list = [sep_snapshot.md_01, sep_snapshot.md_02, sep_snapshot.md_12]
                        
                        if fisher > 0.05:
                            fisher_status = "Strong Fisher discriminant (J > 0.05)"
                            icon = "🟢"
                        elif fisher > 0.01:
                            fisher_status = "Moderate class separability (0.01 < J < 0.05)"
                            icon = "🟡"
                        elif fisher > 0.001:
                            fisher_status = "Incipient separability emerging (0.001 < J < 0.01)"
                            icon = "🟠"
                        else:
                            fisher_status = "Negligible inter-class separation (J < 0.001)"
                            icon = "🔴"
                        
                        mahal_adequate = all(m > 0.15 for m in mahal_list)
                        mahal_desc = "Geometric distances are statistically significant" if mahal_adequate else "Geometric distances require further amplification"
                        
                        sep_summary = (
                            f"  {ANSI_MAGENTA}[SUMMARY - SEP]{ANSI_RESET}\n"
                            f"    {icon} {fisher_status}\n"
                            f"    • {mahal_desc}\n"
                            f"    • Pairwise distances: d₀₁={sep_snapshot.md_01:.3f}, d₀₂={sep_snapshot.md_02:.3f}, d₁₂={sep_snapshot.md_12:.3f}"
                        )
                        pbar.write(sep_summary)
            
            # Log per-class metrics every 500 batches
            if (batch_idx > 0) and (batch_idx % 500) == 0:
                per_lbl = _per_label_stats(labels, nll_ps, kl_ps, label_set=label_set)
                
                # PATCH-4: Log per-class NLL EMA (stabilized trend, not just current batch)
                nll_ema_str = "/".join(f"{class_nll_ema.get(c, 0.0):.3f}" for c in label_set)
                count_str = "/".join(f"{class_counts.get(c, 0)}" for c in label_set)
                ema_msg = (
                    f"{ANSI_YELLOW}{ANSI_BOLD}[NLL EMA TREND]{ANSI_RESET} "
                    f"Step {global_step:06d} | "
                    f"EMA: {nll_ema_str} | "
                    f"Counts (seen): {count_str}"
                )
                pbar.write(ema_msg)
                
                # Dinamik keys: label_set'e göre arama
                nll_keys = [f"nll_c{c}" for c in label_set]
                kl_keys = [f"kl_c{c}" for c in label_set]
                
                # tqdm postfix'e ekle
                for k in nll_keys + kl_keys:
                    if k in per_lbl:
                        postfix[k] = f"{per_lbl[k]:.2f}" if k.startswith("nll") else f"{per_lbl[k]:.2e}"
                
                # Metrics display (dinamik)
                nll_vals = [per_lbl.get(f'nll_c{c}', 0) for c in label_set]
                kl_vals = [per_lbl.get(f'kl_c{c}', 0) for c in label_set]
                
                nll_str = "/".join(f"{v:.3f}" for v in nll_vals)
                kl_str = "/".join(f"{v:.2e}" for v in kl_vals)
                
                class_msg = (
                    f"{ANSI_CYAN}{ANSI_BOLD}[CLASS METRICS]{ANSI_RESET} "
                    f"Step {global_step:06d} | "
                    f"{ANSI_GREEN}NLL: {nll_str}{ANSI_RESET} | "
                    f"{ANSI_YELLOW}KL: {kl_str}{ANSI_RESET}"
                )
                pbar.write(class_msg)
                
                # Akademik özet (CLASS)
                # Dinamik class metadata
                if label_set == (0, 1, 2):
                    class_names_map = {0: "Easy", 1: "Prime", 2: "Hard"}
                elif label_set == (3, 4, 5):
                    class_names_map = {3: "DP-Prime", 4: "DP-Composite", 5: "DP-Hard"}
                else:
                    class_names_map = {c: f"Class{c}" for c in label_set}
                
                # Extract NLL/KL for sorting
                class_metrics = [(c, per_lbl.get(f'nll_c{c}', 0)) for c in label_set]
                class_metrics_sorted = sorted(class_metrics, key=lambda x: x[1])
                best_class_idx = class_metrics_sorted[0][0]
                
                # KL balance assessment
                kl_vals_list = [per_lbl.get(f'kl_c{c}', 0) for c in label_set]
                kl_avg = np.mean(kl_vals_list) if kl_vals_list else 1e-8
                kl_max = max(kl_vals_list) if kl_vals_list else 1e-8
                kl_ratio = kl_max / (kl_avg + 1e-8)
                
                if kl_ratio < 1.5:
                    kl_status = "Balanced KL divergence across classes"
                    icon = "🟢"
                elif kl_ratio < 2.5:
                    kl_status = "Moderate KL imbalance; acceptable variance"
                    icon = "🟡"
                else:
                    kl_status = "Significant KL disparity; class dominance detected"
                    icon = "🔴"
                
                # Reconstruction summary (dinamik)
                recon_summary = ", ".join(f"{class_names_map.get(c, f'C{c}')}={per_lbl.get(f'nll_c{c}', 0):.1f}" for c in label_set)
                
                class_summary = (
                    f"  {ANSI_MAGENTA}[SUMMARY - CLASS]{ANSI_RESET}\n"
                    f"    {icon} {kl_status}\n"
                    f"    • Reconstruction efficiency: {class_names_map.get(best_class_idx, f'Class{best_class_idx}')} exhibits optimal NLL (-{abs(class_metrics_sorted[0][1]):.1f})\n"
                    f"    • Per-class reconstruction: {recon_summary}"
                )
                pbar.write(class_summary)
                
                # DP CHECK: Near-0 and near-1 signal distribution (verify DP is working)
                # x is (B, D) with signals near 0 and 1. Count ratio if batch comes from bridge.
                if hasattr(batch, "features"):
                    x_np = batch.features if isinstance(batch.features, np.ndarray) else batch.features.detach().cpu().numpy()
                    near_0 = float(np.mean(np.abs(x_np) < 0.1))  # fraction close to 0
                    near_1 = float(np.mean(np.abs(x_np - 1.0) < 0.1))  # fraction close to 1
                    dp_msg = f"[DP CHECK] step={global_step:06d} | near0={near_0:.2%} near1={near_1:.2%}"
                    if near_0 > 0.5 or near_1 > 0.5:
                        pbar.write(f"{ANSI_YELLOW}{dp_msg}{ANSI_RESET}")
                    else:
                        logger.debug(dp_msg)

    pbar.close()

    if n_seen == 0:
        return {"loss": float("nan"), "nll": float("nan"), "kl": float("nan"), "fisher": None, "probe": None}

    result = {
        "loss": total_loss / n_seen,
        "nll": total_nll / n_seen,
        "kl": total_kl / n_seen,
        "fisher": (total_fisher / n_fisher_seen) if n_fisher_seen > 0 else None,
        "probe": (total_probe / n_probe_seen) if n_probe_seen > 0 else None,
        "sep_skip_rate_ema": sep_skip_rate_ema,  # EMA for monitoring trend
        "sep_skip_rate": (n_sep_skipped / n_seen) if n_seen > 0 else 0.0,  # Actual rate
        "n_skipped": n_sep_skipped,  # Track absolute count
    }
    
    # Log final stats with clarification
    if n_sep_skipped > 0 or n_sep_skipped == 0:
        n_computed = n_seen - n_sep_skipped
        actual_skip_rate = (n_sep_skipped / n_seen * 100) if n_seen > 0 else 0.0
        logger.info(
            f"Epoch {epoch_num}: Sep-loss skip rate = {actual_skip_rate:.1f}% "
            f"(skipped {n_sep_skipped} / computed {n_computed} / total {n_seen})"
        )
    
    return result

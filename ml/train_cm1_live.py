#!/usr/bin/env python3
"""
train_cm1_live.py: CM1 (Complex-Valued VAE, Differential Privacy Labels)
Labels: 3=dp_prime, 4=dp_easy_composite, 5=dp_hard_composite
Data: Bitstrings from Go Bridge with differential privacy noise (bit values perturbed)
Latent Space: Complex-valued continuous vectors

Features:
  - Graceful Ctrl+C shutdown (signal handling)
  - Colored professional logging + enhanced tqdm
  - Per-label NLL/KL metrics
  - Latent separation analysis
  - KL schedule + free-bits regularization
"""
import sys
import json
import torch
import torch.optim as optim
import argparse
import logging
import atexit
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from models.cm1 import CM1_ComplexAutoEncoder
    from core import create_and_connect_bridge
    from data import StreamDataIterator
    from train_utils import train_state, setup_signal_handlers, train_epoch_generic, ColoredFormatter, _generate_epoch_summary
    from device_support import resolve_device, print_device_info, add_device_argument
except ImportError:
    from ml.models.cm1 import CM1_ComplexAutoEncoder
    from ml.core import create_and_connect_bridge
    from ml.data import StreamDataIterator
    from ml.train_utils import train_state, setup_signal_handlers, train_epoch_generic, ColoredFormatter, _generate_epoch_summary
    from ml.device_support import resolve_device, print_device_info, add_device_argument


# ============================================================
#  SETUP LOGGING (ColoredFormatter only - no basicConfig duplication)
# ============================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.handlers.clear()
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(asctime)s [%(name)s] %(levelname)s %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(handler)
logger.propagate = False

# Register signal handlers FIRST
setup_signal_handlers()
logger.info("Signal handlers registered (Ctrl+C to shutdown)")


def get_model_stats(model: torch.nn.Module) -> dict:
    """Calculate model parameter statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total_params,
        "trainable": trainable_params,
    }


def train_epoch(model, optimizer, bridge, batch_size, device, epoch_num, num_batches, cfg, global_step_offset):
    """
    Train one epoch using RealNoiseMixtureVAE.
    CM1 uses labels: 3, 4, 5 (dp_prime, dp_composite, dp_hard_composite)
    """
    allowed_labels = [cfg["labels"]["dp_prime"], cfg["labels"]["dp_composite"], cfg["labels"]["dp_hard_composite"]]
    
    data_iter = StreamDataIterator(
        bridge,
        batch_size=batch_size,
        balance_labels=True,
        shuffle=False,
        allowed_labels=allowed_labels,
    )
    data_iter.start()
    try:
        return train_epoch_generic(
            model=model,
            optimizer=optimizer,
            data_iter=data_iter,
            device=device,
            num_batches=num_batches,
            epoch_num=epoch_num,
            gradient_clip=1.0,
            global_step_offset=global_step_offset,
            label_set=tuple(allowed_labels),
            debug_every=200,
            analyze_every=200,
        )
    finally:
        data_iter.stop()


def main():
    parser = argparse.ArgumentParser(
        description="RealNoiseMixtureVAE - CM1 Live Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 -m ml.train_cm1_live --epochs 5 --batches-per-epoch 100 --batch-size 32
  python3 -m ml.train_cm1_live --epochs 10 --batches-per-epoch 5000 --batch-size 64 --lr 0.0005
        """
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--batches-per-epoch", type=int, default=100, help="Batches per epoch")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    add_device_argument(parser)
    parser.add_argument("--checkpoint-dir", default="checkpoints_cm1", help="Checkpoint directory")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--bridge-host", default="127.0.0.1", help="Go Bridge host")
    parser.add_argument("--bridge-port", type=int, default=50051, help="Go Bridge port")
    args = parser.parse_args()

    atexit.register(lambda: train_state.shutdown("program exit"))

    logger.info("=" * 80)
    logger.info("🚀 RealNoiseMixtureVAE - CM1 (Complex Mode 1) LIVE TRAINING")
    logger.info("="*80)

    logger.info("Loading configuration...")
    cfg = json.load(open("config/config.json"))
    input_dim = cfg["model"]["input_dimension"]

    logger.info(f"Connecting to Go Bridge: {args.bridge_host}:{args.bridge_port}")
    bridge = create_and_connect_bridge(host=args.bridge_host, port=args.bridge_port)
    if not bridge:
        logger.error("Go Bridge connection failed!")
        return 1
    logger.info("Go Bridge connected successfully")

    device, _ = resolve_device(args.device)
    print_device_info(device)

    logger.info("Creating model...")
    model = CM1_ComplexAutoEncoder(cfg)

    # Lazy build on CPU then move
    _ = model(torch.randn(1, input_dim, device="cpu"))
    model = model.to(device)

    stats = get_model_stats(model)
    logger.info(f"Model built | Trainable params: {stats['trainable']:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_state.model = model
    train_state.checkpoint_dir = args.checkpoint_dir
    train_state.is_training = True
    # Setup separability loss config from config.json
    train_state.sep_cfg = cfg.get("sep_loss", {})

    logger.info("\n"+"="*80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Mode: CM1 (Labels: 3=dp_prime, 4=dp_composite, 5=dp_hard_composite)")
    logger.info(f"Epochs: {args.epochs} | Batches/epoch: {args.batches_per_epoch}")
    logger.info(f"Batch size: {args.batch_size} | Learning rate: {args.lr}")
    logger.info(f"Device: {device} | Checkpoint dir: {args.checkpoint_dir}")
    logger.info("=" * 80 + "\n")

    best_loss = float("inf")
    start_time = datetime.now()

    try:
        global_step = 0
        start_epoch = 0
        
        # ===== RESUME FROM CHECKPOINT =====
        success, epoch_loaded, loss_loaded, warnings_resume = train_state.load_checkpoint_for_resume(
            model=model,
            optimizer=optimizer,
            new_batch_size=args.batch_size,
            ckpt_path=None
        )
        
        if success:
            start_epoch = epoch_loaded + 1
            best_loss = loss_loaded
            logger.info(f"\n✓ RESUMING TRAINING")
            logger.info(f"  From epoch: {epoch_loaded} → Continue from epoch {start_epoch}")
            logger.info(f"  Best loss: {best_loss:.4f}")
            if warnings_resume:
                for w in warnings_resume:
                    logger.warning(f"    • {w}")
        else:
            logger.info("\n🆕 Starting fresh training (no checkpoint found)")
        
        for epoch in range(start_epoch, start_epoch + args.epochs):
            if not train_state.is_training:
                logger.warning("Training interrupted by user signal")
                break

            logger.info(f"\nStarting Epoch {epoch+1}/{args.epochs}")
            
            metrics = train_epoch(
                model=model,
                optimizer=optimizer,
                bridge=bridge,
                batch_size=args.batch_size,
                device=device,
                epoch_num=epoch + 1,
                num_batches=args.batches_per_epoch,
                cfg=cfg,
                global_step_offset=global_step,
            )
            global_step += args.batches_per_epoch

            scheduler.step()

            logger.info(f"Epoch {epoch+1} complete")
            logger.info(f"  Loss: {metrics['loss']:.4f} | NLL: {metrics['nll']:.4f} | KL: {metrics['kl']:.6f}")
            logger.info(f"  Learning rate: {scheduler.get_last_lr()[0]:.6f}")
            
            # Generate Turkish summary with metric trends
            sep_enabled = train_state.sep_cfg and train_state.sep_cfg.get("enabled", False)
            summary = _generate_epoch_summary(epoch + 1, metrics, train_state, sep_enabled=sep_enabled)
            if summary:
                import sys
                sys.stdout.write(summary)
                sys.stdout.flush()

            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                train_state.save_checkpoint(epoch + 1, best_loss, "(BEST)")
                logger.info(f"New best loss: {best_loss:.4f}")

            if (epoch + 1) % args.save_every == 0:
                train_state.save_checkpoint(epoch + 1, metrics["loss"], "(periodic)")

    except KeyboardInterrupt:
        logger.warning("Received Ctrl+C - shutting down gracefully")
        train_state.is_training = False
    finally:
        train_state.is_training = False
        bridge.disconnect()
        elapsed = datetime.now() - start_time
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETED")
        logger.info("="*80)
        logger.info(f"Best loss: {best_loss:.4f}")
        logger.info(f"Total time: {elapsed}")
        logger.info(f"Checkpoints saved to: {args.checkpoint_dir}")
        logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

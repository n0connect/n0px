#!/usr/bin/env python3
"""
ML Trainer Module
=================
Training loop, validation, metrics tracking, and checkpoint management.
Works with any model implementing the BaseModel interface.
"""

import logging
import numpy as np
import time
from typing import Optional, Dict, Any, List
from datetime import datetime

from . import config
from . import utils
from . import core
from . import models
from . import data

logger = utils.setup_logger(__name__, config.LOG_LEVEL)


# ============================================================
#  METRICS TRACKER
# ============================================================
class MetricsTracker:
    """Track training metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {
            "loss": [],
            "accuracy": [],
            "val_accuracy": [],
            "epoch_time": [],
        }
    
    def record(self, metric_name: str, value: float) -> None:
        """Record a metric value."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
    
    def get_last(self, metric_name: str) -> Optional[float]:
        """Get last recorded value."""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return self.metrics[metric_name][-1]
        return None
    
    def get_average(self, metric_name: str, last_n: int = None) -> float:
        """Get average value."""
        if metric_name not in self.metrics:
            return 0.0
        
        values = self.metrics[metric_name]
        if not values:
            return 0.0
        
        if last_n:
            values = values[-last_n:]
        
        return float(np.mean(values))
    
    def __repr__(self) -> str:
        return f"MetricsTracker({len(self.metrics)} metrics)"


# ============================================================
#  TRAINER
# ============================================================
class Trainer:
    """
    Main training loop manager.
    Handles model training, validation, and checkpoint saving.
    """
    
    def __init__(
        self,
        model,  # BaseModel or any model with forward/predict methods
        data_iterator: data.StreamDataIterator,
        learning_rate: float = config.MODEL_LEARNING_RATE,
        checkpoint_dir: str = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model instance (must have forward/predict methods)
            data_iterator: Data iterator (streaming batches)
            learning_rate: Learning rate
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.data_iter = data_iterator
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        
        self.metrics = MetricsTracker()
        self.best_accuracy = 0.0
        self.start_time = None
        self.current_epoch = 0
        
        logger.info(
            f"Trainer initialized with model: "
            f"lr={learning_rate}, "
            f"checkpoint_dir={checkpoint_dir}"
        )
    
    def train_epoch(self, num_batches: int = None) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            num_batches: Number of batches per epoch (default: from config)
        
        Returns:
            Dictionary with epoch metrics
        """
        if num_batches is None:
            num_batches = 100  # Reasonable default
        
        self.model.is_training = True
        
        epoch_start = time.time()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx in range(num_batches):
            batch = self.data_iter.get_batch()
            if batch is None:
                logger.warning(f"No batch at iteration {batch_idx}")
                break
            
            # Forward pass
            logits = self.model.forward(batch.features)
            
            # Compute loss (simple cross-entropy approximation)
            predictions = np.argmax(logits, axis=1)
            batch_loss = self._compute_loss(logits, batch.labels)
            batch_accuracy = np.mean(predictions == batch.labels)
            
            # Backward pass (simple SGD-like update)
            # For a proper implementation, use PyTorch
            
            total_loss += batch_loss
            correct += np.sum(predictions == batch.labels)
            total += len(batch.labels)
            
            if (batch_idx + 1) % 10 == 0:
                logger.debug(
                    f"  Batch {batch_idx + 1}/{num_batches}: "
                    f"loss={batch_loss:.4f}, acc={batch_accuracy:.4f}"
                )
        
        epoch_time = time.time() - epoch_start
        epoch_accuracy = correct / total if total > 0 else 0.0
        epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Record metrics
        self.metrics.record("loss", epoch_loss)
        self.metrics.record("accuracy", epoch_accuracy)
        self.metrics.record("epoch_time", epoch_time)
        
        self.current_epoch += 1
        
        return {
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
            "time": epoch_time,
        }
    
    def validate(self, num_batches: int = 20) -> Dict[str, float]:
        """
        Run validation.
        
        Args:
            num_batches: Number of validation batches
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.is_training = False
        
        correct = 0
        total = 0
        
        for _ in range(num_batches):
            batch = self.data_iter.get_batch()
            if batch is None:
                break
            
            predictions, probs = self.model.predict(batch.features)
            
            correct += np.sum(predictions == batch.labels)
            total += len(batch.labels)
        
        val_accuracy = correct / total if total > 0 else 0.0
        self.metrics.record("val_accuracy", val_accuracy)
        
        return {"accuracy": val_accuracy}
    
    def _compute_loss(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute cross-entropy loss.
        
        Args:
            logits: Model output (batch_size, num_classes)
            labels: True labels (batch_size,)
        
        Returns:
            Scalar loss
        """
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Cross-entropy
        batch_size = len(labels)
        correct_log_probs = -np.log(probs[np.arange(batch_size), labels] + 1e-8)
        loss = np.mean(correct_log_probs)
        
        return float(loss)
    
    def save_checkpoint(self, suffix: str = "") -> str:
        """
        Save model checkpoint.
        
        Args:
            suffix: Suffix for checkpoint filename
        
        Returns:
            Path to saved checkpoint
        """
        if not self.checkpoint_dir:
            logger.warning("No checkpoint directory specified")
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_epoch{self.current_epoch}_{timestamp}{suffix}.pkl"
        
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        self.model.save(filepath)
        
        logger.info(f"Checkpoint saved: {filepath}")
        return filepath
    
    def train(
        self,
        num_epochs: int = None,
        batches_per_epoch: int = None,
        val_every: int = 5,
        save_every: int = None,
    ) -> Dict[str, Any]:
        """
        Full training loop.
        
        Args:
            num_epochs: Total epochs (default: config.EPOCHS)
            batches_per_epoch: Batches per epoch (default: 100)
            val_every: Validate every N epochs
            save_every: Save checkpoint every N epochs (default: config.CHECKPOINT_SAVE_INTERVAL)
        
        Returns:
            Dictionary with training metrics
        """
        if num_epochs is None:
            num_epochs = config.EPOCHS
        if batches_per_epoch is None:
            batches_per_epoch = 100
        if save_every is None:
            save_every = config.CHECKPOINT_SAVE_INTERVAL
        
        utils.print_banner("TRAINING PRIME CLASSIFIER")
        utils.print_section("Training Configuration")
        
        logger.info(
            f"Starting training: "
            f"epochs={num_epochs}, "
            f"batches/epoch={batches_per_epoch}, "
            f"val_every={val_every}, "
            f"save_every={save_every}"
        )
        
        self.start_time = time.time()
        self.data_iter.start()
        
        try:
            for epoch in range(num_epochs):
                # Training
                train_metrics = self.train_epoch(num_batches=batches_per_epoch)
                
                # Validation
                if (epoch + 1) % val_every == 0:
                    val_metrics = self.validate(num_batches=20)
                    logger.info(
                        f"Epoch {epoch + 1}/{num_epochs}: "
                        f"loss={train_metrics['loss']:.4f}, "
                        f"acc={train_metrics['accuracy']:.4f}, "
                        f"val_acc={val_metrics['accuracy']:.4f}, "
                        f"time={train_metrics['time']:.2f}s"
                    )
                else:
                    logger.info(
                        f"Epoch {epoch + 1}/{num_epochs}: "
                        f"loss={train_metrics['loss']:.4f}, "
                        f"acc={train_metrics['accuracy']:.4f}, "
                        f"time={train_metrics['time']:.2f}s"
                    )
                
                # Checkpointing
                if (epoch + 1) % save_every == 0:
                    self.save_checkpoint(suffix="_auto")
        
        finally:
            self.data_iter.stop()
        
        total_time = time.time() - self.start_time
        
        utils.print_section("Training Complete")
        logger.info(f"Total training time: {total_time:.2f}s")
        
        return {
            "total_epochs": self.current_epoch,
            "total_time": total_time,
            "metrics": self.metrics.metrics,
            "best_accuracy": max(self.metrics.metrics.get("val_accuracy", [0])),
        }


# ============================================================
#  INFERENCE ENGINE
# ============================================================
class InferenceEngine:
    """
    Runtime inference on streaming data.
    Provides prediction interface for production use.
    """
    
    def __init__(self, model):
        """
        Initialize inference engine.
        
        Args:
            model: Trained model (must have predict() method)
        """
        self.model = model
        if hasattr(self.model, 'is_training'):
            self.model.is_training = False
        self.predictions_made = 0
        
        logger.info("InferenceEngine initialized")
    
    def predict(self, packet: core.DataPacket) -> Dict[str, Any]:
        """
        Make prediction on single packet.
        
        Args:
            packet: DataPacket from Go Bridge
        
        Returns:
            Dictionary with prediction results
        """
        features = np.array(packet.input_vector, dtype=np.float32).reshape(1, -1)
        
        predicted_class, confidence = self.model.predict_single(features)
        
        result = {
            "predicted_class": int(predicted_class),
            "predicted_label": utils.format_label(predicted_class),
            "confidence": float(confidence),
            "true_label": utils.format_label(packet.label),
            "is_correct": predicted_class == packet.label,
        }
        
        self.predictions_made += 1
        
        return result
    
    def predict_batch(self, packets: List[core.DataPacket]) -> List[Dict]:
        """
        Make predictions on batch.
        
        Args:
            packets: List of DataPackets
        
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(p) for p in packets]

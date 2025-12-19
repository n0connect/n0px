#!/usr/bin/env python3
"""
ML Data Module
==============
Real-time data streaming, batching, and preprocessing.
Handles communication with Go Bridge to fetch live data for ML training.
"""

import logging
import queue
import threading
import numpy as np
from typing import List, Iterator, Optional, Tuple
from dataclasses import dataclass

from . import config
from . import utils
from . import core

logger = utils.setup_logger(__name__, config.LOG_LEVEL)


# ============================================================
#  BATCH DATA STRUCTURE
# ============================================================
@dataclass
class DataBatch:
    """
    Represents a batch of samples ready for model training/inference.
    
    Attributes:
        features: NumPy array of shape (batch_size, feature_dim) where feature_dim = config.DEFAULT_BITS
        labels: NumPy array of shape (batch_size,) with class indices
        raw_bytes: List of raw byte samples
        is_synthetic: List of boolean flags for synthetic data
    """
    features: np.ndarray  # (batch_size, DEFAULT_BITS) - corresponds to prime_bits from config
    labels: np.ndarray    # (batch_size,)
    raw_bytes: List[bytes] = None
    is_synthetic: List[bool] = None
    
    def size(self) -> int:
        """Get batch size."""
        return len(self.features)
    
    def __repr__(self) -> str:
        return (
            f"DataBatch("
            f"size={self.size()}, "
            f"shape={self.features.shape}, "
            f"label_dist={dict(zip(*np.unique(self.labels, return_counts=True)))}"
            f")"
        )


# ============================================================
#  DATA ITERATOR
# ============================================================
class StreamDataIterator:
    """
    Iterator that yields batches of data from Go Bridge stream.
    Handles batching, label balancing, and preprocessing.
    """
    
    def __init__(
        self,
        bridge: core.GRPCBridge,
        batch_size: int = config.BATCH_SIZE,
        balance_labels: bool = True,
        shuffle: bool = True,
        allowed_labels: Optional[set] = None,
    ):
        """
        Initialize data iterator.
        
        Args:
            bridge: Connected GRPCBridge instance
            batch_size: Batch size
            balance_labels: If True, try to balance class distribution
            shuffle: If True, shuffle batch order
            allowed_labels: If provided, only batches with labels in this set are yielded.
                           None = no filtering. Ex: {0,1,2} or {3,4,5}
        """
        self.bridge = bridge
        self.batch_size = batch_size
        self.balance_labels = balance_labels
        self.shuffle = shuffle
        self.allowed_labels = allowed_labels
        
        # Buffering
        self._buffer: queue.Queue[core.DataPacket] = queue.Queue(
            maxsize=batch_size * 2
        )
        self._fetcher_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        
        # Statistics
        self.packets_fetched = 0
        self.batches_yielded = 0
        self.label_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self.filtered_label_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        filter_info = f"allowed_labels={allowed_labels}" if allowed_labels else "no label filtering"
        logger.info(
            f"StreamDataIterator initialized: "
            f"batch_size={batch_size}, "
            f"balance_labels={balance_labels}, "
            f"shuffle={shuffle}, "
            f"{filter_info}"
        )
    
    def start(self) -> None:
        """Start background fetching."""
        if self._running:
            logger.warning("Already running")
            return
        
        self._stop_event.clear()
        self._running = True
        
        self._fetcher_thread = threading.Thread(
            target=self._fetch_loop,
            daemon=True
        )
        self._fetcher_thread.start()
        logger.info("StreamDataIterator started")
    
    def stop(self, timeout: float = 5.0) -> None:
        """Stop background fetching."""
        if not self._running:
            return
        
        self._stop_event.set()
        if self._fetcher_thread:
            self._fetcher_thread.join(timeout=timeout)
        
        self._running = False
        logger.info("StreamDataIterator stopped")
    
    def _fetch_loop(self) -> None:
        """Background loop that fetches packets from bridge."""
        while not self._stop_event.is_set():
            try:
                pkt = self.bridge.get_packet()
                if pkt is None:
                    logger.warning("Bridge returned None, stopping fetcher")
                    break
                
                # Validate packet
                if not pkt.validate():
                    logger.warning("Invalid packet, skipping")
                    continue
                
                # Non-blocking put
                try:
                    self._buffer.put_nowait(pkt)
                    self.packets_fetched += 1
                except queue.Full:
                    logger.debug("Buffer full, dropping oldest packet")
                    try:
                        self._buffer.get_nowait()
                        self._buffer.put_nowait(pkt)
                    except queue.Empty:
                        pass
            
            except Exception as e:
                logger.error(f"Fetch loop error: {e}")
    
    def _get_balanced_batch_quota(self) -> List[core.DataPacket]:
        """
        PATCH-E: Build batch with balanced quotas per class.
        Distributes batch_size slots equally across available classes.
        """
        packets = []
        
        with self._class_buffers_lock:
            if not self._class_buffers:
                return []
            
            active_classes = [c for c, buf in self._class_buffers.items() if buf]
            if not active_classes:
                return []
            
            # Calculate quota per class (round-robin assignment)
            n_classes = len(active_classes)
            quota_per_class = self.batch_size // n_classes
            remainder = self.batch_size % n_classes
            
            # Assign quotas
            class_quotas = {}
            for i, cls_id in enumerate(active_classes):
                quota = quota_per_class + (1 if i < remainder else 0)
                class_quotas[cls_id] = quota
            
            # Fetch from each class buffer up to its quota
            for cls_id, quota in class_quotas.items():
                buf = self._class_buffers[cls_id]
                for _ in range(min(quota, len(buf))):
                    packets.append(buf.pop(0))
        
        return packets
    
    def _get_batch_from_buffer(self, size: int) -> List[core.DataPacket]:
        """Get packets from buffer, waiting if needed."""
        packets = []
        timeout = 5.0
        
        for _ in range(size):
            try:
                pkt = self._buffer.get(timeout=timeout)
                packets.append(pkt)
            except queue.Empty:
                logger.warning(f"Timeout waiting for packets (got {len(packets)}/{size})")
                break
        
        return packets
    
    def _packets_to_batch(self, packets: List[core.DataPacket]) -> DataBatch:
        """Convert list of packets to DataBatch, optionally filtering by allowed_labels."""
        # Update statistics for all packets received
        for p in packets:
            self.label_counts[int(p.label)] += 1
        
        # Filter packets by allowed_labels if specified
        original_count = len(packets)
        if self.allowed_labels is not None:
            packets = [p for p in packets if p.label in self.allowed_labels]
            
            if len(packets) < original_count:
                logger.debug(
                    f"Label filter: {len(packets)}/{original_count} packets retained "
                    f"(allowed_labels={self.allowed_labels})"
                )
        
        # Build batch from (potentially filtered) packets
        features = np.array([p.input_vector for p in packets], dtype=np.float32)
        labels = np.array([p.label for p in packets], dtype=np.int64)
        raw_bytes = [p.raw_bytes for p in packets]
        is_synthetic = [p.is_synthetic for p in packets]
        
        # Update filtered statistics
        for label in labels:
            self.filtered_label_counts[int(label)] += 1
        
        return DataBatch(
            features=features,
            labels=labels,
            raw_bytes=raw_bytes,
            is_synthetic=is_synthetic,
        )
    
    def __iter__(self) -> Iterator[DataBatch]:
        """Iterate over batches indefinitely."""
        self.start()
        
        while self._running:
            packets = self._get_batch_from_buffer(self.batch_size)
            
            if not packets:
                continue
            
            # CRITICAL: No shuffling - preserve C++ Core generation order
            # C++ CSPRNG generates in perfect distribution, shuffling breaks this
            # Each label appears in deterministic pattern from C++ pipeline
            
            batch = self._packets_to_batch(packets)
            self.batches_yielded += 1
            
            yield batch
    
    def get_batch(self) -> Optional[DataBatch]:
        """Get single batch (non-iterator method)."""
        packets = self._get_batch_from_buffer(self.batch_size)
        if not packets:
            return None
        return self._packets_to_batch(packets)
    
    def stats(self) -> dict:
        """Get iterator statistics (includes raw and filtered label distributions)."""
        return {
            "packets_fetched": self.packets_fetched,
            "batches_yielded": self.batches_yielded,
            "label_distribution_raw": self.label_counts,
            "label_distribution_filtered": self.filtered_label_counts,
            "allowed_labels": self.allowed_labels,
            "buffer_size": self._buffer.qsize(),
            "is_running": self._running,
        }


# ============================================================
#  TRAIN/VAL SPLIT
# ============================================================
class DataSplitter:
    """
    Splits streaming data into training and validation sets.
    Maintains class balance if requested.
    """
    
    def __init__(
        self,
        bridge: core.GRPCBridge,
        batch_size: int = config.BATCH_SIZE,
        val_split: float = config.VALIDATION_SPLIT,
        balance_labels: bool = True,
    ):
        """
        Initialize splitter.
        
        Args:
            bridge: Connected GRPCBridge
            batch_size: Batch size
            val_split: Validation split ratio (0.0-1.0)
            balance_labels: Balance class distribution
        """
        self.bridge = bridge
        self.batch_size = batch_size
        self.val_split = val_split
        self.balance_labels = balance_labels
        
        self.train_iter = StreamDataIterator(
            bridge=bridge,
            batch_size=batch_size,
            balance_labels=balance_labels,
        )
        
        logger.info(
            f"DataSplitter initialized: "
            f"batch_size={batch_size}, "
            f"val_split={val_split}"
        )
    
    def get_train_batch(self) -> Optional[DataBatch]:
        """Get training batch."""
        return self.train_iter.get_batch()
    
    def train_iterator(self, max_batches: Optional[int] = None) -> Iterator[DataBatch]:
        """
        Get training iterator.
        
        Args:
            max_batches: Stop after N batches (None = unlimited)
        
        Yields:
            DataBatch instances
        """
        count = 0
        for batch in self.train_iter:
            if max_batches and count >= max_batches:
                break
            yield batch
            count += 1
    
    def validation_batch(self) -> Optional[DataBatch]:
        """
        Get validation batch.
        Currently same as train_batch, but can be overridden for cross-validation.
        """
        return self.get_train_batch()


# ============================================================
#  PREPROCESSING UTILITIES
# ============================================================
class Preprocessor:
    """Data preprocessing and normalization."""
    
    def __init__(
        self,
        normalize: bool = True,
        standardize: bool = True,
    ):
        """
        Initialize preprocessor.
        
        Args:
            normalize: Normalize to [0, 1]
            standardize: Standardize to zero mean, unit variance
        """
        self.normalize = normalize
        self.standardize = standardize
        
        self.mean = None
        self.std = None
        self.min_val = None
        self.max_val = None
        
        logger.info(
            f"Preprocessor initialized: "
            f"normalize={normalize}, standardize={standardize}"
        )
    
    def fit(self, features: np.ndarray) -> None:
        """
        Fit preprocessor on training data.
        
        Args:
            features: Training features (N, D)
        """
        if self.normalize:
            self.min_val = np.min(features)
            self.max_val = np.max(features)
        
        if self.standardize:
            self.mean = np.mean(features, axis=0)
            self.std = np.std(features, axis=0)
            # Prevent division by zero
            self.std[self.std == 0] = 1.0
        
        logger.info("Preprocessor fitted on training data")
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing.
        
        Args:
            features: Input features (N, D)
        
        Returns:
            Preprocessed features
        """
        result = features.copy()
        
        if self.normalize and self.min_val is not None:
            result = (result - self.min_val) / (self.max_val - self.min_val + 1e-8)
        
        if self.standardize and self.mean is not None:
            result = (result - self.mean) / (self.std + 1e-8)
        
        return result
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one call."""
        self.fit(features)
        return self.transform(features)

#!/usr/bin/env python3
"""
ML Core Module
==============
Go Bridge'e gRPC bağlantısı, stream yönetimi, veri alınması.
Bu modül C++ -> Go -> Python haberleşmesinin Python ucunun temelini oluşturur.
"""

import logging
import time
import grpc
import queue
import threading
import sys
from pathlib import Path
from typing import Optional, Iterator, Tuple, List
from dataclasses import dataclass

# Suppress gRPC verbose logging
logging.getLogger("grpc").setLevel(logging.WARNING)
logging.getLogger("grpc._cython").setLevel(logging.WARNING)

# Try absolute imports first (when run as module), then relative
try:
    from . import config
    from . import utils
except ImportError:
    # Fallback for direct script execution
    import config
    import utils

# Add bridge/pb to path for protobuf imports
sys.path.insert(0, str(Path(__file__).parent.parent / "bridge" / "pb"))

# Try to import protobuf - generate if needed
try:
    import prime_bridge_pb2 as pb
    import prime_bridge_pb2_grpc as pb_grpc
except ImportError as e:
    print(f"Protobuf import failed: {e}")
    print("Ensure protobuf is compiled: make proto-sync")
    raise

logger = utils.setup_logger(__name__, config.LOG_LEVEL)


# ============================================================
#  DATA STRUCTURES
# ============================================================
@dataclass
class DataPacket:
    """
    Represents a single data packet from Go Bridge.
    
    Attributes:
        raw_bytes: 32-byte raw integer (256-bit number)
        input_vector: List of 256 float32 values (with noise)
        label: 0=COMPOSITE, 1=PRIME, 2=HARD, 3=DP_PRIME, 4=DP_COMPOSITE, 5=DP_HARD
        is_synthetic: Whether data is synthetic (generated)
    """
    raw_bytes: bytes
    input_vector: List[float]
    label: int
    is_synthetic: bool = False
    
    def __repr__(self) -> str:
        label_name = utils.format_label(self.label)
        return (
            f"DataPacket("
            f"label={label_name}, "
            f"raw={utils.format_bytes_hex(self.raw_bytes, 8)}, "
            f"vector_len={len(self.input_vector)}"
            f")"
        )
    
    def validate(self) -> bool:
        """
        Validate packet structure and contents.
        Checks: raw_bytes size, float_vector length and values, label range.
        """
        if not utils.validate_raw_bytes(self.raw_bytes):
            return False
        if not utils.validate_float_vector(self.input_vector):
            return False
        if not utils.validate_label(self.label):
            return False
        return True
    
    def compute_hashes(self) -> Tuple[int, int]:
        """Compute BLAKE2s hashes for verification."""
        return utils.verify_packet_hashes(self.raw_bytes, self.input_vector)


# ============================================================
#  GRPC BRIDGE CLIENT
# ============================================================
class GRPCBridge:
    """
    Manages gRPC connection to Go Bridge microservice.
    Handles stream management, reconnection, and packet buffering.
    """
    
    def __init__(
        self,
        host: str = config.GRPC_SERVER_HOST,
        port: int = config.GRPC_SERVER_PORT,
        batch_size: int = config.DEFAULT_BATCH_SIZE,
        mixing_ratio: float = config.DEFAULT_MIXING_RATIO,
    ):
        """
        Initialize gRPC Bridge client.
        
        Args:
            host: Go Bridge server hostname
            port: Go Bridge server port
            batch_size: Requested batch size from Go
            mixing_ratio: Ratio of primes to composites (0.0-1.0)
        """
        self.host = host
        self.port = port
        self.address = f"{host}:{port}"
        self.batch_size = batch_size
        self.mixing_ratio = mixing_ratio
        
        self.channel: Optional[grpc.Channel] = None
        self.stub: Optional[pb_grpc.DataProviderStub] = None
        self.stream: Optional[Iterator] = None
        
        self.connected = False
        self._lock = threading.Lock()
        
        logger.info(
            f"GRPCBridge initialized: {self.address} "
            f"(batch_size={batch_size}, mixing_ratio={mixing_ratio})"
        )
    
    def connect(self, timeout: int = config.GRPC_STREAM_TIMEOUT) -> bool:
        """
        Establish connection to Go Bridge.
        
        Args:
            timeout: Connection timeout in seconds
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._lock:
                if self.connected:
                    logger.warning("Already connected, skipping reconnect")
                    return True
                
                logger.info(f"Connecting to gRPC server at {self.address}...")
                
                # Create channel with timeout
                self.channel = grpc.insecure_channel(
                    self.address,
                    options=[
                        ("grpc.keepalive_time_ms", 10000),
                        ("grpc.keepalive_timeout_ms", 5000),
                    ]
                )
                
                self.stub = pb_grpc.DataProviderStub(self.channel)
                
                # Try to connect (this doesn't actually connect until first call)
                # We'll test with a small stream request
                logger.debug("Testing gRPC connection with dummy request...")
                
                # Create stream config
                stream_config = pb.StreamConfig(
                    batch_size=self.batch_size,
                    mixing_ratio=self.mixing_ratio,
                    mixed_mode=True,
                )
                
                # Start stream
                self.stream = self.stub.StreamData(stream_config)
                
                # Try to get first packet (with timeout)
                try:
                    # This is blocking, so we verify connection works
                    first_pkt = next(self.stream)
                    logger.info(
                        f"gRPC connection successful! "
                        f"Received first packet: {utils.format_label(first_pkt.label)}"
                    )
                    self.connected = True
                    return True
                except StopIteration:
                    logger.error("gRPC stream ended immediately (no data)")
                    return False
        
        except grpc.RpcError as e:
            logger.error(f"gRPC RPC Error: {e.code()}: {e.details()}")
            return False
        except Exception as e:
            logger.error(f"gRPC connection error: {e}")
            return False
    
    def disconnect(self) -> None:
        """Close gRPC connection."""
        with self._lock:
            if self.channel:
                self.channel.close()
                self.connected = False
                logger.info("gRPC connection closed")
    
    def get_packet(self) -> Optional[DataPacket]:
        """
        Retrieve single packet from stream.
        Converts protobuf message to DataPacket.
        
        Returns:
            DataPacket or None if stream ends
        """
        if not self.connected or not self.stream:
            logger.error("Not connected to gRPC server")
            return None
        
        try:
            pb_packet = next(self.stream)
            
            # Convert protobuf to DataPacket
            packet = DataPacket(
                raw_bytes=pb_packet.raw_bytes,
                input_vector=list(pb_packet.input_vector),
                label=pb_packet.label,
                is_synthetic=pb_packet.is_synthetic,
            )
            
            return packet
        
        except StopIteration:
            logger.warning("gRPC stream ended")
            self.connected = False
            return None
        except Exception as e:
            logger.error(f"Error retrieving packet: {e}")
            return None
    
    def get_packets(self, count: int) -> List[DataPacket]:
        """
        Retrieve multiple packets from stream.
        
        Args:
            count: Number of packets to retrieve
        
        Returns:
            List of DataPackets
        """
        packets = []
        for _ in range(count):
            pkt = self.get_packet()
            if pkt is None:
                break
            packets.append(pkt)
        return packets
    
    def stream_packets(self) -> Iterator[DataPacket]:
        """
        Generator that yields packets indefinitely.
        Useful for training loops.
        """
        while self.connected:
            pkt = self.get_packet()
            if pkt is None:
                break
            yield pkt


# ============================================================
#  PACKET STREAM WITH BUFFERING
# ============================================================
class BufferedStreamReader:
    """
    High-level stream reader with buffering and background fetching.
    Decouples network I/O from data consumption for better performance.
    """
    
    def __init__(
        self,
        bridge: GRPCBridge,
        buffer_size: int = config.DATA_BUFFER_SIZE,
    ):
        """
        Initialize buffered stream reader.
        
        Args:
            bridge: GRPCBridge instance (must be connected)
            buffer_size: Internal queue size
        """
        self.bridge = bridge
        self.buffer_size = buffer_size
        self.buffer: queue.Queue[DataPacket] = queue.Queue(maxsize=buffer_size)
        
        self._fetcher_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        
        logger.info(f"BufferedStreamReader initialized (buffer_size={buffer_size})")
    
    def start(self) -> None:
        """Start background fetcher thread."""
        if self._running:
            logger.warning("Already running")
            return
        
        self._stop_event.clear()
        self._running = True
        
        self._fetcher_thread = threading.Thread(
            target=self._fetcher_loop,
            daemon=True
        )
        self._fetcher_thread.start()
        logger.info("BufferedStreamReader started")
    
    def stop(self, timeout: float = 5.0) -> None:
        """Stop background fetcher thread."""
        if not self._running:
            return
        
        self._stop_event.set()
        if self._fetcher_thread:
            self._fetcher_thread.join(timeout=timeout)
        
        self._running = False
        logger.info("BufferedStreamReader stopped")
    
    def _fetcher_loop(self) -> None:
        """Background loop that fetches packets and fills buffer."""
        while not self._stop_event.is_set():
            try:
                pkt = self.bridge.get_packet()
                if pkt is None:
                    logger.warning("Stream ended, stopping fetcher")
                    break
                
                # Non-blocking put (skip if buffer full to prevent blocking)
                try:
                    self.buffer.put_nowait(pkt)
                except queue.Full:
                    logger.warning("Buffer full, dropping packet")
            
            except Exception as e:
                logger.error(f"Fetcher loop error: {e}")
                time.sleep(config.GRPC_STREAM_RETRY_INTERVAL)
    
    def get_packet(self, timeout: float = 5.0) -> Optional[DataPacket]:
        """
        Get single packet from buffer.
        
        Args:
            timeout: Wait timeout in seconds
        
        Returns:
            DataPacket or None if timeout/error
        """
        try:
            return self.buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_packets(self, count: int, timeout: float = 5.0) -> List[DataPacket]:
        """
        Get multiple packets from buffer.
        
        Args:
            count: Number of packets
            timeout: Per-packet timeout
        
        Returns:
            List of packets (may have fewer than count if timeout)
        """
        packets = []
        for _ in range(count):
            pkt = self.get_packet(timeout=timeout)
            if pkt is None:
                break
            packets.append(pkt)
        return packets
    
    def buffer_status(self) -> dict:
        """Get buffer status."""
        return {
            "size": self.buffer.qsize(),
            "max_size": self.buffer_size,
            "is_full": self.buffer.full(),
            "is_empty": self.buffer.empty(),
            "utilization": self.buffer.qsize() / self.buffer_size,
        }


# ============================================================
#  INITIALIZATION HELPER
# ============================================================
def create_and_connect_bridge(
    host: str = config.GRPC_SERVER_HOST,
    port: int = config.GRPC_SERVER_PORT,
    max_retries: int = config.GRPC_MAX_RETRIES,
    retry_interval: float = config.GRPC_STREAM_RETRY_INTERVAL,
) -> Optional[GRPCBridge]:
    """
    Helper function to create bridge and attempt connection with retries.
    
    Args:
        host: Server hostname
        port: Server port
        max_retries: Number of connection attempts
        retry_interval: Seconds between retries
    
    Returns:
        Connected GRPCBridge or None if all retries failed
    """
    bridge = GRPCBridge(host=host, port=port)
    
    for attempt in range(max_retries):
        logger.info(f"Connection attempt {attempt + 1}/{max_retries}...")
        
        if bridge.connect():
            return bridge
        
        if attempt < max_retries - 1:
            logger.info(f"Retrying in {retry_interval}s...")
            time.sleep(retry_interval)
    
    logger.error(f"Failed to connect after {max_retries} attempts")
    return None

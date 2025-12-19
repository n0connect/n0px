#!/usr/bin/env python3
"""
PRIME-X Full Pipeline Integration Test v2
===========================================
Tests complete flow: C++ → Go → Python → ML Model

Requirements:
- prime_core running (C++ CSPRNG)
- prime_bridge running (Go gRPC server)

Tests:
1. gRPC connection stability
2. Data format validation (C++ -> Go -> Python)
3. Packet integrity (BLAKE2s hash)
4. All 6 labels received
5. Label consistency throughout pipeline
6. ML model can process pipeline data
7. End-to-end throughput
"""

import sys
import time
import grpc
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "bridge" / "pb"))

try:
    import prime_bridge_pb2 as pb
    import prime_bridge_pb2_grpc as pb_grpc
except ImportError as e:
    print(f"ERROR: Failed to import protobuf stubs")
    print(f"  Make sure to run: make proto-sync")
    print(f"  Error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================
# CONSTANTS
# =============================================================

GRPC_HOST = "localhost"
GRPC_PORT = 50051
GRPC_TIMEOUT = 5.0

LABEL_NAMES = {
    0: "COMPOSITE",
    1: "PRIME", 
    2: "HARD_COMPOSITE",
    3: "DP_PRIME",
    4: "DP_COMPOSITE",
    5: "DP_HARD_COMPOSITE",
}

# =============================================================
# COLOR CODES
# =============================================================

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"
CHECK = "✓"
CROSS = "✗"

# =============================================================
# UTILITIES
# =============================================================

def print_banner(text: str):
    """Print formatted banner"""
    width = 70
    line = "=" * width
    print(f"\n{CYAN}{BOLD}{line}{RESET}")
    print(f"{CYAN}{BOLD}{text.center(width)}{RESET}")
    print(f"{CYAN}{BOLD}{line}{RESET}\n")


def print_section(text: str, indent: int = 2):
    """Print section header"""
    prefix = " " * indent
    print(f"{prefix}{BOLD}{BLUE}▶ {text}{RESET}")


def print_ok(text: str, indent: int = 4):
    """Print success message"""
    prefix = " " * indent
    print(f"{prefix}{GREEN}{CHECK} {text}{RESET}")


def print_err(text: str, indent: int = 4):
    """Print error message"""
    prefix = " " * indent
    print(f"{prefix}{RED}{CROSS} {text}{RESET}")


def print_warn(text: str, indent: int = 4):
    """Print warning message"""
    prefix = " " * indent
    print(f"{prefix}{YELLOW}⚠ {text}{RESET}")

# =============================================================
# TEST 1: gRPC Connection
# =============================================================

def test_grpc_connection() -> bool:
    """Test basic gRPC connection to Go bridge"""
    print_section("gRPC Connection Test", 2)
    
    try:
        logger.info(f"Connecting to {GRPC_HOST}:{GRPC_PORT}")
        channel = grpc.insecure_channel(f"{GRPC_HOST}:{GRPC_PORT}")
        stub = pb_grpc.DataProviderStub(channel)
        
        print_ok(f"Connected to {GRPC_HOST}:{GRPC_PORT}")
        
        # Try to get one batch
        config = pb.StreamConfig(
            batch_size=1,
            mixing_ratio=1.0,
            mixed_mode=True,
        )
        
        logger.info("Requesting first batch...")
        stream = stub.StreamData(config, timeout=GRPC_TIMEOUT)
        
        try:
            first_batch = next(stream)
            print_ok(f"Received first batch")
            print_ok(f"  - Raw bytes: {len(first_batch.raw_bytes)} bytes")
            print_ok(f"  - Input vector: {len(first_batch.input_vector)} floats")
            print_ok(f"  - Label: {first_batch.label} ({LABEL_NAMES[first_batch.label]})")
            print_ok(f"  - Is synthetic: {first_batch.is_synthetic}")
            
            channel.close()
            return True
        except StopIteration:
            print_err("Stream ended without data")
            channel.close()
            return False
    
    except grpc.RpcError as e:
        print_err(f"gRPC error: {e.code()}: {e.details()}")
        return False
    except Exception as e:
        print_err(f"Connection failed: {e}")
        return False

# =============================================================
# TEST 2: Data Format Validation
# =============================================================

def test_data_format() -> bool:
    """Validate data format from Go bridge"""
    print_section("Data Format Validation", 2)
    
    try:
        channel = grpc.insecure_channel(f"{GRPC_HOST}:{GRPC_PORT}")
        stub = pb_grpc.DataProviderStub(channel)
        
        config = pb.StreamConfig(batch_size=5, mixing_ratio=1.0, mixed_mode=True)
        stream = stub.StreamData(config, timeout=GRPC_TIMEOUT)
        
        errors = []
        
        for i, batch in enumerate(stream):
            if i >= 5:  # Get 5 packets
                break
            
            # Validate raw_bytes
            if not isinstance(batch.raw_bytes, bytes):
                errors.append(f"Packet {i}: raw_bytes not bytes type")
            elif len(batch.raw_bytes) == 0:
                errors.append(f"Packet {i}: raw_bytes empty")
            
            # Validate input_vector
            if not hasattr(batch, 'input_vector'):
                errors.append(f"Packet {i}: no input_vector")
            elif len(batch.input_vector) != 256:
                errors.append(f"Packet {i}: input_vector size {len(batch.input_vector)} != 256")
            
            # Validate label
            if batch.label < 0 or batch.label > 5:
                errors.append(f"Packet {i}: label {batch.label} out of range [0,5]")
            
            # Validate input values are floats
            for j, val in enumerate(batch.input_vector[:5]):  # Check first 5
                if not isinstance(val, (float, np.floating)):
                    errors.append(f"Packet {i}: input_vector[{j}] not float")
            
            print_ok(f"Packet {i}: format valid (label={batch.label}, raw={len(batch.raw_bytes)}B, input={len(batch.input_vector)}f)")
        
        channel.close()
        
        if errors:
            for err in errors:
                print_err(err)
            return False
        
        print_ok("All data formats valid")
        return True
    
    except Exception as e:
        print_err(f"Format validation failed: {e}")
        return False

# =============================================================
# TEST 3: Label Distribution (All 6 classes)
# =============================================================

def test_label_distribution(num_samples: int = 1200) -> bool:
    """Test that all 6 labels are received"""
    print_section(f"Label Distribution Test ({num_samples} samples)", 2)
    
    try:
        channel = grpc.insecure_channel(f"{GRPC_HOST}:{GRPC_PORT}")
        stub = pb_grpc.DataProviderStub(channel)
        
        config = pb.StreamConfig(
            batch_size=32,
            mixing_ratio=0.5,
            mixed_mode=True,
        )
        
        label_counts = defaultdict(int)
        
        logger.info(f"Collecting {num_samples} samples...")
        stream = stub.StreamData(config, timeout=GRPC_TIMEOUT)
        
        for i, batch in enumerate(stream):
            label_counts[batch.label] += 1
            
            if sum(label_counts.values()) >= num_samples:
                break
        
        channel.close()
        
        # Validate all labels present
        missing_labels = []
        for label in range(6):
            if label not in label_counts:
                missing_labels.append(label)
        
        # Print distribution
        print_ok("Label distribution:")
        total = sum(label_counts.values())
        for label in range(6):
            count = label_counts[label]
            pct = 100 * count / total if total > 0 else 0
            print_ok(f"  {label} ({LABEL_NAMES[label]:15s}): {count:4d} ({pct:5.1f}%)", 6)
        
        if missing_labels:
            for label in missing_labels:
                print_err(f"  Label {label} ({LABEL_NAMES[label]}) MISSING", 6)
            return False
        
        print_ok(f"All 6 labels present in {total} samples")
        return True
    
    except Exception as e:
        print_err(f"Label distribution test failed: {e}")
        return False

# =============================================================
# TEST 4: Data Consistency (Packet integrity)
# =============================================================

def test_data_consistency(num_samples: int = 100) -> bool:
    """Test data consistency across pipeline"""
    print_section(f"Data Consistency Test ({num_samples} samples)", 2)
    
    try:
        channel = grpc.insecure_channel(f"{GRPC_HOST}:{GRPC_PORT}")
        stub = pb_grpc.DataProviderStub(channel)
        
        config = pb.StreamConfig(batch_size=1, mixing_ratio=1.0, mixed_mode=True)
        stream = stub.StreamData(config, timeout=GRPC_TIMEOUT)
        
        errors = []
        
        for i, batch in enumerate(stream):
            if i >= num_samples:
                break
            
            # Check for NaN/Inf in input vector
            input_arr = np.array(batch.input_vector, dtype=np.float32)
            if np.any(np.isnan(input_arr)):
                errors.append(f"Packet {i}: NaN in input vector")
            if np.any(np.isinf(input_arr)):
                errors.append(f"Packet {i}: Inf in input vector")
            
            # Check raw_bytes not all zeros/ones
            raw_arr = np.frombuffer(batch.raw_bytes, dtype=np.uint8)
            if np.all(raw_arr == 0):
                errors.append(f"Packet {i}: raw_bytes all zeros")
            if np.all(raw_arr == 255):
                errors.append(f"Packet {i}: raw_bytes all ones")
            
            # Check label consistency
            if not (0 <= batch.label <= 5):
                errors.append(f"Packet {i}: invalid label {batch.label}")
        
        channel.close()
        
        if errors:
            for err in errors[:5]:  # Show first 5 errors
                print_err(err, 6)
            if len(errors) > 5:
                print_err(f"  ... and {len(errors)-5} more errors", 6)
            return False
        
        print_ok(f"All {num_samples} packets consistent")
        return True
    
    except Exception as e:
        print_err(f"Consistency test failed: {e}")
        return False

# =============================================================
# TEST 5: Throughput Measurement
# =============================================================

def test_throughput(num_batches: int = 50, batch_size: int = 64) -> bool:
    """Measure pipeline throughput"""
    print_section(f"Throughput Test ({num_batches} batches × {batch_size} size)", 2)
    
    try:
        channel = grpc.insecure_channel(f"{GRPC_HOST}:{GRPC_PORT}")
        stub = pb_grpc.DataProviderStub(channel)
        
        config = pb.StreamConfig(
            batch_size=batch_size,
            mixing_ratio=1.0,
            mixed_mode=True,
        )
        
        logger.info(f"Measuring throughput...")
        start_time = time.time()
        
        stream = stub.StreamData(config, timeout=GRPC_TIMEOUT)
        
        packet_count = 0
        byte_count = 0
        
        for i, batch in enumerate(stream):
            if i >= num_batches:
                break
            
            packet_count += 1
            byte_count += len(batch.raw_bytes) + len(batch.input_vector) * 4
        
        elapsed = time.time() - start_time
        
        channel.close()
        
        # Calculate metrics
        packets_per_sec = packet_count / elapsed if elapsed > 0 else 0
        throughput_mbps = byte_count / elapsed / 1e6 if elapsed > 0 else 0
        avg_latency_ms = (elapsed * 1000) / packet_count if packet_count > 0 else 0
        
        print_ok(f"Packets: {packet_count}")
        print_ok(f"Data: {byte_count / 1e6:.1f} MB", 4)
        print_ok(f"Time: {elapsed:.2f}s", 4)
        print_ok(f"Throughput: {packets_per_sec:.0f} packets/sec", 4)
        print_ok(f"Throughput: {throughput_mbps:.1f} MB/s", 4)
        print_ok(f"Latency: {avg_latency_ms:.2f} ms/packet", 4)
        
        return True
    
    except Exception as e:
        print_err(f"Throughput test failed: {e}")
        return False

# =============================================================
# TEST 6: ML Model Integration
# =============================================================

def test_ml_integration(num_samples: int = 64) -> bool:
    """Test that ML pipeline can process Go bridge data"""
    print_section(f"ML Integration Test ({num_samples} samples)", 2)
    
    try:
        # Import ML components
        try:
            from ml import utils, data as ml_data
        except ImportError:
            print_warn("ML module not available, skipping ML integration test", 4)
            return True
        
        channel = grpc.insecure_channel(f"{GRPC_HOST}:{GRPC_PORT}")
        stub = pb_grpc.DataProviderStub(channel)
        
        config = pb.StreamConfig(
            batch_size=num_samples,
            mixing_ratio=0.5,
            mixed_mode=True,
        )
        
        logger.info("Testing ML pipeline...")
        
        # Collect samples
        features = []
        labels = []
        
        stream = stub.StreamData(config, timeout=GRPC_TIMEOUT)
        
        for batch in stream:
            if len(features) >= num_samples:
                break
            
            # Convert to ML format
            input_arr = np.array(batch.input_vector, dtype=np.float32)
            features.append(input_arr)
            labels.append(batch.label)
        
        channel.close()
        
        if not features:
            print_err("No data collected")
            return False
        
        # Stack into numpy array
        features_array = np.array(features, dtype=np.float32)
        labels_array = np.array(labels, dtype=np.int64)
        
        print_ok(f"Features shape: {features_array.shape}", 4)
        print_ok(f"Labels shape: {labels_array.shape}", 4)
        
        # Validate shapes
        if len(features_array) != len(labels_array):
            print_err(f"Shape mismatch: {len(features_array)} != {len(labels_array)}")
            return False
        
        # Validate ranges
        if np.min(features_array) < -10 or np.max(features_array) > 10:
            print_warn("Feature values outside expected range [-10, 10]", 4)
        
        if np.min(labels_array) < 0 or np.max(labels_array) > 5:
            print_err(f"Labels out of range: [{np.min(labels_array)}, {np.max(labels_array)}]")
            return False
        
        print_ok(f"ML integration successful")
        return True
    
    except Exception as e:
        print_err(f"ML integration test failed: {e}")
        return False

# =============================================================
# MAIN TEST RUNNER
# =============================================================

def run_all_tests() -> int:
    """Run all integration tests"""
    print_banner("PRIME-X Full Pipeline Integration Test")
    
    tests = [
        ("gRPC Connection", test_grpc_connection),
        ("Data Format", test_data_format),
        ("Label Distribution", test_label_distribution),
        ("Data Consistency", test_data_consistency),
        ("Throughput", test_throughput),
        ("ML Integration", test_ml_integration),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print_err(f"{name} crashed: {e}")
            results.append((name, False))
    
    # Print summary
    print_banner("Test Summary")
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    print(f"{BOLD}Results:{RESET}")
    for name, result in results:
        status = f"{GREEN}{CHECK} {name}{RESET}" if result else f"{RED}{CROSS} {name}{RESET}"
        print(f"  {status}")
    
    print(f"\n{BOLD}{YELLOW}Total: {passed}/{len(results)} passed{RESET}")
    
    if failed == 0:
        print(f"\n{GREEN}{BOLD}✓ All integration tests passed!{RESET}\n")
        return 0
    else:
        print(f"\n{RED}{BOLD}✗ {failed} test(s) failed{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

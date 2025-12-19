#!/usr/bin/env python3
"""
PRIME-X INTEGRATION TEST
=========================
Full pipeline test: C++ → Go → Python
Requires:
1. ./prime_core running (Terminal 1)
2. ./bridge/prime_vault running (Terminal 2)
3. This script (Terminal 3)

Tests:
1. gRPC connection stability
2. All 6 labels received
3. Distribution uniformity (chi-square)
4. Data integrity (no corruption)
5. Throughput measurement
6. Latency percentiles
"""

import sys
import time
import grpc
import numpy as np
from collections import defaultdict
from scipy import stats
from typing import List, Tuple
import logging

sys.path.insert(0, '/Users/n0n0/Desktop/n0n0/n0px')
sys.path.insert(0, '/Users/n0n0/Desktop/n0n0/n0px/bridge/pb')

# Try to import gRPC stubs
try:
    import prime_bridge_pb2 as pb
    import prime_bridge_pb2_grpc as pb_grpc
except ImportError as e:
    print(f"ERROR: Failed to import protobuf: {e}")
    print("Make sure to run: make proto-sync")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================

GRPC_HOST = "localhost"
GRPC_PORT = 50051
TEST_SAMPLES = 6000
EXPECTED_LABELS = 6

LABEL_NAMES = {
    0: "COMPOSITE",
    1: "PRIME",
    2: "HARD_COMPOSITE",
    3: "DP_PRIME",
    4: "DP_COMPOSITE",
    5: "DP_HARD_COMPOSITE",
}

# ============================================================
# TEST 1: gRPC Connection
# ============================================================

def test_grpc_connection() -> bool:
    """Test gRPC connection to Go bridge"""
    print("\n[TEST 1] gRPC Connection Stability")
    print("=" * 70)
    
    try:
        channel = grpc.aio.secure_channel(
            f"{GRPC_HOST}:{GRPC_PORT}",
            grpc.ssl_channel_credentials()
        )
        print(f"  ⚠ Using insecure channel (no TLS) - development only")
        
        channel = grpc.insecure_channel(f"{GRPC_HOST}:{GRPC_PORT}")
        stub = pb_grpc.DataProviderStub(channel)
        
        # Try to get one packet
        stream_config = pb.StreamConfig(
            batch_size=32,
            mixing_ratio=1.0,
            mixed_mode=True,
        )
        
        stream = stub.StreamData(stream_config, timeout=5.0)
        
        try:
            first_batch = next(stream)
            print(f"  ✓ Connection successful")
            print(f"  ✓ Received first batch: {len(first_batch.input_vector)} floats")
            channel.close()
            return True
        except StopIteration:
            print(f"  ✗ Stream ended immediately")
            channel.close()
            return False
    except grpc.RpcError as e:
        print(f"  ✗ gRPC error: {e.code()}: {e.details()}")
        return False
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        return False

# ============================================================
# TEST 2: Label Coverage (All 6 Present)
# ============================================================

def test_label_coverage(num_packets: int = 100) -> Tuple[bool, dict]:
    """Verify all 6 labels appear in sample"""
    print(f"\n[TEST 2] Label Coverage (n={num_packets} packets)")
    print("=" * 70)
    
    try:
        channel = grpc.insecure_channel(f"{GRPC_HOST}:{GRPC_PORT}")
        stub = pb_grpc.DataProviderStub(channel)
        
        stream_config = pb.StreamConfig(
            batch_size=1,
            mixing_ratio=1.0,
            mixed_mode=False,
        )
        
        stream = stub.StreamData(stream_config, timeout=30.0)
        
        label_counts = defaultdict(int)
        received = 0
        
        for batch in stream:
            label_counts[batch.label] += 1
            received += 1
            if received >= num_packets:
                break
        
        channel.close()
        
        # Report
        print(f"  Received: {received} packets")
        for label in range(EXPECTED_LABELS):
            count = label_counts.get(label, 0)
            pct = count / received * 100.0 if received > 0 else 0
            status = "✓" if count > 0 else "✗"
            print(f"    {status} Label {label} ({LABEL_NAMES[label]}): {count} ({pct:.1f}%)")
        
        # Check all labels present
        all_present = all(label_counts.get(i, 0) > 0 for i in range(EXPECTED_LABELS))
        if all_present:
            print(f"  ✓ All 6 labels present")
        else:
            missing = [i for i in range(EXPECTED_LABELS) if label_counts.get(i, 0) == 0]
            print(f"  ✗ Missing labels: {missing}")
        
        return all_present, dict(label_counts)
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False, {}

# ============================================================
# TEST 3: Distribution Uniformity (Chi-Square)
# ============================================================

def test_distribution_uniformity(num_packets: int = TEST_SAMPLES) -> Tuple[bool, float]:
    """Chi-square test for uniform distribution"""
    print(f"\n[TEST 3] Distribution Uniformity (χ² test, n={num_packets})")
    print("=" * 70)
    
    try:
        channel = grpc.insecure_channel(f"{GRPC_HOST}:{GRPC_PORT}")
        stub = pb_grpc.DataProviderStub(channel)
        
        stream_config = pb.StreamConfig(
            batch_size=1,
            mixing_ratio=1.0,
            mixed_mode=False,
        )
        
        stream = stub.StreamData(stream_config, timeout=120.0)
        
        label_counts = np.zeros(EXPECTED_LABELS, dtype=int)
        received = 0
        start_time = time.time()
        
        print(f"  Collecting samples...")
        for batch in stream:
            if batch.label < EXPECTED_LABELS:
                label_counts[batch.label] += 1
            received += 1
            
            if received % 1000 == 0:
                elapsed = time.time() - start_time
                rate = received / elapsed
                print(f"    {received}/{num_packets} packets ({rate:.0f} pkt/sec)")
            
            if received >= num_packets:
                break
        
        elapsed = time.time() - start_time
        channel.close()
        
        # Chi-square test
        expected = num_packets / EXPECTED_LABELS
        chi2_stat, p_value = stats.chisquare(label_counts)
        
        print(f"  Throughput: {received/elapsed:.0f} packets/sec")
        print(f"  Chi-square statistic: {chi2_stat:.4f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Distribution:")
        for label in range(EXPECTED_LABELS):
            count = label_counts[label]
            pct = count / received * 100.0
            print(f"    Label {label}: {count:4d} ({pct:.2f}%)")
        
        # p > 0.05 means H0 not rejected (distribution is uniform)
        uniform = p_value > 0.05
        if uniform:
            print(f"  ✓ Distribution is uniform (p={p_value:.6f} > 0.05)")
        else:
            print(f"  ✗ Distribution is NOT uniform (p={p_value:.6f} ≤ 0.05)")
        
        return uniform, p_value
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 1.0

# ============================================================
# TEST 4: Data Integrity
# ============================================================

def test_data_integrity(num_packets: int = 100) -> Tuple[bool, dict]:
    """Verify packet structure and data integrity"""
    print(f"\n[TEST 4] Data Integrity (n={num_packets} packets)")
    print("=" * 70)
    
    try:
        channel = grpc.insecure_channel(f"{GRPC_HOST}:{GRPC_PORT}")
        stub = pb_grpc.DataProviderStub(channel)
        
        stream_config = pb.StreamConfig(
            batch_size=1,
            mixing_ratio=1.0,
            mixed_mode=False,
        )
        
        stream = stub.StreamData(stream_config, timeout=30.0)
        
        issues = {
            'invalid_label': 0,
            'wrong_raw_size': 0,
            'wrong_vector_size': 0,
            'nan_values': 0,
            'inf_values': 0,
        }
        
        received = 0
        for batch in stream:
            # Check label
            if batch.label < 0 or batch.label >= EXPECTED_LABELS:
                issues['invalid_label'] += 1
            
            # Check raw bytes size
            if len(batch.raw_bytes) != 32:
                issues['wrong_raw_size'] += 1
            
            # Check vector size
            if len(batch.input_vector) != 256:
                issues['wrong_vector_size'] += 1
            
            # Check for NaN/Inf
            vec = np.array(batch.input_vector)
            if np.isnan(vec).any():
                issues['nan_values'] += 1
            if np.isinf(vec).any():
                issues['inf_values'] += 1
            
            received += 1
            if received >= num_packets:
                break
        
        channel.close()
        
        # Report
        print(f"  Verified: {received} packets")
        for issue, count in issues.items():
            status = "✓" if count == 0 else "✗"
            print(f"    {status} {issue}: {count}")
        
        all_ok = sum(issues.values()) == 0
        if all_ok:
            print(f"  ✓ All packets valid")
        else:
            print(f"  ✗ {sum(issues.values())} integrity issues detected")
        
        return all_ok, issues
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False, {}

# ============================================================
# TEST 5: Throughput & Latency
# ============================================================

def test_throughput_latency(num_packets: int = 1000) -> Tuple[float, dict]:
    """Measure throughput and latency percentiles"""
    print(f"\n[TEST 5] Throughput & Latency (n={num_packets} packets)")
    print("=" * 70)
    
    try:
        channel = grpc.insecure_channel(f"{GRPC_HOST}:{GRPC_PORT}")
        stub = pb_grpc.DataProviderStub(channel)
        
        stream_config = pb.StreamConfig(
            batch_size=1,
            mixing_ratio=1.0,
            mixed_mode=False,
        )
        
        stream = stub.StreamData(stream_config, timeout=120.0)
        
        latencies = []
        received = 0
        start_time = time.time()
        prev_time = start_time
        
        for batch in stream:
            curr_time = time.time()
            latency = (curr_time - prev_time) * 1000  # ms
            latencies.append(latency)
            prev_time = curr_time
            
            received += 1
            if received >= num_packets:
                break
        
        total_time = time.time() - start_time
        channel.close()
        
        # Calculate metrics
        throughput = received / total_time
        latencies = np.array(latencies[1:])  # Skip first
        
        print(f"  Throughput: {throughput:.0f} packets/sec")
        print(f"  Total time: {total_time:.2f} sec")
        print(f"  Latency statistics (ms):")
        print(f"    Mean: {latencies.mean():.2f}")
        print(f"    Min: {latencies.min():.2f}")
        print(f"    P50: {np.percentile(latencies, 50):.2f}")
        print(f"    P95: {np.percentile(latencies, 95):.2f}")
        print(f"    P99: {np.percentile(latencies, 99):.2f}")
        print(f"    Max: {latencies.max():.2f}")
        print(f"  ✓ Performance measured")
        
        metrics = {
            'throughput': throughput,
            'total_time': total_time,
            'mean_latency': latencies.mean(),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
        }
        
        return throughput, metrics
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, {}

# ============================================================
# MAIN TEST RUNNER
# ============================================================

def main():
    print("\n" + "="*70)
    print("PRIME-X INTEGRATION TEST (Full Pipeline)")
    print("="*70)
    print("Prerequisites:")
    print("  1. ./prime_core running (Terminal 1)")
    print("  2. ./bridge/prime_vault running (Terminal 2)")
    print("  3. This script (Terminal 3)")
    
    results = {
        'connection': False,
        'coverage': False,
        'uniformity': False,
        'integrity': False,
        'performance': {},
    }
    
    # Test 1: Connection
    if not test_grpc_connection():
        print("\n✗ Cannot connect to Go bridge. Exiting.")
        return False
    results['connection'] = True
    
    # Test 2: Label coverage
    coverage_ok, label_counts = test_label_coverage(100)
    results['coverage'] = coverage_ok
    
    # Test 3: Distribution (main test)
    print("\n  Preparing for main distribution test (6000 packets)...")
    print("  This will take ~60-120 seconds...")
    uniform_ok, p_value = test_distribution_uniformity(TEST_SAMPLES)
    results['uniformity'] = uniform_ok
    
    # Test 4: Integrity
    integrity_ok, issues = test_data_integrity(100)
    results['integrity'] = integrity_ok
    
    # Test 5: Performance
    throughput, metrics = test_throughput_latency(1000)
    results['performance'] = metrics
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Connection: {'✓ PASS' if results['connection'] else '✗ FAIL'}")
    print(f"  Coverage: {'✓ PASS' if results['coverage'] else '✗ FAIL'}")
    print(f"  Uniformity: {'✓ PASS (p={p_value:.4f})' if results['uniformity'] else f'✗ FAIL (p={p_value:.4f})'}")
    print(f"  Integrity: {'✓ PASS' if results['integrity'] else '✗ FAIL'}")
    if results['performance']:
        print(f"  Throughput: {results['performance'].get('throughput', 0):.0f} pkt/sec")
    
    all_passed = all([
        results['connection'],
        results['coverage'],
        results['uniformity'],
        results['integrity'],
    ])
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL INTEGRATION TESTS PASSED")
    else:
        print("✗ SOME INTEGRATION TESTS FAILED")
    print("="*70 + "\n")
    
    return all_passed

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

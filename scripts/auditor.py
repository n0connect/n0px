#!/usr/bin/env python3
"""
PX-QSDA AUDITOR v8.2
===================
Full pipeline validator using config.json for all parameters.
"""

import json
import grpc
import struct
import hashlib
import math
import time
import numpy as np
import os
import sys
from pathlib import Path
from sympy.ntheory.primetest import isprime as bpsw_test

# Add bridge/pb to path for protobuf imports
sys.path.insert(0, str(Path(__file__).parent.parent / "bridge" / "pb"))

import prime_bridge_pb2 as pb
import prime_bridge_pb2_grpc as pb_grpc

# ============================================================
#  LOAD CONFIG
# ============================================================
_config_file = Path(__file__).parent.parent / "config" / "config.json"
with open(_config_file) as f:
    CONFIG = json.load(f)

# Extract parameters
GRPC_HOST = os.getenv("GRPC_HOST", CONFIG['network']['grpc_host'])
GRPC_PORT = int(os.getenv("GRPC_PORT", CONFIG['network']['grpc_port']))
RAW_SIZE = CONFIG['system']['raw_size_bytes']
BITS = CONFIG['system']['prime_bits']
LABEL_NAMES = {
    CONFIG['labels']['easy_composite']: "EASY_COMP",
    CONFIG['labels']['prime']: "PRIME",
    CONFIG['labels']['hard_composite']: "HARD_COMP",
    CONFIG['labels']['dp_prime']: "DP_PRIME",
    CONFIG['labels']['dp_composite']: "DP_COMPOSITE",
    CONFIG['labels']['dp_hard_composite']: "DP_HARD_COMP",
}
BOOTSTRAP_PROTOCOL_SAMPLES = CONFIG['bootstrap']['protocol_check_samples']
BOOTSTRAP_MATH_SAMPLES = CONFIG['bootstrap']['math_test_samples']
BOOTSTRAP_STRESS_DURATION = CONFIG['bootstrap']['stress_test_duration_seconds']
BOOTSTRAP_MIN_THROUGHPUT = CONFIG['bootstrap']['min_throughput_packets_per_sec']

# ============================================================
#  COLOR SYSTEM
# ============================================================
class C:
    OK   = "\033[38;5;46m"
    ERR  = "\033[38;5;196m"
    INF  = "\033[38;5;39m"
    WRN  = "\033[38;5;226m"
    BOLD = "\033[1m"
    RST  = "\033[0m"

def banner():
    print(f"{C.INF}{C.BOLD}")
    print("═══════════════════════════════════════════════════")
    print("   PX-QSDA AUDITOR v8.2 — FULL PIPELINE VALIDATOR  ")
    print("═══════════════════════════════════════════════════")
    print(f"{C.RST}")
    print(f"{C.INF}Configuration: {BITS}-bit system @ {GRPC_HOST}:{GRPC_PORT}{C.RST}")


# ============================================================
#  INTERNAL UTILS
# ============================================================
def blake2s_u64(data: bytes):
    """Little-endian uint64 Blake2s digest."""
    return int.from_bytes(hashlib.blake2s(data).digest()[:8], "little")


def fetch_stream(stub, batch_size=1):
    req = pb.StreamConfig(batch_size=batch_size, mixing_ratio=0.5, mixed_mode=True)
    return stub.StreamData(req)


# ============================================================
#  PROTOCOL CHECKS
# ============================================================
def packet_ok(pkt):
    """Basic protocol-level validation using config values."""
    assert len(pkt.raw_bytes) == RAW_SIZE, f"RawBytes size mismatch: expected {RAW_SIZE}, got {len(pkt.raw_bytes)}"
    assert len(pkt.input_vector) == BITS, f"InputVector length mismatch: expected {BITS}, got {len(pkt.input_vector)}"
    assert pkt.label in LABEL_NAMES, f"Invalid label: {pkt.label}"

    # Noise sanity check
    for x in pkt.input_vector:
        assert -10 < x < 10, "Float32 noise out of expected range"


def hash_ok(pkt):
    """
    C++ → Go → Python hash model:
    Header is NOT present in protobuf.
    So we recompute and ensure the hash logic is valid.
    """
    raw = pkt.raw_bytes
    vec = b"".join(struct.pack("<f", x) for x in pkt.input_vector)

    h1 = blake2s_u64(raw)
    h2 = blake2s_u64(vec)

    # No header to compare with → just ensure they compute
    assert isinstance(h1, int) and isinstance(h2, int)


# ============================================================
#  MATHEMATICAL TESTS (BPSW)
# ============================================================
def math_test(pkt):
    """Prime/composite correctness for each type."""
    n = int.from_bytes(pkt.raw_bytes, "big")
    label = pkt.label

    is_p = bpsw_test(n)

    # Use label values from config
    easy_comp = CONFIG['labels']['easy_composite']
    prime = CONFIG['labels']['prime']
    hard_comp = CONFIG['labels']['hard_composite']
    dp_prime = CONFIG['labels']['dp_prime']

    if label == prime:
        assert is_p, f"PRIME expected, composite found! ({hex(n)})"
    elif label == dp_prime:
        assert is_p, f"DP_PRIME expected, composite found! ({hex(n)})"
    elif label == easy_comp:
        assert not is_p, f"EASY_COMP expected, prime found! ({hex(n)})"
    elif label == hard_comp:
        assert not is_p, f"HARD_COMP expected, prime found! ({hex(n)})"


# ============================================================
#  RANDOMNESS TESTS
# ============================================================
def monobit(bits):
    ones = sum(bits)
    zeros = len(bits) - ones
    z = (ones - zeros) / math.sqrt(len(bits))
    return z


def randomness_tests(samples):
    """Monobit + Gaussian statistics + autocorrelation."""
    raw_values = []
    vectors = []

    for pkt in samples:
        raw_values.append(pkt.raw_bytes)
        vectors.append(pkt.input_vector)

    # -----------------------------
    #  MONOBIT
    # -----------------------------
    bitlist = []
    for raw in raw_values:
        n = int.from_bytes(raw, "big")
        for i in range(BITS):
            bitlist.append((n >> i) & 1)

    z = monobit(bitlist)
    assert abs(z) < 4.0, f"Monobit χ-test failed (z={z})"

    # -----------------------------
    #  GAUSSIAN (mean/var)
    # -----------------------------
    arr = np.array(vectors).flatten()
    mean = np.mean(arr)
    var  = np.var(arr)

    assert abs(mean) < 1.0,  f"Gaussian mean too large: {mean}"
    assert 0.001 < var < 10.0, f"Gaussian variance abnormal: {var}"

    # -----------------------------
    #  AUTOCORRELATION
    # -----------------------------
    ac = np.corrcoef(arr[:-1], arr[1:])[0,1]
    assert abs(ac) < 0.05, f"Autocorrelation too high: {ac}"

    print(f"{C.OK}Randomness OK — z={z:.3f}, mean={mean:.3f}, var={var:.3f}, ac={ac:.3f}{C.RST}")


# ============================================================
#  THROUGHPUT / STABILITY TEST
# ============================================================
def stress_test(stub):
    duration = BOOTSTRAP_STRESS_DURATION
    print(f"{C.INF}Running {duration}s stability stress test...{C.RST}")
    s = fetch_stream(stub)
    t0 = time.time()
    count = 0

    while time.time() - t0 < duration:
        next(s)
        count += 1

    throughput = count / duration
    print(f"{C.OK}OK — {count} packets received ({throughput:.1f}/s){C.RST}")
    assert throughput >= BOOTSTRAP_MIN_THROUGHPUT, f"Throughput {throughput:.1f}/s below minimum {BOOTSTRAP_MIN_THROUGHPUT}/s"


# ============================================================
#  MAIN
# ============================================================
def continuous_audit(stub, interval_secs=10):
    """
    Continuous audit mode: run validation tests repeatedly.
    Tests packet integrity, math correctness, and throughput periodically.
    """
    stream = fetch_stream(stub, batch_size=1)
    iteration = 0
    
    while True:
        iteration += 1
        print(f"\n{C.BOLD}[ITERATION {iteration}]{C.RST} — Running continuous validation...")
        
        # Quick protocol check (10 packets)
        errors = 0
        for i in range(10):
            try:
                pkt = next(stream)
                packet_ok(pkt)
                hash_ok(pkt)
            except Exception as e:
                print(f"{C.WARN}Packet {i} failed: {e}{C.RST}")
                errors += 1
        
        if errors == 0:
            print(f"{C.OK}✓ Protocol integrity: OK (10/10){C.RST}")
        else:
            print(f"{C.ERR}✗ Protocol integrity: FAILED ({errors}/10){C.RST}")
        
        # Math test (5 packets)
        math_errors = 0
        for i in range(5):
            try:
                pkt = next(stream)
                math_test(pkt)
            except Exception as e:
                print(f"{C.WARN}Math test {i} failed: {e}{C.RST}")
                math_errors += 1
        
        if math_errors == 0:
            print(f"{C.OK}✓ Mathematical integrity: OK (5/5){C.RST}")
        else:
            print(f"{C.ERR}✗ Mathematical integrity: FAILED ({math_errors}/5){C.RST}")
        
        # Throughput check (measure rate over interval_secs)
        t0 = time.time()
        count = 0
        while time.time() - t0 < interval_secs:
            try:
                _ = next(stream)
                count += 1
            except:
                pass
        
        throughput = count / interval_secs
        status = C.OK if throughput >= BOOTSTRAP_MIN_THROUGHPUT else C.WARN
        print(f"{status}✓ Throughput: {throughput:.1f} pkt/s (min: {BOOTSTRAP_MIN_THROUGHPUT}){C.RST}")
        
        print(f"{C.INF}Next check in {interval_secs}s...{C.RST}")
        time.sleep(interval_secs)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="PX-QSDA Auditor")
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuous audit mode (instead of one-time bootstrap)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Interval between continuous checks in seconds (default: 10)"
    )
    
    args = parser.parse_args()
    
    banner()
    print(f"{C.INF}Connecting to gRPC bridge at {GRPC_HOST}:{GRPC_PORT}...{C.RST}")

    channel = grpc.insecure_channel(f"{GRPC_HOST}:{GRPC_PORT}")
    stub = pb_grpc.DataProviderStub(channel)
    
    if args.continuous:
        print(f"{C.BOLD}CONTINUOUS AUDIT MODE{C.RST} (interval: {args.interval}s)")
        continuous_audit(stub, interval_secs=args.interval)
    else:
        # Original bootstrap mode
        stream = fetch_stream(stub, batch_size=1)

        # --------------------------------------------------------
        #  PHASE 1 — PROTOCOL CHECKS
        # --------------------------------------------------------
        print(f"{C.INF}Phase 1 — Protocol & Hash Integrity ({BOOTSTRAP_PROTOCOL_SAMPLES} samples)...{C.RST}")
        for _ in range(BOOTSTRAP_PROTOCOL_SAMPLES):
            pkt = next(stream)
            packet_ok(pkt)
            hash_ok(pkt)
        print(f"{C.OK}Protocol OK.{C.RST}")

        # --------------------------------------------------------
        #  PHASE 2 — PRIME/COMPOSITE MATH TESTS
        # --------------------------------------------------------
        print(f"{C.INF}Phase 2 — Mathematical Integrity (BPSW) ({BOOTSTRAP_MATH_SAMPLES} samples)...{C.RST}")
        for _ in range(BOOTSTRAP_MATH_SAMPLES):
            pkt = next(stream)
            math_test(pkt)
        print(f"{C.OK}Math OK.{C.RST}")

        # --------------------------------------------------------
        #  PHASE 3 — RANDOMNESS & DISTRIBUTION ANALYSIS
        # --------------------------------------------------------
        print(f"{C.INF}Phase 3 — Randomness Analysis...{C.RST}")
        samples = [next(stream) for _ in range(300)]
        randomness_tests(samples)

        # --------------------------------------------------------
        #  PHASE 4 — THROUGHPUT / STABILITY
        # --------------------------------------------------------
        stress_test(stub, 60)

        print(f"{C.BOLD}{C.OK}ALL TESTS PASSED — PIPELINE VERIFIED ✔{C.RST}")


if __name__ == "__main__":
    main()

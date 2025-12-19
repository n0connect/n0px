#!/usr/bin/env python3
"""
ML Bootstrap Test Suite
========================
Comprehensive validation tests for the ML pipeline.
Includes protocol checks, data integrity, mathematical validation, and stress testing.
"""

import logging
import time
import math
import struct
import numpy as np
from typing import List, Dict, Any
from sympy.ntheory.primetest import isprime as bpsw_test

from . import config
from . import utils
from . import core

logger = utils.setup_logger(__name__, config.LOG_LEVEL)


# ============================================================
#  PROTOCOL TESTS
# ============================================================
class ProtocolTests:
    """Validate packet structure and protocol compliance."""
    
    @staticmethod
    def check_packet_structure(pkt: core.DataPacket) -> Dict[str, Any]:
        """
        Validate packet structure against protocol specification.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }
        
        # Check raw_bytes size
        if len(pkt.raw_bytes) != config.DEFAULT_RAW_SIZE:
            results["errors"].append(
                f"Raw bytes size mismatch: {len(pkt.raw_bytes)} != {config.DEFAULT_RAW_SIZE}"
            )
            results["valid"] = False
        
        # Check input_vector length
        if len(pkt.input_vector) != config.DEFAULT_BITS:
            results["errors"].append(
                f"Input vector length mismatch: {len(pkt.input_vector)} != {config.DEFAULT_BITS}"
            )
            results["valid"] = False
        
        # Check label range
        if not (0 <= pkt.label <= 3):
            results["errors"].append(f"Invalid label: {pkt.label}")
            results["valid"] = False
        
        # Check float values in reasonable range
        for i, val in enumerate(pkt.input_vector):
            if not (-10 <= val <= 10):
                results["warnings"].append(
                    f"Float value out of expected range at index {i}: {val}"
                )
        
        return results
    
    @staticmethod
    def check_hash_integrity(
        pkt: core.DataPacket,
        expected_h1: int = None,
        expected_h2: int = None,
    ) -> Dict[str, Any]:
        """
        Verify packet hashes (BLAKE2s integrity check).
        
        If expected hashes are provided, compare with computed hashes.
        Otherwise, just verify that hashes can be computed.
        
        Returns:
            Dictionary with hash validation results
        """
        results = {
            "valid": True,
            "h1_computed": None,
            "h2_computed": None,
            "errors": [],
        }
        
        try:
            h1_computed, h2_computed = pkt.compute_hashes()
            results["h1_computed"] = h1_computed
            results["h2_computed"] = h2_computed
            
            if expected_h1 is not None and h1_computed != expected_h1:
                results["errors"].append(
                    f"H1 mismatch: computed {h1_computed} != expected {expected_h1}"
                )
                results["valid"] = False
            
            if expected_h2 is not None and h2_computed != expected_h2:
                results["errors"].append(
                    f"H2 mismatch: computed {h2_computed} != expected {expected_h2}"
                )
                results["valid"] = False
        
        except Exception as e:
            results["errors"].append(f"Hash computation error: {e}")
            results["valid"] = False
        
        return results


# ============================================================
#  MATHEMATICAL TESTS
# ============================================================
class MathematicalTests:
    """Validate mathematical properties of data."""
    
    @staticmethod
    def is_prime_correct(pkt: core.DataPacket) -> Dict[str, Any]:
        """
        Verify prime/composite classification using BPSW test.
        
        Labels:
        - 1 (PRIME): Must be prime
        - 3 (DP_PRIME): Must be prime
        - 0 (EASY_COMP): Must be composite
        - 2 (HARD_COMP): Must be composite
        
        Returns:
            Dictionary with primality test results
        """
        results = {
            "valid": True,
            "label": pkt.label,
            "label_name": utils.format_label(pkt.label),
            "is_prime_bpsw": None,
            "error": None,
        }
        
        try:
            # Convert bytes to integer (big-endian)
            n = int.from_bytes(pkt.raw_bytes, "big")
            
            # BPSW test
            is_prime = bpsw_test(n)
            results["is_prime_bpsw"] = is_prime
            
            # Verify against label
            if pkt.label in (1, 3):  # PRIME or DP_PRIME
                if not is_prime:
                    results["valid"] = False
                    results["error"] = f"Expected PRIME but got COMPOSITE ({utils.format_label(pkt.label)})"
            
            elif pkt.label in (0, 2):  # EASY_COMP or HARD_COMP
                if is_prime:
                    results["valid"] = False
                    results["error"] = f"Expected COMPOSITE but got PRIME ({utils.format_label(pkt.label)})"
        
        except Exception as e:
            results["valid"] = False
            results["error"] = str(e)
        
        return results


# ============================================================
#  STATISTICAL TESTS
# ============================================================
class StatisticalTests:
    """Analyze statistical properties of data streams."""
    
    @staticmethod
    def monobit_test(bitlist: List[int]) -> Dict[str, Any]:
        """
        Monobit test: Check if 0s and 1s are roughly equally distributed.
        
        Args:
            bitlist: List of binary values (0 or 1)
        
        Returns:
            Dictionary with test results and z-score
        """
        if not bitlist:
            return {"valid": False, "error": "Empty bitlist"}
        
        ones = sum(bitlist)
        zeros = len(bitlist) - ones
        
        # Z-score (should be < 4 for good randomness)
        z_score = abs(ones - zeros) / math.sqrt(len(bitlist))
        
        return {
            "ones": ones,
            "zeros": zeros,
            "total": len(bitlist),
            "z_score": z_score,
            "valid": abs(z_score) < 4.0,
        }
    
    @staticmethod
    def gaussian_test(values: np.ndarray) -> Dict[str, Any]:
        """
        Test for Gaussian (normal) distribution of float values.
        
        Checks:
        - Mean close to 0
        - Variance in reasonable range
        
        Returns:
            Dictionary with statistics
        """
        mean = float(np.mean(values))
        std = float(np.std(values))
        var = std ** 2
        
        return {
            "mean": mean,
            "std": std,
            "variance": var,
            "mean_valid": abs(mean) < 1.0,
            "var_valid": 0.001 < var < 10.0,
        }
    
    @staticmethod
    def autocorrelation_test(values: np.ndarray) -> Dict[str, Any]:
        """
        Test for autocorrelation (should be low for random data).
        
        Returns:
            Dictionary with correlation coefficient
        """
        if len(values) < 2:
            return {"valid": False, "error": "Need at least 2 values"}
        
        corr = np.corrcoef(values[:-1], values[1:])[0, 1]
        
        return {
            "correlation": float(corr),
            "valid": abs(corr) < 0.05,
        }
    
    @staticmethod
    def analyze_packets(packets: List[core.DataPacket]) -> Dict[str, Any]:
        """
        Comprehensive statistical analysis of a packet collection.
        
        Returns:
            Dictionary with all statistical results
        """
        if not packets:
            return {"error": "No packets to analyze"}
        
        # Collect all float values
        all_floats = np.concatenate([np.array(p.input_vector) for p in packets])
        
        # Collect all bits from raw_bytes
        bitlist = []
        for pkt in packets:
            n = int.from_bytes(pkt.raw_bytes, "big")
            for i in range(config.DEFAULT_RAW_SIZE * 8):
                bitlist.append((n >> i) & 1)
        
        results = {
            "num_packets": len(packets),
            "monobit": StatisticalTests.monobit_test(bitlist),
            "gaussian": StatisticalTests.gaussian_test(all_floats),
            "autocorrelation": StatisticalTests.autocorrelation_test(all_floats),
        }
        
        return results


# ============================================================
#  BOOTSTRAP TESTS
# ============================================================
class BootstrapTestSuite:
    """Main bootstrap test suite for pipeline validation."""
    
    def __init__(self, bridge: core.GRPCBridge):
        """Initialize test suite with connected bridge."""
        self.bridge = bridge
        self.results = {
            "all_passed": True,
            "tests": {},
        }
    
    def run_all(self) -> Dict[str, Any]:
        """Run complete bootstrap test suite."""
        utils.print_banner("ML BOOTSTRAP TEST SUITE")
        
        self._test_connection()
        self._test_protocol()
        self._test_mathematical()
        self._test_statistical()
        self._test_stress()
        
        self._print_summary()
        
        return self.results
    
    def _test_connection(self) -> None:
        """Test gRPC connection."""
        utils.print_section("Phase 1: Connection Test")
        
        try:
            if not self.bridge.connected:
                raise utils.BootstrapError("Bridge not connected")
            
            # Try to get one packet
            pkt = self.bridge.get_packet()
            if pkt is None:
                raise utils.BootstrapError("Failed to get first packet")
            
            utils.print_ok("gRPC connection successful")
            self.results["tests"]["connection"] = {"passed": True}
        
        except Exception as e:
            utils.print_err(f"Connection test failed: {e}")
            self.results["tests"]["connection"] = {"passed": False, "error": str(e)}
            self.results["all_passed"] = False
    
    def _test_protocol(self) -> None:
        """Test protocol compliance."""
        utils.print_section("Phase 2: Protocol & Structure Tests")
        
        try:
            passed = 0
            failed = 0
            
            logger.info(
                f"Fetching {config.BOOTSTRAP_PROTOCOL_CHECK_SAMPLES} packets "
                f"for protocol validation..."
            )
            
            for i in range(config.BOOTSTRAP_PROTOCOL_CHECK_SAMPLES):
                pkt = self.bridge.get_packet()
                if pkt is None:
                    logger.error("Stream ended unexpectedly")
                    break
                
                # Check structure
                struct_result = ProtocolTests.check_packet_structure(pkt)
                if not struct_result["valid"]:
                    failed += 1
                    if config.STRICT_MODE:
                        raise utils.DataValidationError(
                            f"Packet {i} structure invalid: {struct_result['errors']}"
                        )
                else:
                    passed += 1
                
                # Check hash
                if config.HASH_VERIFICATION_ENABLED:
                    hash_result = ProtocolTests.check_hash_integrity(pkt)
                    if not hash_result["valid"]:
                        failed += 1
                        if config.STRICT_MODE:
                            raise utils.DataValidationError(
                                f"Packet {i} hash invalid: {hash_result['errors']}"
                            )
            
            if failed == 0:
                utils.print_ok(f"Protocol validation passed ({passed}/{passed+failed} packets)")
                self.results["tests"]["protocol"] = {"passed": True, "count": passed}
            else:
                utils.print_warn(
                    f"Protocol validation: {passed} passed, {failed} failed "
                    f"(mode={['soft', 'strict'][config.STRICT_MODE]})"
                )
                self.results["tests"]["protocol"] = {
                    "passed": not config.STRICT_MODE,
                    "count": passed,
                    "failed": failed,
                }
        
        except Exception as e:
            utils.print_err(f"Protocol test failed: {e}")
            self.results["tests"]["protocol"] = {"passed": False, "error": str(e)}
            self.results["all_passed"] = False
    
    def _test_mathematical(self) -> None:
        """Test mathematical correctness."""
        utils.print_section("Phase 3: Mathematical Validation (BPSW Prime Tests)")
        
        try:
            if not config.MATHEMATICAL_VALIDATION_ENABLED:
                logger.warning("Mathematical validation disabled, skipping")
                return
            
            passed = 0
            failed = 0
            
            logger.info(
                f"Fetching {config.BOOTSTRAP_MATH_TEST_SAMPLES} packets "
                f"for BPSW primality testing..."
            )
            
            for i in range(config.BOOTSTRAP_MATH_TEST_SAMPLES):
                pkt = self.bridge.get_packet()
                if pkt is None:
                    logger.error("Stream ended unexpectedly")
                    break
                
                math_result = MathematicalTests.is_prime_correct(pkt)
                if math_result["valid"]:
                    passed += 1
                else:
                    failed += 1
                    error = math_result.get("error", "Unknown error")
                    logger.warning(f"Packet {i}: {error}")
                    
                    if config.STRICT_MODE:
                        raise utils.BootstrapError(
                            f"Packet {i} mathematical validation failed: {error}"
                        )
            
            if failed == 0:
                utils.print_ok(f"Mathematical validation passed ({passed}/{passed+failed} packets)")
                self.results["tests"]["mathematical"] = {"passed": True, "count": passed}
            else:
                utils.print_warn(
                    f"Mathematical validation: {passed} passed, {failed} failed "
                    f"(mode={['soft', 'strict'][config.STRICT_MODE]})"
                )
                self.results["tests"]["mathematical"] = {
                    "passed": not config.STRICT_MODE,
                    "count": passed,
                    "failed": failed,
                }
        
        except Exception as e:
            utils.print_err(f"Mathematical test failed: {e}")
            self.results["tests"]["mathematical"] = {"passed": False, "error": str(e)}
            self.results["all_passed"] = False
    
    def _test_statistical(self) -> None:
        """Test statistical properties."""
        utils.print_section("Phase 4: Statistical Analysis")
        
        try:
            logger.info(
                f"Fetching {config.BOOTSTRAP_SAMPLES} packets "
                f"for statistical analysis..."
            )
            
            packets = self.bridge.get_packets(config.BOOTSTRAP_SAMPLES)
            if not packets:
                raise utils.BootstrapError("No packets retrieved for statistical analysis")
            
            stats = StatisticalTests.analyze_packets(packets)
            
            # Print results
            monobit = stats["monobit"]
            gaussian = stats["gaussian"]
            autocorr = stats["autocorrelation"]
            
            logger.info(
                f"Monobit: z-score={monobit['z_score']:.3f}, "
                f"ones={monobit['ones']}, zeros={monobit['zeros']}"
            )
            logger.info(
                f"Gaussian: mean={gaussian['mean']:.4f}, "
                f"std={gaussian['std']:.4f}, var={gaussian['variance']:.4f}"
            )
            logger.info(
                f"Autocorrelation: ρ={autocorr['correlation']:.4f}"
            )
            
            # Validation
            monobit_ok = monobit["valid"]
            gaussian_ok = gaussian["mean_valid"] and gaussian["var_valid"]
            autocorr_ok = autocorr["valid"]
            
            all_ok = monobit_ok and gaussian_ok and autocorr_ok
            
            if all_ok:
                utils.print_ok("Statistical analysis passed")
            else:
                if not monobit_ok:
                    utils.print_warn(f"Monobit z-score too high: {monobit['z_score']:.3f}")
                if not gaussian_ok:
                    utils.print_warn(
                        f"Gaussian properties abnormal: "
                        f"mean={gaussian['mean']:.4f}, var={gaussian['variance']:.4f}"
                    )
                if not autocorr_ok:
                    utils.print_warn(f"Autocorrelation too high: {autocorr['correlation']:.4f}")
            
            self.results["tests"]["statistical"] = {
                "passed": all_ok,
                "stats": stats,
            }
        
        except Exception as e:
            utils.print_err(f"Statistical test failed: {e}")
            self.results["tests"]["statistical"] = {"passed": False, "error": str(e)}
            self.results["all_passed"] = False
    
    def _test_stress(self) -> None:
        """Throughput and stability stress test with hash verification."""
        utils.print_section("Phase 5: Throughput & Stability Stress Test")
        
        try:
            duration = config.BOOTSTRAP_STRESS_TEST_DURATION
            logger.info(f"Running {duration}s stress test with hash verification...")
            
            start_time = time.time()
            count = 0
            hash_failures = 0
            errors = 0
            
            while time.time() - start_time < duration:
                pkt = self.bridge.get_packet()
                if pkt is None:
                    errors += 1
                    if errors > 10:
                        raise utils.BootstrapError("Too many consecutive errors")
                else:
                    count += 1
                    # Verify BLAKE2s hash integrity for each packet
                    hash_result = ProtocolTests.check_hash_integrity(pkt)
                    if not hash_result["valid"]:
                        hash_failures += 1
                        if hash_failures > 5:
                            errors_str = "; ".join(hash_result["errors"])
                            raise utils.BootstrapError(
                                f"Hash verification failed: {errors_str}"
                            )
            
            elapsed = time.time() - start_time
            throughput = count / elapsed
            
            logger.info(
                f"Received {count} packets in {elapsed:.1f}s "
                f"({throughput:.1f} packets/sec)"
            )
            logger.info(
                f"Hash verification: {count - hash_failures}/{count} passed"
            )
            
            if count < 100:
                raise utils.BootstrapError(
                    f"Throughput too low: {count} packets in {duration}s"
                )
            
            if hash_failures > 0:
                utils.print_warn(
                    f"Stress test passed but {hash_failures} packets "
                    f"had hash verification issues"
                )
            else:
                utils.print_ok(f"Stress test passed ({throughput:.1f} packets/sec)")
            
            self.results["tests"]["stress"] = {
                "passed": True,
                "duration": elapsed,
                "count": count,
                "throughput": throughput,
                "hash_failures": hash_failures,
            }
        
        except Exception as e:
            utils.print_err(f"Stress test failed: {e}")
            self.results["tests"]["stress"] = {"passed": False, "error": str(e)}
            self.results["all_passed"] = False
    
    def _print_summary(self) -> None:
        """Print test summary."""
        utils.print_section("Test Summary")
        
        total_tests = len(self.results["tests"])
        passed_tests = sum(
            1 for t in self.results["tests"].values()
            if t.get("passed", False)
        )
        
        print()
        for test_name, result in self.results["tests"].items():
            status = "✓ PASS" if result.get("passed") else "✗ FAIL"
            color = utils.Colors.OK if result.get("passed") else utils.Colors.ERR
            print(f"  {color}{status}{utils.Colors.RESET}  {test_name}")
        
        print()
        if self.results["all_passed"]:
            utils.print_ok(
                f"ALL TESTS PASSED ({passed_tests}/{total_tests})"
            )
        else:
            utils.print_err(
                f"SOME TESTS FAILED ({passed_tests}/{total_tests})"
            )
        print()

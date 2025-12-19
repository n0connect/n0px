#!/usr/bin/env python3
"""
ML Utilities Module
===================
Logging, color output, helper functions, validation utilities.
"""

import logging
import sys
import struct
import hashlib
from typing import List, Tuple

# ============================================================
#  ANSI COLOR CODES
# ============================================================
class Colors:
    """Terminal color codes for pretty output."""
    OK      = "\033[38;5;46m"    # Green
    ERR     = "\033[38;5;196m"   # Red
    INF     = "\033[38;5;39m"    # Blue
    WRN     = "\033[38;5;226m"   # Yellow
    BOLD    = "\033[1m"
    RESET   = "\033[0m"
    GRAY    = "\033[38;5;240m"
    CYAN    = "\033[38;5;51m"
    PURPLE  = "\033[38;5;129m"


# ============================================================
#  LOGGING SETUP
# ============================================================
def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Create and configure a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler with formatting
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        f"{Colors.INF}[%(asctime)s]{Colors.RESET} "
        f"{Colors.BOLD}%(name)s{Colors.RESET}: %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger


# ============================================================
#  BANNER & UI
# ============================================================
def print_banner(title: str = "PRIME-X ML PIPELINE") -> None:
    """Print a fancy banner to console."""
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("═" * 60)
    print(f"  {title}")
    print("═" * 60)
    print(f"{Colors.RESET}")


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.INF}[{title}]{Colors.RESET}")
    print(f"{Colors.GRAY}{'─' * 50}{Colors.RESET}")


def print_ok(msg: str) -> None:
    """Print a success message."""
    print(f"{Colors.OK}✓ {msg}{Colors.RESET}")


def print_err(msg: str) -> None:
    """Print an error message."""
    print(f"{Colors.ERR}✗ {msg}{Colors.RESET}")


def print_warn(msg: str) -> None:
    """Print a warning message."""
    print(f"{Colors.WRN}⚠ {msg}{Colors.RESET}")


def print_info(msg: str) -> None:
    """Print an info message."""
    print(f"{Colors.INF}ℹ {msg}{Colors.RESET}")


# ============================================================
#  CRYPTO HELPERS
# ============================================================
def blake2s_u64(data: bytes) -> int:
    """
    Compute BLAKE2s hash and extract first 8 bytes as little-endian uint64.
    Used for verifying data integrity (must match Go bridge hash).
    """
    h = hashlib.blake2s(data).digest()
    return int.from_bytes(h[:8], "little")


def verify_packet_hashes(raw_bytes: bytes, float_vector: List[float]) -> Tuple[int, int]:
    """
    Compute BLAKE2s hashes for raw bytes and float vector.
    
    Args:
        raw_bytes: 32-byte raw data from packet
        float_vector: 256 float32 values from packet
    
    Returns:
        Tuple of (h1_calc, h2_calc)
    """
    # Raw bytes hash
    h1_calc = blake2s_u64(raw_bytes)
    
    # Float vector hash
    float_bytes = b"".join(struct.pack("<f", x) for x in float_vector)
    h2_calc = blake2s_u64(float_bytes)
    
    return h1_calc, h2_calc


# ============================================================
#  DATA VALIDATION
# ============================================================
def validate_raw_bytes(raw_bytes: bytes, expected_size: int = 32) -> bool:
    """Validate raw bytes size."""
    if len(raw_bytes) != expected_size:
        return False
    return True


def validate_float_vector(floats: List[float], expected_length: int = None) -> bool:
    """
    Validate float vector.
    - Length must match expected (or config.DEFAULT_BITS if not specified)
    - Values should be in reasonable range (-10 to +10)
    """
    if expected_length is None:
        from . import config
        expected_length = config.DEFAULT_BITS
    if len(floats) != expected_length:
        return False
    
    for val in floats:
        if not isinstance(val, (int, float)):
            return False
        if val < -10 or val > 10:
            return False
    
    return True


def validate_label(label: int) -> bool:
    """Validate packet label (0-5 for 6-class classification)."""
    return label in (0, 1, 2, 3, 4, 5)


# ============================================================
#  FORMATTING HELPERS
# ============================================================
def format_bytes_hex(data: bytes, max_len: int = 16) -> str:
    """Format bytes as hex string."""
    if len(data) > max_len:
        return data[:max_len].hex() + "..."
    return data.hex()


def format_int_from_bytes(data: bytes) -> str:
    """Convert bytes to integer and format as hex."""
    n = int.from_bytes(data, "big")
    return hex(n)


def format_label(label: int) -> str:
    """Format label enum to human-readable name."""
    from . import config
    return config.LABEL_NAMES.get(label, f"UNKNOWN({label})")


def progress_bar(current: int, total: int, width: int = 30) -> str:
    """Generate a progress bar string."""
    if total == 0:
        return "0%"
    
    pct = current / total
    filled = int(width * pct)
    empty = width - filled
    
    bar = (
        f"{Colors.OK}{'█' * filled}{Colors.RESET}"
        f"{Colors.GRAY}{'░' * empty}{Colors.RESET}"
    )
    return f"[{bar}] {pct*100:.1f}%"


# ============================================================
#  STATISTICS & ANALYSIS
# ============================================================
def compute_statistics(values: List[float]) -> dict:
    """
    Compute basic statistics on a list of values.
    
    Returns:
        dict with mean, std, min, max
    """
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}
    
    import numpy as np
    arr = np.array(values)
    
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": len(values),
    }


# ============================================================
#  EXCEPTION CLASSES
# ============================================================
class PipelineException(Exception):
    """Base exception for pipeline errors."""
    pass


class GRPCConnectionError(PipelineException):
    """Error connecting to gRPC server."""
    pass


class DataValidationError(PipelineException):
    """Data validation failed."""
    pass


class BootstrapError(PipelineException):
    """Bootstrap test failed."""
    pass

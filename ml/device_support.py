#!/usr/bin/env python3
"""
Device Support Utility
=======================
Intelligent device detection and management for PyTorch.
Supports: CPU, CUDA (NVIDIA), MPS (Apple Silicon), other accelerators.
"""

import platform
import logging
from typing import Tuple, Optional

import torch

logger = logging.getLogger(__name__)


# ============================================================
#  DEVICE DETECTION & VALIDATION
# ============================================================
def get_available_devices() -> dict:
    """
    Detect all available compute devices.
    
    Returns:
        dict: {
            'cpu': bool,
            'cuda': bool,
            'cuda_count': int,
            'mps': bool,
            'mps_available': bool,
            'system': str,
            'pytorch_version': str
        }
    """
    info = {
        'cpu': True,  # Always available
        'cuda': torch.cuda.is_available(),
        'cuda_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'mps': False,
        'mps_available': False,
        'system': platform.system(),
        'pytorch_version': torch.__version__,
    }
    
    # MPS available on macOS with Apple Silicon (PyTorch 1.12+)
    if hasattr(torch.backends, 'mps'):
        info['mps'] = torch.backends.mps.is_available()
        info['mps_available'] = torch.backends.mps.is_available()
    
    return info


def validate_device_string(device_str: str) -> bool:
    """
    Validate device string before creation.
    
    Args:
        device_str: Device string ('cpu', 'cuda', 'cuda:0', 'mps', etc.)
    
    Returns:
        bool: True if device is available
    """
    device_str = device_str.strip().lower()
    
    if device_str == 'cpu':
        return True
    
    if device_str.startswith('cuda'):
        if not torch.cuda.is_available():
            return False
        if ':' in device_str:
            try:
                gpu_id = int(device_str.split(':')[1])
                return gpu_id < torch.cuda.device_count()
            except (ValueError, IndexError):
                return False
        return True
    
    if device_str == 'mps':
        return (hasattr(torch.backends, 'mps') and 
                torch.backends.mps.is_available())
    
    return False


def resolve_device(device_str: Optional[str] = None) -> Tuple[torch.device, str]:
    """
    Intelligently resolve device selection.
    
    Priority (if available):
    1. Explicitly specified device (if valid)
    2. MPS (Apple Silicon)
    3. CUDA (NVIDIA GPU)
    4. CPU (fallback)
    
    Args:
        device_str: User-specified device ('cpu', 'cuda', 'mps', 'auto', None)
    
    Returns:
        Tuple[torch.device, description_string]
    
    Raises:
        ValueError: If specified device is invalid/unavailable
    """
    if device_str and device_str.lower() == 'auto':
        device_str = None
    
    # User explicitly specified device
    if device_str:
        device_str = device_str.strip().lower()
        if not validate_device_string(device_str):
            raise ValueError(
                f"❌ Device '{device_str}' not available. "
                f"Check: torch.cuda.is_available() or torch.backends.mps.is_available()"
            )
        device = torch.device(device_str)
        desc = _get_device_description(device)
        return device, desc
    
    # Auto-detect: MPS > CUDA > CPU
    info = get_available_devices()
    
    # Priority 1: MPS (Apple Silicon)
    if info['mps_available']:
        device = torch.device('mps')
        desc = _get_device_description(device)
        return device, desc
    
    # Priority 2: CUDA (NVIDIA)
    if info['cuda']:
        device = torch.device('cuda:0')
        desc = _get_device_description(device)
        return device, desc
    
    # Fallback: CPU
    device = torch.device('cpu')
    desc = _get_device_description(device)
    return device, desc


def _get_device_description(device: torch.device) -> str:
    """Generate human-readable device description."""
    if device.type == 'cpu':
        return "CPU (PyTorch compute)"
    
    elif device.type == 'cuda':
        gpu_id = device.index or 0
        return f"CUDA GPU {gpu_id} ({torch.cuda.get_device_name(gpu_id)})"
    
    elif device.type == 'mps':
        return "Apple Metal Performance Shaders (MPS)"
    
    return str(device)


def print_device_info(device: torch.device):
    """Print detailed device information."""
    info = get_available_devices()
    
    logger.info("=" * 70)
    logger.info("COMPUTE DEVICE CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"System: {info['system']}")
    logger.info(f"PyTorch: {info['pytorch_version']}")
    logger.info(f"Selected Device: {_get_device_description(device)}")
    logger.info("-" * 70)
    
    logger.info("Available Devices:")
    logger.info(f"  ✓ CPU (always available)")
    
    if info['mps_available']:
        logger.info(f"  ✓ MPS (Apple Metal Performance Shaders)")
    else:
        logger.info(f"  ✗ MPS (not available on this system)")
    
    if info['cuda']:
        logger.info(f"  ✓ CUDA ({info['cuda_count']} GPU{'s' if info['cuda_count'] > 1 else ''})")
        for i in range(info['cuda_count']):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"      • {i}: {props.name}")
    else:
        logger.info(f"  ✗ CUDA (not available)")
    
    logger.info("=" * 70)


# ============================================================
#  DEVICE-AWARE UTILITIES
# ============================================================
def get_device_memory_info(device: torch.device) -> dict:
    """Get memory usage information for device."""
    info = {'device': str(device)}
    
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device.index or 0)
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        total = props.total_memory / 1024**3
        
        info.update({
            'type': 'CUDA GPU',
            'total_gb': total,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'free_gb': total - allocated
        })
    elif device.type == 'mps':
        info['type'] = 'Metal (MPS)'
        info['note'] = 'MPS memory managed automatically by macOS'
    else:
        import psutil
        vm = psutil.virtual_memory()
        info.update({
            'type': 'CPU + System RAM',
            'total_gb': vm.total / 1024**3,
            'available_gb': vm.available / 1024**3,
            'percent_used': vm.percent
        })
    
    return info


def empty_cache(device: torch.device):
    """Clear device cache if available."""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        # MPS manages memory automatically, no explicit cache clearing
        pass


# ============================================================
#  DEVICE-AWARE DTYPE SELECTION
# ============================================================
def get_optimal_dtype(device: torch.device, prefer_half: bool = False) -> torch.dtype:
    """
    Select optimal dtype for device.
    
    Args:
        device: torch.device
        prefer_half: Try to use float16 (MPS supports float16 well)
    
    Returns:
        torch.dtype (torch.float32 or torch.float16)
    """
    # MPS has good float16 support
    if prefer_half and device.type == 'mps':
        return torch.float16
    
    # CUDA can use float16 but needs careful handling
    if prefer_half and device.type == 'cuda':
        return torch.float16
    
    # Default to float32
    return torch.float32


# ============================================================
#  ARGPARSE HELPERS
# ============================================================
def add_device_argument(parser):
    """
    Add device argument to argparse parser.
    
    Usage:
        parser = argparse.ArgumentParser()
        add_device_argument(parser)
        args = parser.parse_args()
        device, desc = resolve_device(args.device)
    """
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cpu', 'cuda', 'cuda:0', 'mps', or 'auto' (default: auto-detect)",
    )


# ============================================================
#  EXAMPLE USAGE
# ============================================================
if __name__ == "__main__":
    import sys
    from utils import setup_logger
    
    setup_logger(__name__, "INFO")
    
    # Show all available devices
    info = get_available_devices()
    logger.info(f"\nDevice Info:\n{info}\n")
    
    # Resolve device (auto-detect)
    device, desc = resolve_device()
    logger.info(f"Auto-detected device: {desc}")
    print_device_info(device)
    
    # Show memory
    mem_info = get_device_memory_info(device)
    logger.info(f"\nMemory Info:\n{mem_info}")
    
    # Test tensor on device
    x = torch.randn(100, 100, device=device)
    logger.info(f"✓ Created test tensor on {device}: shape={x.shape}, dtype={x.dtype}")

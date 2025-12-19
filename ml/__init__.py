#!/usr/bin/env python3
"""
PRIME-X ML Pipeline
===================
Real-time machine learning pipeline for prime/composite classification.
Connects to Go Bridge microservice for cryptographic data streams.

Main Entry Point:
    python -m ml [command] [options]

Quick Start:
    1. Start Go Bridge: bridge/main.go
    2. Run bootstrap tests: python -m ml bootstrap
    3. Train model: python -m ml train --epochs 100
    4. Run inference: python -m ml inference --model-path model.pkl
"""

__version__ = "1.0.0"
__author__ = "PRIME-X Development Team"

# Import public API
from . import config
from . import utils
from . import core
from . import models
from . import data
from . import trainer
from . import tests

# Convenience imports
from .core import DataPacket, GRPCBridge, BufferedStreamReader, create_and_connect_bridge
from .models import BaseModel, ModelRegistry  # Plugin system exports
from .data import DataBatch, StreamDataIterator, Preprocessor
from .trainer import Trainer, InferenceEngine, MetricsTracker
from .tests import BootstrapTestSuite, ProtocolTests, MathematicalTests, StatisticalTests
from .utils import (
    print_banner, print_section, print_ok, print_err, print_warn, print_info,
    Colors, setup_logger
)

__all__ = [
    # Core modules
    "config",
    "utils",
    "core",
    "models",
    "data",
    "trainer",
    "tests",
    
    # Classes - Data
    "DataPacket",
    "DataBatch",
    
    # Classes - Networking
    "GRPCBridge",
    "BufferedStreamReader",
    
    # Classes - Models
    "PrimeClassifierNN",
    "PrimeClassifierTorch",
    
    # Classes - Data Pipeline
    "StreamDataIterator",
    "Preprocessor",
    
    # Classes - Training
    "Trainer",
    "InferenceEngine",
    "MetricsTracker",
    
    # Classes - Testing
    "BootstrapTestSuite",
    "ProtocolTests",
    "MathematicalTests",
    "StatisticalTests",
    
    # Utilities
    "Colors",
    "setup_logger",
    
    # Functions
    "create_model",
    "create_and_connect_bridge",
    "print_banner",
    "print_section",
    "print_ok",
    "print_err",
    "print_warn",
    "print_info",
]

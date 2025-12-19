#!/usr/bin/env python3
"""
ML Configuration Module
=======================
Centralized configuration management.
Loads config.json from project root, applies environment variable overrides.
"""

import json
import os
import sys
from pathlib import Path

# ============================================================
#  PROTOBUF PATH SETUP
# ============================================================
# Inject bridge/pb into sys.path for protobuf imports
_project_root = Path(__file__).parent.parent
_pb_path = _project_root / "bridge" / "pb"
if str(_pb_path) not in sys.path:
    sys.path.insert(0, str(_pb_path))

# ============================================================
#  LOAD CONFIG FROM PROJECT ROOT
# ============================================================
_config_file = _project_root / "config" / "config.json"

if not _config_file.exists():
    raise FileNotFoundError(
        f"config.json not found at {_config_file}. "
        f"Must be in config/ directory."
    )

with open(_config_file, 'r') as f:
    _config = json.load(f)

# ============================================================
#  SYSTEM PARAMETERS
# ============================================================
PRIME_BITS = _config['system']['prime_bits']
RAW_SIZE_BYTES = _config['system']['raw_size_bytes']

# ============================================================
#  NETWORK CONFIGURATION (with env overrides)
# ============================================================
GRPC_SERVER_HOST = os.getenv("GRPC_HOST", _config['network']['grpc_host'])
GRPC_SERVER_PORT = int(os.getenv("GRPC_PORT", _config['network']['grpc_port']))
GRPC_SERVER_ADDRESS = f"{GRPC_SERVER_HOST}:{GRPC_SERVER_PORT}"

ZMQ_HOST = os.getenv("ZMQ_HOST", _config['network']['zmq_host'])
ZMQ_PORT_CORE_PUSH = int(os.getenv("ZMQ_PORT_PUSH", _config['network']['zmq_port_core_push']))
ZMQ_PORT_CORE_PUB = int(os.getenv("ZMQ_PORT_PUB", _config['network']['zmq_port_core_pub']))

# ============================================================
#  GRPC STREAMING CONFIGURATION
# ============================================================
DEFAULT_BATCH_SIZE = _config['model']['batch_size']
DEFAULT_MIXING_RATIO = 0.5  # 50% Prime, 50% Composite
GRPC_STREAM_TIMEOUT = 30  # seconds
GRPC_STREAM_RETRY_INTERVAL = 2  # seconds
GRPC_MAX_RETRIES = 5

# ============================================================
#  DATA PROTOCOL
# ============================================================
DEFAULT_RAW_SIZE = RAW_SIZE_BYTES
DEFAULT_BITS = PRIME_BITS
FLOAT_VECTOR_SIZE = DEFAULT_BITS * 4  # bytes

# Data labels (from config)
LABEL_EASY_COMP = _config['labels']['easy_composite']
LABEL_PRIME = _config['labels']['prime']
LABEL_HARD_COMP = _config['labels']['hard_composite']
LABEL_DP_PRIME = _config['labels']['dp_prime']
LABEL_DP_COMPOSITE = _config['labels']['dp_composite']
LABEL_DP_HARD_COMP = _config['labels']['dp_hard_composite']

LABEL_NAMES = {
    LABEL_EASY_COMP: "EASY_COMP",
    LABEL_PRIME: "PRIME",
    LABEL_HARD_COMP: "HARD_COMP",
    LABEL_DP_PRIME: "DP_PRIME",
    LABEL_DP_COMPOSITE: "DP_COMPOSITE",
    LABEL_DP_HARD_COMP: "DP_HARD_COMP",
}

# ============================================================
#  CRYPTOGRAPHIC PARAMETERS
# ============================================================
NOISE_SIGMA = _config['crypto']['noise_sigma']
MILLER_RABIN_ROUNDS = _config['crypto']['miller_rabin_rounds']
CHACHA20_KEY_SIZE = _config['crypto']['chacha20_key_size']
BLAKE2S_OUTPUT_BITS = _config['crypto']['blake2s_output_bits']

# ============================================================
#  BUFFER CONFIGURATION
# ============================================================
CPP_INTERNAL_QUEUE_MAX = _config['buffer']['cpp_internal_queue_max']
GO_BUFFER_SIZE = _config['buffer']['go_channel_buffer_size']
PYTHON_QUEUE_BUFFER_SIZE = _config['buffer']['python_queue_buffer_size']
DATA_PREFETCH_COUNT = _config['buffer']['python_prefetch_batches']

# ============================================================
#  MODEL CONFIGURATION
# ============================================================
MODEL_INPUT_DIM = _config['model']['input_dimension']
MODEL_HIDDEN_DIM = _config['model']['hidden_dimension']
MODEL_NUM_LAYERS = _config['model']['num_layers']
MODEL_DROPOUT = _config['model']['dropout']
MODEL_LEARNING_RATE = _config['model']['learning_rate']
MODEL_WEIGHT_DECAY = _config['model']['weight_decay']

# CRITICAL VALIDATION: input_dimension must equal prime_bits
# (Each bit of the RSA composite becomes one feature in the ML model)
if MODEL_INPUT_DIM != DEFAULT_BITS:
    raise ValueError(
        f"Model configuration error: input_dimension ({MODEL_INPUT_DIM}) must equal "
        f"prime_bits ({DEFAULT_BITS}). Both control the same 'bits' concept. "
        f"Update config.json: model.input_dimension = {DEFAULT_BITS}"
    )

# Training parameters
BATCH_SIZE = _config['model']['batch_size']
EPOCHS = _config['model']['epochs']
CHECKPOINT_SAVE_INTERVAL = _config['model']['checkpoint_save_interval']

# Validation
VALIDATION_SPLIT = 0.2

# ============================================================
#  DP DETECTION CONFIGURATION
# ============================================================
DP_DETECTION_CONFIG = _config.get('dp_detection', {
    "mid_threshold": 0.80,
    "entropy_threshold": 0.20,
    "mean_range_min": 0.35,
    "mean_range_max": 0.65,
    "low_mid_threshold": 0.50,
    "low_entropy_threshold": 0.10,
})

# ============================================================
#  DATA PIPELINE
# ============================================================
DATA_BUFFER_SIZE = PYTHON_QUEUE_BUFFER_SIZE
DATA_WORKER_THREADS = 2

# ============================================================
#  STORAGE
# ============================================================
CHECKPOINT_DIR = os.getenv("ML_CHECKPOINT_DIR", _config['storage']['checkpoint_dir'])
DB_PATH = _config['storage']['db_path']

# Create checkpoint dir if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================
#  LOGGING & OUTPUT
# ============================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", _config['logging']['level'])
LOG_FORMAT = _config['logging']['format']
LOG_FILE = os.getenv("LOG_FILE", None)

# ============================================================
#  BOOTSTRAP / TESTING CONFIGURATION
# ============================================================
BOOTSTRAP_SAMPLES = _config['bootstrap']['sample_count']
BOOTSTRAP_PROTOCOL_CHECK_SAMPLES = _config['bootstrap']['protocol_check_samples']
BOOTSTRAP_MATH_TEST_SAMPLES = _config['bootstrap']['math_test_samples']
BOOTSTRAP_STRESS_TEST_DURATION = _config['bootstrap']['stress_test_duration_seconds']
BOOTSTRAP_MIN_THROUGHPUT = _config['bootstrap']['min_throughput_packets_per_sec']

# Assertion strictness
STRICT_MODE = True
HASH_VERIFICATION_ENABLED = True
MATHEMATICAL_VALIDATION_ENABLED = True

# ============================================================
#  ADVANCED OPTIONS
# ============================================================
DEVICE = "cpu"  # "cpu" or "cuda"
SEED = 42
VERBOSITY_LEVEL = 2  # 0=quiet, 1=normal, 2=verbose

# ============================================================
#  DEBUG: Print loaded configuration
# ============================================================
if os.getenv("ML_DEBUG_CONFIG"):
    print(f"[CONFIG] Loaded from {_config_file}")
    print(f"[CONFIG] System: {PRIME_BITS}-bit, {RAW_SIZE_BYTES} bytes")
    print(f"[CONFIG] gRPC: {GRPC_SERVER_ADDRESS}")
    print(f"[CONFIG] ZMQ: {ZMQ_HOST}:{ZMQ_PORT_CORE_PUSH}")
    print(f"[CONFIG] Noise σ: {NOISE_SIGMA}, MR rounds: {MILLER_RABIN_ROUNDS}")

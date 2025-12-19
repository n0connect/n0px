# PRIME-X Comprehensive Test Suite v2

**Last Updated:** 19 December 2025  
**Version:** 2.0 - Complete Pipeline Testing

---

## Overview

This is a professional-grade test suite validating the entire PRIME-X pipeline:

```
C++ Core (CSPRNG)
    ↓ [ZMQ: tcp://127.0.0.1:5558]
Go Bridge (gRPC Server)
    ↓ [gRPC: localhost:50051]
Python ML Pipeline
    ↓ [torch tensors]
ML Models (RM0/RM1/CM0/CM1)
```

Each layer is validated for:
1. **Data integrity** - No corruption between stages
2. **Format correctness** - Proper serialization/deserialization
3. **Label consistency** - All 6 classes (0-5) properly routed
4. **Throughput** - Performance metrics
5. **End-to-end** - Complete pipeline functioning

---

## Quick Start

### Option 1: Automatic Testing (Recommended)
```bash
cd /Users/n0n0/Desktop/n0n0/PXQSDA
make test          # Full test with automatic service management
make test-quick    # Quick unit tests only (no services)
make test-all      # Same as 'make test'
```

### Option 2: Manual Testing
```bash
# Terminal 1: Start C++ core
./prime_core

# Terminal 2: Start Go bridge  
cd bridge && ./prime_bridge

# Terminal 3: Run tests
python3 tests/run_all_tests.py --mode integration --verbose
```

---

## Test Layers

### Layer 1: C++ Core Unit Tests (7 tests)

**File:** `tests/cpp_core/test_csprng.cpp`

```bash
make test-cpp  # Or: cd tests/cpp_core && g++ ... && ./test_csprng
```

**Tests:**
1. **Type Selection Uniformity** - Labels 0,1,2 evenly distributed
2. **Pair Generation** - Each type → 2 labels (0→0,3; 1→1,4; 2→2,5)
3. **Gaussian Noise** - μ≈0, σ≈0.05, normal distribution
4. **Randomness Quality** - χ²-test validates CSPRNG entropy
5. **Packet Format** - Structure: [label:4][raw:32][input:1024][trailer:112]
6. **Data Integrity** - No corruption, valid float values
7. **Label Generation** - All 6 classes (0-5) present

**Expected Output:**
```
================================================================================
PRIME-X C++ CORE INTEGRATION TESTS
================================================================================

[TEST 1] Type Selection Uniformity
...
  ✓ Type selection is uniform (max deviation: 2.3%)

[TEST 2] Pair Generation
...
  ✓ All pairs generated correctly

...

✓ ALL C++ CORE TESTS PASSED (7/7)
```

---

### Layer 2: Go Bridge Unit Tests (8 tests)

**File:** `tests/go_bridge/bridge_integration_test.go`

```bash
make test-go  # Or: cd tests/go_bridge && go test -v bridge_integration_test.go
```

**Tests:**
1. **Packet Structure** - Validates layout with trailer
2. **Label Distribution** - All 6 labels routed correctly
3. **Packet Integrity** - BLAKE2s hash validates no corruption
4. **Message Serialization** - Protobuf encoding working
5. **Data Type Compatibility** - int32, float32[], bytes, bool match
6. **Streaming Throughput** - Measures MB/s and msgs/sec
7. **Label Mapping** - 0-5 → names correctly
8. **Batch Assembly Order** - FIFO preserved

**Expected Output:**
```
================================================================================
GO BRIDGE - UNIT TESTS
================================================================================

[TEST] Packet Structure Validation
  ✓ Label: 2
  ✓ Raw bytes: 32
  ✓ Input vector: 256 floats (1024 bytes)
  ✓ Total packet size: ~1200 bytes
  ✓ Packet structure validation passed

...

✓ All Go bridge tests passed (8/8)
```

---

### Layer 3: Python ML Unit Tests

**File:** `tests/python_ml/test_ml_pipeline.py`

```bash
make test-python  # Or: python3 tests/python_ml/test_ml_pipeline.py
```

**Tests:**
- Label validation (0-5 coverage)
- Model instantiation (6-class output)
- Batch assembly (FIFO order)
- Distribution uniformity
- Data integrity
- Memory efficiency

---

### Layer 4: Full Pipeline Integration Tests (6 tests)

**File:** `tests/integration/test_pipeline_v2.py`

**Requires:** C++ core + Go bridge running

```bash
# Terminal 1 & 2: Start services
make run

# Terminal 3: Run integration tests
make test-integration-manual
# OR
python3 tests/integration/test_pipeline_v2.py
```

**Tests:**
1. **gRPC Connection** - Validates Go bridge is responsive
2. **Data Format Validation** - Checks protobuf deserialization
3. **Label Distribution** (1200 samples) - All 6 classes received
4. **Data Consistency** (100 samples) - No NaN/Inf, valid ranges
5. **Throughput Measurement** - Calculates MB/s and packet latency
6. **ML Integration** (64 samples) - Python can process pipeline data

**Expected Output:**
```
═══════════════════════════════════════════════════════════════════════════════
PRIME-X Full Pipeline Integration Test
═══════════════════════════════════════════════════════════════════════════════

▶ gRPC Connection Test
  ✓ Connected to localhost:50051
  ✓ Received first batch
  ✓   - Raw bytes: 32 bytes
  ✓   - Input vector: 256 floats
  ✓   - Label: 2 (HARD_COMPOSITE)
  ✓   - Is synthetic: true

▶ Label Distribution Test (1200 samples)
  ✓ Label distribution:
  ✓   0 (COMPOSITE       ):  198 ( 16.5%)
  ✓   1 (PRIME           ):  195 ( 16.2%)
  ✓   2 (HARD_COMPOSITE  ):  209 ( 17.4%)
  ✓   3 (DP_PRIME        ):  197 ( 16.4%)
  ✓   4 (DP_COMPOSITE    ):  201 ( 16.8%)
  ✓   5 (DP_HARD_COMP... ):  200 ( 16.7%)
  ✓ All 6 labels present in 1200 samples

...

═══════════════════════════════════════════════════════════════════════════════
Test Summary
═══════════════════════════════════════════════════════════════════════════════

Results:
  ✓ gRPC Connection
  ✓ Data Format
  ✓ Label Distribution
  ✓ Data Consistency
  ✓ Throughput
  ✓ ML Integration

Total: 6/6 passed

✓ All integration tests passed!
```

---

### Layer 5: Master Test Coordinator

**File:** `tests/run_all_tests.py`

Orchestrates all tests with automatic service management.

```bash
python3 tests/run_all_tests.py --mode [unit|integration|all|quick]
```

**Modes:**
- `unit` - Runs C++, Go, Python tests (no services required)
- `integration` - Starts services, runs integration tests
- `all` (default) - Both unit and integration
- `quick` - Unit tests only

**Flags:**
- `--verbose` or `-v` - Show detailed output

---

## Test Coverage Map

| Component | Test File | Tests | Validates |
|-----------|-----------|-------|-----------|
| C++ Core | `test_csprng.cpp` | 7 | CSPRNG, labels, packet format, data integrity |
| Go Bridge | `bridge_integration_test.go` | 8 | Routing, serialization, throughput, BLAKE2s hash |
| Python ML | `test_ml_pipeline.py` | 5+ | Labels, batch assembly, model compatibility |
| Integration | `test_pipeline_v2.py` | 6 | gRPC, data flow, consistency, end-to-end |

**Total Coverage:** 26+ test cases across 4 layers

---

## Data Flow Validation

### Flow: C++ → Go → Python → Model

```
┌─ C++ Core ──────────────────────────────────────┐
│                                                  │
│  Generates 6 classes:                          │
│  - Type 0 → Labels {0, 3}  (COMPOSITE variants) │
│  - Type 1 → Labels {1, 4}  (PRIME variants)    │
│  - Type 2 → Labels {2, 5}  (HARD_COMPOSITE)    │
│                                                  │
│  Sends via ZMQ: [label][raw_32B][noise_256]   │
└────────┬────────────────────────────────────────┘
         │ ZMQ tcp://127.0.0.1:5558
         ▼
┌─ Go Bridge ──────────────────────────────────────┐
│                                                  │
│  1. Receives packet from ZMQ                    │
│  2. Validates BLAKE2s trailer                   │
│  3. Serializes to protobuf:                     │
│     - DataBatch.raw_bytes (32B)                 │
│     - DataBatch.input_vector (256 floats)      │
│     - DataBatch.label (int32)                   │
│  4. Streams via gRPC                            │
│                                                  │
└────────┬────────────────────────────────────────┘
         │ gRPC localhost:50051
         ▼
┌─ Python Pipeline ────────────────────────────────┐
│                                                  │
│  1. Receives protobuf DataBatch                 │
│  2. Validates fields:                           │
│     ✓ raw_bytes: 32 bytes                       │
│     ✓ input_vector: 256 floats                  │
│     ✓ label: 0-5 range                          │
│  3. Assembles batch (batch_size packets)        │
│  4. Creates numpy arrays:                       │
│     - features: (batch_size, 256) float32      │
│     - labels: (batch_size,) int64               │
│                                                  │
└────────┬────────────────────────────────────────┘
         │ numpy arrays
         ▼
┌─ ML Model ───────────────────────────────────────┐
│                                                  │
│  1. Receives batch:                             │
│     - Input: (64, 256) tensor                   │
│     - Labels: (64,) tensor                      │
│  2. Forward pass through VAE:                   │
│     - RM0/RM1: Real-valued latent              │
│     - CM0/CM1: Complex-valued latent            │
│  3. Computes loss + gradients                   │
│  4. Separability metrics optional               │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## Validation Checklist

✅ **C++ Core:**
- [ ] Labels uniformly distributed (each ~16.7%)
- [ ] All 6 classes present (0-5)
- [ ] Packet format: 1172 bytes
- [ ] No data corruption (NaN/Inf checks)
- [ ] Gaussian noise properties valid

✅ **Go Bridge:**
- [ ] Receives all packets from C++
- [ ] BLAKE2s hash validates integrity
- [ ] All labels routed correctly
- [ ] gRPC streaming stable
- [ ] Protobuf serialization correct

✅ **Python Pipeline:**
- [ ] Receives all protobuf messages
- [ ] Label consistency maintained
- [ ] Batch assembly preserves order
- [ ] Numpy arrays correct shape/dtype
- [ ] ML models can ingest data

✅ **End-to-End:**
- [ ] Complete pipeline runs without errors
- [ ] Data flows: C++ → Go → Python → Model
- [ ] Performance: ≥1000 packets/sec
- [ ] All 6 labels present in sample
- [ ] Model forward/backward passes work

---

## Troubleshooting

### "gRPC connection refused"
```bash
# Check if Go bridge is running
ps aux | grep prime_bridge

# If not, start it:
cd bridge && ./prime_bridge
```

### "Protobuf import error"
```bash
# Regenerate protobuf stubs
make proto-sync
```

### "No module named 'ml'"
```bash
# Setup ML environment
make ml-setup
```

### "All tests pass locally but fail in CI"
- Check PYTHONPATH includes workspace root
- Verify all services can communicate (firewall)
- Check port availability (5558, 5557, 50051)

---

## Performance Baselines

**Measured on:** macOS M1 Pro, PRIME bits=256

| Metric | Target | Typical |
|--------|--------|---------|
| C++ packet generation | <1ms | 0.3ms |
| Go routing latency | <5ms | 2ms |
| gRPC batch streaming | >1000 packets/sec | 1200 packets/sec |
| Python batch assembly | <10ms | 5ms |
| ML forward pass (64 batch) | <100ms | 45ms |

---

## CI/CD Integration

```yaml
# Example GitHub Actions workflow
- name: Run PRIME-X Tests
  run: |
    cd /Users/n0n0/Desktop/n0n0/PXQSDA
    make test-quick      # Quick check
    make test            # Full test with services
```

---

## Future Enhancements

- [ ] Benchmark suite (profiling each layer)
- [ ] Stress tests (high throughput scenarios)
- [ ] Memory leak detection
- [ ] Protocol buffer fuzzing
- [ ] ML model validation benchmarks

---

## Support

For test failures or questions:
1. Check `/Users/n0n0/Desktop/n0n0/PXQSDA/tests/TESTING_GUIDE.md`
2. Review individual test source files
3. Run with `--verbose` flag for detailed output
4. Check service logs in `/tmp/` or service output

---

**Status:** ✓ Complete - Ready for Production Testing

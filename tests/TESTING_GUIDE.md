# PRIME-X Testing Guide

## Overview
Professional test suite for PRIME-X 6-class randomness pipeline.

- **C++ Core**: CSPRNG validation (ChaCha20)
- **Go Bridge**: Routing + distribution testing  
- **Python ML**: Model + pipeline testing
- **Integration**: End-to-end pipeline validation

---

## Quick Start

### 1. Build System
```bash
cd /Users/n0n0/Desktop/n0n0/PXQSDA
make proto-go proto-py all ml-setup
```

### 2. Run Unit Tests

#### C++ Tests
```bash
cd tests/cpp_core
g++ -std=c++17 -O2 test_csprng.cpp -o test_csprng -lssl -lcrypto -lgmp
./test_csprng
```

**Expected**: 5/5 tests pass
- Type selection uniformity
- Pair generation correctness
- Gaussian noise properties
- Chi-square randomness
- Packet format validation

#### Go Tests
```bash
cd tests/go_bridge
go test -v test_routing.go -run TestPacketVerify
go test -v test_routing.go -run TestRoutingCoverage
go test -v test_routing.go -run TestDistribution
go test -v test_routing.go -run TestHysteresis
go test -v test_routing.go -run TestLabelPassthrough
go test -v test_routing.go -run TestHashIntegrity
```

**Expected**: 6/6 unit tests + benchmarks
- Packet hash verification (BLAKE2s)
- All 6 types routed correctly
- Label distribution (33.33% per channel)
- Buffer hysteresis working
- Labels pass through unchanged
- Hash integrity verified

#### Python Tests
```bash
cd /Users/n0n0/Desktop/n0n0/PXQSDA
python3 -m pytest tests/python_ml/test_ml_pipeline.py -v
```

**Expected**: 18+ tests pass across 6 classes
- Label range validation
- Model 6-class output shape
- Softmax preservation
- Batch FIFO ordering
- Chi-square distribution test
- Data integrity check
- Memory efficiency

---

## Integration Test (Full Pipeline)

### Prerequisites (3 Terminals)

**Terminal 1 - C++ Core**
```bash
cd /Users/n0n0/Desktop/n0n0/PXQSDA
./prime_core
```
Expected output:
```
[ChaCha20-CSPRNG] Initializing...
[Worker 0] Starting generation loop
[Worker 1] Starting generation loop
...
[Sender] Pushing types to ZMQ...
```

**Terminal 2 - Go Bridge**
```bash
cd /Users/n0n0/Desktop/n0n0/PXQSDA
go run bridge/main.go
```
Expected output:
```
[zmqIngestor] Listening on tcp://127.0.0.1:5555
[gRPCServer] Listening on :50051
[router] Routing packets to channels...
```

**Terminal 3 - Python Integration Test**
```bash
python3 tests/integration/test_pipeline.py
```

### What the Integration Test Does

1. **Connection Test** (10 sec)
   - Verify gRPC connection to Go bridge
   - Check basic packet reception

2. **Label Coverage** (5-10 sec)
   - Verify all 6 labels appear
   - Min threshold: 1 packet per label in 100 samples

3. **Distribution Test** (60-120 sec) ⭐ **MAIN TEST**
   - Collect 6000 packets
   - Chi-square test for uniformity
   - Expected: p-value > 0.05 (uniform distribution)
   - Each label should have ~16.67%

4. **Data Integrity** (10 sec)
   - Verify packet structure
   - Check for NaN/Inf values
   - Validate raw bytes size (32 bytes)
   - Validate vector size (256 floats)

5. **Performance** (30-60 sec)
   - Measure throughput (packets/sec)
   - Latency percentiles (P50, P95, P99)
   - Expected: >100 packets/sec on modern hardware

### Expected Results

```
[TEST 1] gRPC Connection Stability
  ✓ Connection successful
  ✓ Received first batch: 256 floats

[TEST 2] Label Coverage (n=100 packets)
  ✓ Label 0 (COMPOSITE): 16 (16.0%)
  ✓ Label 1 (PRIME): 17 (17.0%)
  ✓ Label 2 (HARD_COMPOSITE): 17 (17.0%)
  ✓ Label 3 (DP_PRIME): 17 (17.0%)
  ✓ Label 4 (DP_COMPOSITE): 16 (16.0%)
  ✓ Label 5 (DP_HARD_COMPOSITE): 17 (17.0%)
  ✓ All 6 labels present

[TEST 3] Distribution Uniformity (χ² test, n=6000)
  Chi-square statistic: 3.4567
  P-value: 0.751234
  ✓ Distribution is uniform (p=0.751234 > 0.05)

[TEST 4] Data Integrity (n=100 packets)
  ✓ invalid_label: 0
  ✓ wrong_raw_size: 0
  ✓ wrong_vector_size: 0
  ✓ nan_values: 0
  ✓ inf_values: 0
  ✓ All packets valid

[TEST 5] Throughput & Latency (n=1000 packets)
  Throughput: 215 packets/sec
  Latency statistics (ms):
    Mean: 4.67
    P95: 8.23
    P99: 12.45
  ✓ Performance measured

SUMMARY
  Connection: ✓ PASS
  Coverage: ✓ PASS
  Uniformity: ✓ PASS (p=0.751234)
  Integrity: ✓ PASS
  Throughput: 215 pkt/sec

✓ ALL INTEGRATION TESTS PASSED
```

---

## Critical Metrics

### Chi-Square Test Interpretation
- **p-value > 0.05**: Distribution is uniform ✓
- **p-value ≤ 0.05**: Distribution is NOT uniform ✗
  
For 6 labels with 6000 samples:
- Expected count per label: 1000
- Chi-square distribution: df = 5
- Critical value (α=0.05): 11.07

### Label Distribution Expected
Each label should appear in ~16.67% (1000/6000) of samples.

Acceptable range (95% confidence):
```
Lower: 16.67% - 2.5% ≈ 14.17%
Upper: 16.67% + 2.5% ≈ 19.17%
```

### Performance Expectations
| Metric | Target | Minimum | Status |
|--------|--------|---------|--------|
| Throughput | >200 pkt/sec | >100 pkt/sec | Measured |
| P95 Latency | <10 ms | <50 ms | Measured |
| P99 Latency | <20 ms | <100 ms | Measured |
| Packet loss | 0% | 0% | Verify |

---

## Troubleshooting

### "Cannot connect to Go bridge"
- Verify Terminal 2 is running: `ps aux | grep main.go`
- Check port 50051 is open: `lsof -i :50051`
- Verify gRPC binary is compiled: `go build bridge/main.go`

### "Distribution is NOT uniform (p ≤ 0.05)"
- Check Go routing fix is applied (main.go line 150+)
- Verify all 6 types in switch statement: `case 0,4:` `case 1,3:` `case 2,5:`
- Collect more samples (6000+) for better statistics

### "Invalid label received"
- Check Python DataPacket label range (0-5)
- Verify Go StreamData gRPC is sending correct labels
- Check C++ enum values match (types 0-5)

### "NaN/Inf values detected"
- Check Gaussian noise generation in C++
- Verify float normalization in Go
- Check Python model softmax operation

### Low throughput (<100 pkt/sec)
- Check system load: `top`
- Verify no blocking in Go zmqIngestor
- Check Python model inference time
- Profile with benchmarks

---

## Test Statistics

### Sample Sizes
- **Unit Tests**: Fast (< 1 sec each)
- **Label Coverage**: 100 packets (< 10 sec)
- **Distribution Test**: 6000 packets (60-120 sec) ⭐
- **Integrity Test**: 100 packets (< 10 sec)
- **Performance Test**: 1000 packets (30-60 sec)

**Total Integration Test**: ~2-3 minutes

### Statistical Significance
- Chi-square test: α = 0.05 (5% significance level)
- Sample size: 6000 > 5*100 (sufficient for chi-square)
- Expected value per cell: 1000 > 5 (chi-square assumption met)

---

## Test Coverage

### Unit Test Coverage
```
C++ Core:
  ✓ Type selection (10K samples)
  ✓ Pair generation (all 3 pairs)
  ✓ Gaussian noise (μ≈0, σ≈0.05)
  ✓ Chi-square test (p > 0.05)
  ✓ Packet format (1076 bytes)

Go Bridge:
  ✓ Packet hash verification
  ✓ Routing coverage (all 6 types)
  ✓ Channel distribution (33.33% each)
  ✓ Buffer hysteresis
  ✓ Label passthrough
  ✓ Hash integrity

Python ML:
  ✓ Label validation (0-5)
  ✓ Model shape (batch, 6)
  ✓ Softmax preservation
  ✓ FIFO ordering
  ✓ Distribution chi-square
  ✓ Data integrity
  ✓ Memory efficiency
```

### Integration Test Coverage
```
Pipeline Flow:
  ✓ C++ → ZMQ (labels 0-5)
  ✓ Go routing (all 3 channels)
  ✓ gRPC streaming (6 batches)
  ✓ Python reception (no shuffle)
  ✓ Label preservation (end-to-end)

Data Quality:
  ✓ No packet loss
  ✓ Distribution uniformity
  ✓ Data integrity
  ✓ No corruption

Performance:
  ✓ Throughput
  ✓ Latency
  ✓ Memory usage
```

---

## Continuous Testing

### Run After Each Change
```bash
# After code changes to Go
cd tests/go_bridge && go test -v test_routing.go

# After code changes to Python
python3 -m pytest tests/python_ml/test_ml_pipeline.py -v

# After code changes to C++
cd tests/cpp_core && g++ -O2 test_csprng.cpp -o test_csprng && ./test_csprng
```

### Nightly Full Suite
```bash
#!/bin/bash
cd /Users/n0n0/Desktop/n0n0/PXQSDA

echo "Building system..."
make proto-go proto-py all ml-setup

echo "Running C++ tests..."
cd tests/cpp_core
g++ -std=c++17 -O2 test_csprng.cpp -o test_csprng -lssl -lcrypto -lgmp
./test_csprng || exit 1

echo "Running Go tests..."
cd ../go_bridge
go test -v test_routing.go || exit 1

echo "Running Python tests..."
cd /Users/n0n0/Desktop/n0n0/PXQSDA
python3 -m pytest tests/python_ml/test_ml_pipeline.py -v || exit 1

echo "✓ ALL TESTS PASSED"
```

---

## Documentation References

- **C++ ChaCha20**: `prime_core.cpp` - Worker thread type generation
- **Go Routing**: `bridge/main.go` - zmqIngestor() routing logic
- **Python Pipeline**: `ml/core.py` - DataPacket class definition
- **Protocol Buffers**: `bridge/pb/prime_bridge.proto` - Message definitions
- **Test Data Flow**: Each test collects real data from running system

---

## Success Criteria

✅ **All tests must pass** for production deployment:

1. ✓ C++ unit tests: 5/5 pass
2. ✓ Go unit tests: 6/6 pass  
3. ✓ Python unit tests: 18+/18+ pass
4. ✓ Integration test connection: successful
5. ✓ Integration test coverage: all 6 labels
6. ✓ Integration test uniformity: p > 0.05 (chi-square)
7. ✓ Integration test integrity: 0 issues
8. ✓ Integration test throughput: > 100 pkt/sec

---

**Last Updated**: Session 6 - Test Suite Implementation
**Test Framework**: Pytest (Python), go test (Go), Custom (C++)
**Tested On**: macOS, Linux
**Python Version**: 3.9+
**Go Version**: 1.25+
**C++ Version**: C++17

# n0px Protocol Specification

**Version:** 1.0  
**Date:** December 2025  
**Status:** Experimental Only

---

## Table of Contents

1. [Protocol Overview](#protocol-overview)
2. [Packet Format](#packet-format)
3. [Message Types](#message-types)
4. [Data Types & Encoding](#data-types--encoding)
5. [Error Handling](#error-handling)
6. [Communication Flows](#communication-flows)
7. [Security Considerations](#security-considerations)

---

## Protocol Overview

PX-QSDA uses a **three-layer communication protocol**:

```
Layer 1: C++ → Go       [ZMQ PUSH/PULL]  Raw binary packets (1172 bytes)
Layer 2: Go → Python    [gRPC Streaming]  Protocol Buffer messages
Layer 3: Go ← C++       [ZMQ PUB/SUB]    Backpressure commands (TEXT)
```

### Transport Protocols

| Layer | Pattern | Transport | Port | Reliability |
|-------|---------|-----------|------|-------------|
| C++→Go | PUSH/PULL | TCP | 5558 | At-most-once (datagrams) |
| Go→Python | Streaming RPC | HTTP/2 | 50051 | Ordered, reliable |
| Go←C++ | PUB/SUB | TCP | 5557 | Fire-and-forget |

---

## Packet Format

### Layer 1: Binary Packet (1172 bytes)

Used for **C++ Core ↔ Go Bridge** communication over ZMQ.

```
Offset  Size   Field           Type        Byte Order  Description
──────────────────────────────────────────────────────────────────
0       4      TYPE            int32       LE          Data type: 0-5
4       32     RAW             bytes       BE          256-bit integer
36      1024   FLOATS          float32[256]LE         Float vector
1060    4      MAGIC           char[4]     N/A         "PXSV"
1064    1      VERSION         uint8       N/A         Format version=1
1065    3      RESERVED        uint8[3]    N/A         Padding
1068    8      SEQUENCE        uint64      LE          Packet sequence number
1076    32     H_RAW           bytes       HEX         BLAKE2s(RAW)
1108    32     H_VEC           bytes       HEX         BLAKE2s(FLOATS)
1140    32     H_ALL           bytes       HEX         BLAKE2s(metadata+all)
──────────────────────────────────────────────────────────────────
TOTAL: 1172 bytes
```

### Byte Order Convention

- **LE (Little-Endian)**: Intel convention (0x1234 → 0x34 0x12)
- **BE (Big-Endian)**: Network byte order (0x1234 → 0x12 0x34)
- **HEX**: Hexadecimal representation (32 bytes → 64 chars)
- **N/A**: Fixed-width fields (no byte order applicable)

### Example Packet (Annotated)

```
[1] TYPE = 0x00000001 (LE) → TYPE_PRIME
    Bytes: 01 00 00 00

[2] RAW = 256-bit prime number
    Bytes: 00 00 00 00 ... DD EE FF (32 bytes, BE)

[3] FLOATS = 256 × float32 (IEEE754, LE)
    Bytes: 3F 80 00 00 3F 40 00 00 ... (1024 bytes)
           [    1.0  ] [    0.75 ] ...

[4] MAGIC = "PXSV"
    Bytes: 50 58 53 56

[5] VERSION = 0x01
    Bytes: 01

[6] RESERVED = 0x00 0x00 0x00
    Bytes: 00 00 00

[7] SEQUENCE = 0x000000001234ABCD (LE)
    Bytes: CD AB 34 12 00 00 00 00

[8] H_RAW = BLAKE2s-256(RAW)
    Bytes: A1 B2 C3 D4 ... (32 bytes)

[9] H_VEC = BLAKE2s-256(FLOATS)
    Bytes: E5 F6 G7 H8 ... (32 bytes)

[10] H_ALL = BLAKE2s-256(TYPE||RAW||FLOATS||SEQ||rawsize||bits)
     Bytes: I9 J0 K1 L2 ... (32 bytes)
```

---

## Message Types

### Type Field Values (4 bytes, little-endian int32)

```c
TYPE_EASY_COMPOSITE     = 0   // Easy composite (non-prime, random)
TYPE_PRIME              = 1   // Prime number
TYPE_HARD_COMPOSITE     = 2   // Hard composite (p×q, RSA-style)

TYPE_DP_EASY_COMPOSITE  = 4   // DP-perturbed easy composite
TYPE_DP_PRIME           = 3   // DP-perturbed prime
TYPE_DP_HARD_COMPOSITE  = 5   // DP-perturbed hard composite
```

**Note:** Raw types (0,1,2) and DP types (3,4,5) have **identical numeric values** 
but are emitted twice per generation for differential privacy simulation.

### Type Routing (Go Bridge)

```
TYPE → Channel Mapping:
  0, 4 → chanEasy       (Easy Composite)
  1, 3 → chanPrime      (Prime)
  2, 5 → chanHard       (Hard Composite)
```

---

## Data Types & Encoding

### Integer Encoding (RAW field)

**Format:** Big-endian (network byte order)

```
256-bit integer n ∈ [2^255, 2^256 - 1]

Example (Big-Endian):
  n = 0x0123456789ABCDEF...
  Bytes: 01 23 45 67 89 AB CD EF ... (32 bytes total)
```

**Constraints:**
- Most Significant Bit (msb) = 1 (ensures 256-bit length)
- Least Significant Bit (lsb) = 1 (ensures odd)
- For primes: IS_PRIME(n) verified with Miller-Rabin (25 rounds)
- For composites: Generated via specific algorithm (see ARCHITECTURE_RFC.md)

### Float Vector Encoding (FLOATS field)

**Format:** IEEE754 single-precision (float32), little-endian

```
For each bit i ∈ [0, 256):
  bit_value = (n >> i) & 1         // Extract bit from integer
  noise = BoxMuller(σ=0.05)        // Gaussian noise
  float_value = (float32)bit_value + noise
  
  Store as IEEE754 float32 (4 bytes, LE)
  at offset: 36 + (i × 4) in packet

Float Range: [-0.15, 1.15] (nominal + noise)
```

**Example:**
```
Bit 0 = 1:  f32 = 1.0 + (-0.03) = 0.97
Bit 1 = 0:  f32 = 0.0 + (+0.04) = 0.04
...
Bit 255 = 1: f32 = 1.0 + (-0.02) = 0.98

Stored as little-endian bytes:
Bit 0: 3F 78 AE 8C  (IEEE754 LE for 0.97)
Bit 1: 3D A3 D70A  (IEEE754 LE for 0.04)
...
```

**Validation (Python/Go):**
```python
for i in range(256):
    u32 = bytes_to_u32_le(floats[i*4:i*4+4])
    f32 = ieee754_from_bits(u32)
    
    # Reject invalid floats
    if is_nan(f32) or is_inf(f32):
        raise PacketCorrupt()
```

### Hash Encoding (H_RAW, H_VEC, H_ALL)

**Format:** BLAKE2s-256 digest (32 bytes)

```
BLAKE2s-256(data) → [byte0, byte1, ..., byte31]

Example (hexadecimal representation):
  h = [0xA1, 0xB2, 0xC3, ..., 0xFF]
  Hex String: "A1B2C3...FF"
  Raw Bytes: A1 B2 C3 ... FF (32 bytes)
```

**Hash Computation (Go):**

```go
import "golang.org/x/crypto/blake2s"

h := blake2s.Sum256(data)
// h is [32]byte

// Constant-time comparison
func ctEq32(a, b *[32]byte) bool {
    v := 0
    for i := 0; i < 32; i++ {
        v |= a[i] ^ b[i]
    }
    return v == 0
}
```

---

## Error Handling

### Packet Validation Errors

| Condition | Action | Code | Recovery |
|-----------|--------|------|----------|
| Size mismatch | Reject | CORRUPT | Drop packet |
| Magic mismatch | Reject | CORRUPT | Drop packet |
| Version != 1 | Reject | CORRUPT | Drop packet |
| H_RAW mismatch | Reject | CORRUPT | Drop packet |
| H_VEC mismatch | Reject | CORRUPT | Drop packet |
| H_ALL mismatch | Reject | CORRUPT | Drop packet |
| NaN/Inf in floats | Reject | CORRUPT | Drop packet |
| Type ∉ {0..5} | Reject | INVALID | Drop packet |

### Statistics Tracking

Go Bridge maintains atomic counters:

```go
type Stats struct {
    recv    atomic.Uint64  // Packets received from C++
    corrupt atomic.Uint64  // Validation failures
    sent    atomic.Uint64  // Packets sent to Python via gRPC
}
```

**Monitoring Query:**
```bash
curl http://localhost:8080/stats
# Response:
# {
#   "recv": 1000000,
#   "corrupt": 123,
#   "sent": 999877,
#   "per_type": { "0": 333000, "1": 333000, "2": 333000, ... }
# }
```

---

## Communication Flows

### Flow 1: Packet Generation & Transmission (C++ → Go)

```
C++ Core                           Go Bridge
═════════                          ═════════

1. Generate number
   └─ ChaCha20 RNG
   └─ Primality test (Miller-Rabin)
   
2. Encode packet
   └─ Binary vector (256 floats)
   └─ BLAKE2s hashing (h_raw, h_vec, h_all)
   └─ Trailer assembly
   
3. ZMQ PUSH
   └─ Send 1172 bytes
   └─ Rate: 100k-500k packets/sec
                                    │
                                    ↓
                                4. ZMQ PULL (with 1000ms timeout)
                                5. Parse packet structure
                                6. Constant-time hash verification
                                7. Type routing
                                   └─ chanPrime, chanHard, chanEasy
                                8. Update statistics
```

### Flow 2: Backpressure Control (Go → C++)

```
Go Bridge                          C++ Core
═════════                          ════════

Monitor buffer fill:
- If chanPrime/chanHard/chanEasy > 90% full
  └─ Send "STATE:0|0|0:92|45|32"
                                    │
                                    ↓
                                1. Receive STATE message
                                2. Parse: STATE:p|h|e:pbuf|hbuf|ebuf
                                3. Update worker flags:
                                   - g_run_p = (p == 1)
                                   - g_run_h = (h == 1)
                                   - g_run_e = (e == 1)
                                4. Pause type-specific workers
                                5. Queue empties
                                   
- If fill < 75%
  └─ Send "STATE:1|1|1:45|32|20"
                                    │
                                    ↓
                                6. Resume workers
                                7. Fill channels
```

### Flow 3: Data Streaming (Go → Python)

```
Go Bridge                          Python ML
═════════                          ══════════

gRPC StreamPrimes():
- Client connects to localhost:50051
- Server opens bidirectional stream
- For each packet in chanPrime:
  └─ Convert to protobuf Packet
  └─ Send via gRPC
                                    │
                                    ↓
                                1. Receive Packet (protobuf)
                                2. Deserialize:
                                   - type: int32
                                   - raw: bytes (32)
                                   - input: float32[256]
                                   - seq: uint64
                                3. Create DataPacket object
                                4. Enqueue to buffer
                                5. Batch when buffer reaches 64
                                6. Forward to VAE model
```

---

## Backpressure State Machine

```
    ┌─────────────┐
    │   CREATED   │
    └──────┬──────┘
           │ start()
           ↓
    ┌─────────────────────┐
    │    MONITORING       │  Fill% monitored every 100ms
    │  (all channels open)│
    └──────┬──────────────┘
           │
     +─────┴─────+
     │           │
     │ Fill>90%  │
     │ for type  │ (independently per type)
     ↓           ↓
  ┌─────────┐  ┌──────────────┐
  │ SENDING │  │   CRITICAL   │
  │ STATE   │→ │    (PAUSE)   │
  └─────────┘  └──────┬───────┘
                      │
                      │ Fill<75% for type
                      │ (with hysteresis)
                      ↓
                ┌──────────────┐
                │   SENDING    │
                │   RESUME     │
                └──────┬───────┘
                       │
                       ↓ (loop back)
                   MONITORING
```

### State Transitions

```
CREATED
  └─ start() → MONITORING

MONITORING
  ├─ if channel[i] fill > 90% → Send CRITICAL for channel[i]
  └─ else → Continue monitoring

CRITICAL (for channel[i])
  ├─ if channel[i] fill < 75% → Send OPEN for channel[i]
  └─ else → Stay CRITICAL

Repeat indefinitely until shutdown.
```

---

## Security Considerations

### Integrity Protection

**Triple Hashing Strategy:**

1. **h_raw** protects integer representation
   - Detects tampering with number value
   
2. **h_vec** protects float encoding
   - Detects bit flips in noisy representation
   
3. **h_all** protects metadata & structure
   - Detects packet reassembly attacks
   - Detects sequence reordering

Any single bit change causes all three to fail (with probability 1 - 2^(-256)).

### Timing Attack Resistance

**Constant-Time Comparison:**

```go
func ctEq32(a, b *[32]byte) bool {
    var v byte = 0
    for i := 0; i < 32; i++ {
        v |= a[i] ^ b[i]
    }
    return v == 0
}
```

**Property:** Always executes 32 XOR operations, regardless of match position.
- Prevents attackers from detecting which bytes differ
- Mitigates byte-by-byte early-exit attacks

### Threats Not Mitigated

1. **Replay Attacks** (future mitigation: sequence tracking)
2. **Man-in-the-Middle** (future mitigation: TLS/mTLS)
3. **Type Confusion** (partially mitigated: type routing)
4. **Probabilistic Primality Failure** (acceptable: < 2^(-50))

---

## References

- [ARCHITECTURE_RFC.md](./ARCHITECTURE_RFC.md) - Detailed RFC 7539-based specification
- [README.md](./README.md) - Quick start guide
- RFC 7539: ChaCha20 and Poly1305
- RFC 4648: Data Encodings
- NIST FIPS 186-4: Digital Signature Standard

---

**Document End**

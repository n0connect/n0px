package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"net"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/fatih/color"
	zmq "github.com/pebbe/zmq4"
	"golang.org/x/crypto/blake2s"
	"google.golang.org/grpc"

	pb "prime.bridge/pb"
)

const (
	// Trailer layout
	trMagic       = "PXSV"
	trailerSize   = 112
	hashSize      = 32
	supportVecLen = 96
)

// ---- Config ----

type Config struct {
	System struct {
		PrimeBits    int `json:"prime_bits"`
		RawSizeBytes int `json:"raw_size_bytes"`
	} `json:"system"`
	Network struct {
		ZMQHost           string `json:"zmq_host"`
		ZMQPortCorePush   int    `json:"zmq_port_core_push"`
		ZMQPortCorePub    int    `json:"zmq_port_core_pub"`
		GRPCPort          int    `json:"grpc_port"`
		BufferSizePackets int    `json:"buffer_size_packets"`
	} `json:"network"`
	Buffer struct {
		ChannelSize int `json:"go_channel_buffer_size"`
	} `json:"buffer"`
}

func loadConfig(path string) (Config, error) {
	var cfg Config
	b, err := os.ReadFile(path)
	if err != nil {
		return cfg, err
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return cfg, err
	}
	if cfg.System.PrimeBits <= 0 || cfg.System.RawSizeBytes <= 0 {
		return cfg, errors.New("invalid system.prime_bits or system.raw_size_bytes")
	}
	if cfg.Network.GRPCPort == 0 {
		cfg.Network.GRPCPort = 50051
	}
	if cfg.Buffer.ChannelSize == 0 {
		cfg.Buffer.ChannelSize = 2000
	}
	if cfg.Network.ZMQPortCorePush == 0 {
		cfg.Network.ZMQPortCorePush = 5556
	}
	if cfg.Network.ZMQPortCorePub == 0 {
		cfg.Network.ZMQPortCorePub = 5557
	}
	if cfg.Network.ZMQHost == "" {
		cfg.Network.ZMQHost = "127.0.0.1"
	}
	return cfg, nil
}

func packetSize(cfg Config) int {
	base := 4 + cfg.System.RawSizeBytes + (cfg.System.PrimeBits * 4)
	return base + trailerSize
}

// ---- Data types ----

type DataPacket struct {
	Raw   []byte
	Input []float32
	Type  int32
	Seq   uint64
}

type Stats struct {
	recv    atomic.Uint64
	corrupt atomic.Uint64
	sent    atomic.Uint64
}

type BufferState struct {
	p atomic.Uint32 // 1=open, 0=full
	h atomic.Uint32
	e atomic.Uint32
}

func (bs *BufferState) set(p, h, e bool) {
	if p {
		bs.p.Store(1)
	} else {
		bs.p.Store(0)
	}
	if h {
		bs.h.Store(1)
	} else {
		bs.h.Store(0)
	}
	if e {
		bs.e.Store(1)
	} else {
		bs.e.Store(0)
	}
}

func (bs *BufferState) canRoute(t int32) bool {
	switch t {
	case 0, 4:
		return bs.e.Load() == 1
	case 1, 3:
		return bs.p.Load() == 1
	case 2, 5:
		return bs.h.Load() == 1
	default:
		return false
	}
}

// ---- Integrity ----

func blake256(b []byte) [32]byte {
	return blake2s.Sum256(b)
}

func ctEq32(a *[32]byte, b *[32]byte) bool {
	// constant-ish time compare (manual)
	var v byte = 0
	for i := 0; i < 32; i++ {
		v |= a[i] ^ b[i]
	}
	return v == 0
}

func verifyAndParse(cfg Config, data []byte, st *Stats) (DataPacket, bool) {
	want := packetSize(cfg)
	if len(data) != want {
		st.corrupt.Add(1)
		return DataPacket{}, false
	}

	rawSize := cfg.System.RawSizeBytes
	bits := cfg.System.PrimeBits
	baseSize := 4 + rawSize + (bits * 4)

	// trailer offsets
	trOff := baseSize
	if string(data[trOff:trOff+4]) != trMagic {
		st.corrupt.Add(1)
		return DataPacket{}, false
	}
	ver := data[trOff+4]
	if ver != 1 {
		st.corrupt.Add(1)
		return DataPacket{}, false
	}
	seq := binary.LittleEndian.Uint64(data[trOff+8 : trOff+16])

	// sections
	typeCode := int32(binary.LittleEndian.Uint32(data[0:4]))
	rawBytes := data[4 : 4+rawSize]
	floatBytes := data[4+rawSize : baseSize]

	// support vector
	sv := data[trOff+16 : trOff+16+supportVecLen]
	var gotRaw, gotVec, gotAll [32]byte
	copy(gotRaw[:], sv[0:32])
	copy(gotVec[:], sv[32:64])
	copy(gotAll[:], sv[64:96])

	expRaw := blake256(rawBytes)
	expVec := blake256(floatBytes)

	// all_hash = blake2s(type||raw||vec||seq||raw_size||bits)
	h, _ := blake2s.New256(nil)
	_, _ = h.Write(data[0:4])
	_, _ = h.Write(rawBytes)
	_, _ = h.Write(floatBytes)

	var seqLE [8]byte
	binary.LittleEndian.PutUint64(seqLE[:], seq)
	_, _ = h.Write(seqLE[:])

	var meta [8]byte
	binary.LittleEndian.PutUint32(meta[0:4], uint32(rawSize))
	binary.LittleEndian.PutUint32(meta[4:8], uint32(bits))
	_, _ = h.Write(meta[:])

	sum := h.Sum(nil)
	var expAll [32]byte
	copy(expAll[:], sum)

	if !ctEq32(&gotRaw, &expRaw) || !ctEq32(&gotVec, &expVec) || !ctEq32(&gotAll, &expAll) {
		st.corrupt.Add(1)
		return DataPacket{}, false
	}

	// floats decode (safe)
	input := make([]float32, bits)
	for i := 0; i < bits; i++ {
		u := binary.LittleEndian.Uint32(floatBytes[i*4 : (i+1)*4])
		f := math.Float32frombits(u)
		// basic NaN/Inf guard (keep it strict; corrupt say)
		if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
			st.corrupt.Add(1)
			return DataPacket{}, false
		}
		input[i] = f
	}

	safeRaw := make([]byte, len(rawBytes))
	copy(safeRaw, rawBytes)

	st.recv.Add(1)

	return DataPacket{
		Raw:   safeRaw,
		Input: input,
		Type:  typeCode,
		Seq:   seq,
	}, true
}

// ---- Bridge ----

type Bridge struct {
	cfg   Config
	stats Stats
	state BufferState

	chanP chan DataPacket
	chanH chan DataPacket
	chanE chan DataPacket

	cmdSock *zmq.Socket
}

func NewBridge(cfg Config) *Bridge {
	b := &Bridge{
		cfg:   cfg,
		chanP: make(chan DataPacket, cfg.Buffer.ChannelSize),
		chanH: make(chan DataPacket, cfg.Buffer.ChannelSize),
		chanE: make(chan DataPacket, cfg.Buffer.ChannelSize),
	}
	b.state.set(true, true, true)
	return b
}

func (b *Bridge) zmqIngestor(ctx context.Context, wg *sync.WaitGroup) {
	defer wg.Done()

	pull, err := zmq.NewSocket(zmq.PULL)
	if err != nil {
		log.Fatalf("ZMQ socket error: %v", err)
	}
	defer pull.Close()

	addr := fmt.Sprintf("tcp://%s:%d", b.cfg.Network.ZMQHost, b.cfg.Network.ZMQPortCorePush)
	if err := pull.Connect(addr); err != nil {
		log.Fatalf("ZMQ connect error: %v", err)
	}
	_ = pull.SetRcvtimeo(250 * time.Millisecond)

	green := color.New(color.FgGreen, color.Bold)
	green.Printf("✓ ZMQ PULL connected to %s\n", addr)

	pktCount := uint64(0)
	startTime := time.Now()

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		msg, err := pull.RecvBytes(0)
		if err != nil {
			// timeout expected
			continue
		}

		pkt, ok := verifyAndParse(b.cfg, msg, &b.stats)
		if !ok {
			continue
		}

		pktCount++

		// Show progress every 500 packets
		if pktCount%500 == 0 {
			elapsed := time.Since(startTime).Seconds()
			rate := float64(pktCount) / elapsed

			yellow := color.New(color.FgYellow)
			yellow.Printf("\r[ZMQ] Processed: %6d packets | Rate: %.1f pkt/s | Verified: %d | Dropped: %d",
				pktCount, rate, b.stats.recv.Load(), b.stats.corrupt.Load())
			os.Stdout.Sync() // Flush output to ensure carriage return works in tmux
		}

		// buffer pressure gate (drop if target full)
		if !b.state.canRoute(pkt.Type) {
			b.stats.corrupt.Add(1)
			continue
		}

		switch pkt.Type {
		case 0, 4:
			select {
			case b.chanE <- pkt:
			default:
				b.stats.corrupt.Add(1)
			}
		case 1, 3:
			select {
			case b.chanP <- pkt:
			default:
				b.stats.corrupt.Add(1)
			}
		case 2, 5:
			select {
			case b.chanH <- pkt:
			default:
				b.stats.corrupt.Add(1)
			}
		default:
			b.stats.corrupt.Add(1)
		}
	}
}

func (b *Bridge) bufferMonitor(ctx context.Context, wg *sync.WaitGroup) {
	defer wg.Done()

	t := time.NewTicker(2 * time.Millisecond)
	defer t.Stop()

	crit := int(float64(cap(b.chanP)) * 0.90)
	rec := int(float64(cap(b.chanP)) * 0.75)

	stateP, stateH, stateE := true, true, true
	var lastUI time.Time

	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			pLen, hLen, eLen := len(b.chanP), len(b.chanH), len(b.chanE)

			if stateP && pLen >= crit {
				stateP = false
			} else if !stateP && pLen <= rec {
				stateP = true
			}
			if stateH && hLen >= crit {
				stateH = false
			} else if !stateH && hLen <= rec {
				stateH = true
			}
			if stateE && eLen >= crit {
				stateE = false
			} else if !stateE && eLen <= rec {
				stateE = true
			}

			b.state.set(stateP, stateH, stateE)

			// C++ core'a state publish (PUB)
			if b.cmdSock != nil {
				pPct := (pLen * 100) / cap(b.chanP)
				hPct := (hLen * 100) / cap(b.chanH)
				ePct := (eLen * 100) / cap(b.chanE)
				cmd := fmt.Sprintf("STATE:%d|%d|%d:%d|%d|%d", btoi(stateP), btoi(stateH), btoi(stateE), pPct, hPct, ePct)
				_, _ = b.cmdSock.Send(cmd, zmq.DONTWAIT)
			}

			// UI throttle
			if time.Since(lastUI) > 100*time.Millisecond {
				lastUI = time.Now()
				b.renderUI(pLen, hLen, eLen, stateP, stateH, stateE)
			}
		}
	}
}

func btoi(v bool) int {
	if v {
		return 1
	}
	return 0
}

func (b *Bridge) renderUI(p, h, e int, sp, sh, se bool) {
	// Clear screen
	fmt.Print("\033[H\033[2J")

	// Title banner
	cyan := color.New(color.FgCyan, color.Bold)
	green := color.New(color.FgGreen, color.Bold)
	red := color.New(color.FgRed, color.Bold)
	yellow := color.New(color.FgYellow, color.Bold)
	magenta := color.New(color.FgMagenta, color.Bold)
	white := color.New(color.FgWhite, color.Bold)

	cyan.Println("╔════════════════════════════════════════════════════════════════╗")
	cyan.Println("║   PRIME-X BRIDGE - Integrity Trailer v1 (BLAKE2s Verified)    ║")
	cyan.Println("╚════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Stats header
	recv := b.stats.recv.Load()
	sent := b.stats.sent.Load()
	corrupt := b.stats.corrupt.Load()

	fmt.Printf("  ")
	green.Printf("✓ RCV: %8d  ", recv)
	green.Printf("✓ SENT: %8d  ", sent)
	red.Printf("✗ DROP: %8d\n", corrupt)
	fmt.Println()

	// Buffer state bars
	fmt.Println(cyan.Sprint("  ├─ BUFFER STATE ─────────────────────────────────────────"))

	// Prime buffer
	pPct := (p * 100) / cap(b.chanP)
	pBar := b.renderBar(p, cap(b.chanP), pPct, sp)
	stateStr := "OPEN "
	if !sp {
		stateStr = color.RedString("FULL ")
	}
	fmt.Printf("  │  P (Prime)       [%s] %s %3d%% (%d/%d)\n",
		pBar, stateStr, pPct, p, cap(b.chanP))

	// Hard composite buffer
	hPct := (h * 100) / cap(b.chanH)
	hBar := b.renderBar(h, cap(b.chanH), hPct, sh)
	stateStr = "OPEN "
	if !sh {
		stateStr = color.RedString("FULL ")
	}
	fmt.Printf("  │  H (Hard Comp.)  [%s] %s %3d%% (%d/%d)\n",
		hBar, stateStr, pPct, h, cap(b.chanH))

	// Easy composite buffer
	ePct := (e * 100) / cap(b.chanE)
	eBar := b.renderBar(e, cap(b.chanE), ePct, se)
	stateStr = "OPEN "
	if !se {
		stateStr = color.RedString("FULL ")
	}
	fmt.Printf("  │  E (Easy Comp.)  [%s] %s %3d%% (%d/%d)\n",
		eBar, stateStr, ePct, e, cap(b.chanE))

	fmt.Println(cyan.Sprint("  └─────────────────────────────────────────────────────────"))
	fmt.Println()

	// Network config
	fmt.Println(cyan.Sprint("  ├─ NETWORK CONFIG ────────────────────────────────────────"))
	fmt.Printf("  │  gRPC Server   : %s\n", yellow.Sprint(fmt.Sprintf(":%d", b.cfg.Network.GRPCPort)))
	fmt.Printf("  │  ZMQ PULL      : %s\n", yellow.Sprint(fmt.Sprintf("%s:%d", b.cfg.Network.ZMQHost, b.cfg.Network.ZMQPortCorePush)))
	fmt.Printf("  │  ZMQ PUB       : %s\n", yellow.Sprint(fmt.Sprintf("*:%d", b.cfg.Network.ZMQPortCorePub)))
	fmt.Printf("  │  System Config : BITS=%d RAW=%d PACKET=%d\n",
		magenta.Sprint(b.cfg.System.PrimeBits),
		magenta.Sprint(b.cfg.System.RawSizeBytes),
		magenta.Sprint(packetSize(b.cfg)))
	fmt.Println(cyan.Sprint("  └─────────────────────────────────────────────────────────"))
	fmt.Println()

	// Trailer protocol info
	fmt.Println(cyan.Sprint("  ┌─ INTEGRITY TRAILER v1 ──────────────────────────────────"))
	fmt.Println(white.Sprint("  │ Protocol: [type:4] [raw:raw_sz] [floats:bits*4] [trailer:112]"))
	fmt.Println(white.Sprint("  │ Trailer:  [magic:PXSV] [version:1] [seq:8] [hashes:96]"))
	fmt.Println(white.Sprint("  │ Hashes:   blake2s-256 (raw | vec | all) + constant-time verify"))
	fmt.Println(cyan.Sprint("  └────────────────────────────────────────────────────────────"))
	fmt.Println()

	fmt.Print(cyan.Sprint("  ★ READY FOR DATA INGESTION"))
}

func (b *Bridge) renderBar(used, total, pct int, open bool) string {
	barLen := 20
	filled := (pct * barLen) / 100

	bar := ""
	for i := 0; i < barLen; i++ {
		if i < filled {
			if pct >= 90 {
				bar += color.RedString("█")
			} else if pct >= 70 {
				bar += color.YellowString("█")
			} else {
				bar += color.GreenString("█")
			}
		} else {
			bar += "░"
		}
	}
	return bar
}

// ---- gRPC ----

type grpcServer struct {
	pb.UnimplementedDataProviderServer
	b *Bridge
}

func (s *grpcServer) StreamData(req *pb.StreamConfig, srv pb.DataProvider_StreamDataServer) error {
	for {
		if srv.Context().Err() != nil {
			return nil
		}

		var pkt DataPacket
		got := false

		// simple round-robin attempts
		for i := 0; i < 3; i++ {
			select {
			case pkt = <-s.b.chanP:
				got = true
			case pkt = <-s.b.chanH:
				got = true
			case pkt = <-s.b.chanE:
				got = true
			default:
			}
			if got {
				break
			}
		}

		if !got {
			time.Sleep(1 * time.Millisecond)
			continue
		}

		out := &pb.DataBatch{
			RawBytes:    pkt.Raw,
			InputVector: pkt.Input,
			Label:       pkt.Type,
		}

		if err := srv.Send(out); err != nil {
			return err
		}
		s.b.stats.sent.Add(1)
	}
}

// ---- main ----

func main() {
	// Config path: bridge runs from bridge/ dir, config is in parent dir
	configPath := "../config/config.json"
	cfg, err := loadConfig(configPath)
	if err != nil {
		log.Fatalf("config load failed: %v", err)
	}

	cyan := color.New(color.FgCyan, color.Bold)
	green := color.New(color.FgGreen, color.Bold)
	yellow := color.New(color.FgYellow, color.Bold)

	cyan.Println("\n╔════════════════════════════════════════════════════════════╗")
	cyan.Println("║  PRIME-X BRIDGE v5.3 - Starting Initialization...        ║")
	cyan.Println("╚════════════════════════════════════════════════════════════╝\n")

	green.Printf("✓ Config loaded: BITS=%d RAW=%d PACKET=%d TRAILER=112\n",
		cfg.System.PrimeBits, cfg.System.RawSizeBytes, packetSize(cfg))

	bridge := NewBridge(cfg)

	// PUB socket (state -> C++)
	cmdSock, err := zmq.NewSocket(zmq.PUB)
	if err != nil {
		log.Fatalf("PUB socket create failed: %v", err)
	}
	defer cmdSock.Close()

	pubAddr := fmt.Sprintf("tcp://*:%d", cfg.Network.ZMQPortCorePub)
	if err := cmdSock.Bind(pubAddr); err != nil {
		log.Fatalf("PUB bind failed: %v", err)
	}
	bridge.cmdSock = cmdSock
	green.Printf("✓ PUB bound: %s\n", pubAddr)

	ctx, cancel := context.WithCancel(context.Background())
	var wg sync.WaitGroup

	// shutdown
	sigC := make(chan os.Signal, 2)
	signal.Notify(sigC, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigC
		cancel()
	}()

	wg.Add(2)
	go bridge.zmqIngestor(ctx, &wg)
	go bridge.bufferMonitor(ctx, &wg)

	// gRPC serve
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", cfg.Network.GRPCPort))
	if err != nil {
		log.Fatalf("listen failed: %v", err)
	}

	gs := grpc.NewServer()
	pb.RegisterDataProviderServer(gs, &grpcServer{b: bridge})

	go func() {
		<-ctx.Done()
		gs.GracefulStop()
		_ = lis.Close()
	}()

	green.Printf("✓ gRPC listening: %s\n\n", lis.Addr().String())
	yellow.Println("★ Bridge ready. Waiting for C++ core connection on ZMQ PULL...\n")

	if err := gs.Serve(lis); err != nil && !errors.Is(err, net.ErrClosed) {
		log.Fatalf("grpc serve failed: %v", err)
	}

	wg.Wait()

	cyan.Println("\n╔════════════════════════════════════════════════════════════╗")
	cyan.Println("║  BRIDGE SHUTDOWN                                          ║")
	cyan.Println("╚════════════════════════════════════════════════════════════╝\n")

	green.Printf("✓ Statistics:\n")
	green.Printf("  Received Packets: %d\n", bridge.stats.recv.Load())
	green.Printf("  Sent to gRPC:     %d\n", bridge.stats.sent.Load())
	yellow.Printf("  Dropped/Corrupt:  %d\n\n", bridge.stats.corrupt.Load())
}

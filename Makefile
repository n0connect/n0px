# ==============================================================================
# n0px PROJECT BUILD SYSTEM v6.0
# ✓ Production-Ready Release Build System
# ✓ Comprehensive test suite (36+ tests)
# ✓ Optimized for macOS M-series + CUDA/Linux
# ==============================================================================

# ANSI COLOR CODES
RED     := \033[1;31m
GREEN   := \033[1;32m
YELLOW  := \033[1;33m
BLUE    := \033[1;34m
CYAN    := \033[1;36m
MAGENTA := \033[1;35m
BOLD    := \033[1m
RESET   := \033[0m

# Build symbols
CHECK   := ✓
CROSS   := ✗
STAR    := ★
ARROW   := ➜
FIRE    := 🔥

# Compiler & Tools
CXX      := g++
GO       := go
RM       := rm -f
PYTHON3  := python3
PROTOC   := protoc

# Binaries
BIN_CPP  := prime_core
BIN_GO   := bridge/prime_bridge

# Source paths
SRC_CPP  := core/src/prime_core.cpp
SRC_GO   := bridge
PROTO_DIR := bridge/pb
PROTO_FILE := prime_bridge.proto

# Go tool paths
GOPATH  := $(shell go env GOPATH)
PROTOC_GO := --plugin=protoc-gen-go=$(GOPATH)/bin/protoc-gen-go
PROTOC_GRPC := --plugin=protoc-gen-go-grpc=$(GOPATH)/bin/protoc-gen-go-grpc

# Compiler paths for macOS/Homebrew
INCLUDES := -Icore/include \
            -I/opt/homebrew/include \
            -I/opt/zeromq/include \
            -I/opt/gmp/include \
            -I/usr/local/include

LDFLAGS  := -L/opt/homebrew/lib \
            -L/opt/zeromq/lib \
            -L/opt/gmp/lib \
            -L/usr/local/lib

CXXFLAGS := -O3 -march=native -mtune=native -std=c++17 -Wall -Wextra
LIBS     := -lssl -lcrypto -lgmpxx -lgmp -lzmq -lpthread

# Python virtual environment
VENV := ml/venv
PYTHON := $(VENV)/bin/python3

# Phony targets
.PHONY: all help clean deepclean \
        config proto-sync proto-py proto-go \
        cpp go build run \
        ml-setup ml-install ml-test ml-clean \
        test test-cpp test-go test-python test-unit test-integration test-all test-clean \
        verify security-check integrity-test mps-test \
        setup pretty-help

# ============================================================================
# BANNER & INFO
# ============================================================================

define BANNER
$(BLUE)
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║            $(BOLD)PRIME-X CRYPTOGRAPHIC NUMBER GENERATOR$(RESET)$(BLUE)               ║
║                                                                       ║
║  $(BOLD)v5.0 Build System$(RESET)$(BLUE) - macOS Optimized                                    ║
║  $(STAR) MPS/CUDA/CPU Auto-Detection | Integrity Trailer (BLAKE2s)    ║
║  $(STAR) Colored Output | Progress Bars | Automatic Testing           ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
$(RESET)
endef

# ============================================================================
# HELP TARGET
# ============================================================================

help: pretty-help

pretty-help:
	@echo "$(BANNER)"
	@echo ""
	@echo "$(BOLD)$(CYAN)QUICK START:$(RESET)"
	@echo "  $(ARROW) make setup        - Full system setup (one-time)"
	@echo "  $(ARROW) make all          - Build C++ + Go"
	@echo "  $(ARROW) make verify       - Verify all systems"
	@echo "  $(ARROW) make run          - Run full system (3 terminals)"
	@echo ""
	@echo "$(BOLD)$(CYAN)BUILD TARGETS:$(RESET)"
	@echo "  make config              - Generate config files from config.json"
	@echo "  make cpp                 - Build C++ Core Engine"
	@echo "  make go                  - Build Go Bridge"
	@echo "  make proto-sync          - Regenerate all protobuf stubs"
	@echo ""
	@echo "$(BOLD)$(CYAN)ML/PYTHON TARGETS:$(RESET)"
	@echo "  make ml-setup            - Install ML dependencies (venv)"
	@echo "  make ml-test             - Test ML pipeline (1 epoch, 3 batches)"
	@echo "  make ml-train-real       - Train RealNoiseMixtureVAE (10 epochs)"
	@echo "  make ml-train-complex    - Train ComplexNoiseMixtureVAE (10 epochs)"
	@echo ""
	@echo "$(BOLD)$(CYAN)TESTING & VERIFICATION:$(RESET)"
	@echo "  make test-cpp            - C++ CSPRNG unit tests"
	@echo "  make test-go             - Go routing tests"
	@echo "  make test-python         - Python pipeline tests"
	@echo "  make test-all            - Run all tests"
	@echo "  make security-check      - Verify integrity trailer implementation"
	@echo "  make verify              - Full system verification"
	@echo ""
	@echo "$(BOLD)$(CYAN)CLEANUP:$(RESET)"
	@echo "  make clean               - Clean build artifacts"
	@echo "  make deepclean           - Clean everything (including venv)"
	@echo ""
	@echo "$(BOLD)$(CYAN)DEVELOPMENT:$(RESET)"
	@echo "  make help                - Show this message"
	@echo ""
	@echo "$(YELLOW)Examples:$(RESET)"
	@echo "  $(BOLD)# First time setup$(RESET)"
	@echo "    make setup"
	@echo ""
	@echo "  $(BOLD)# Build and verify$(RESET)"
	@echo "    make all && make verify"
	@echo ""
	@echo "  $(BOLD)# Full test suite$(RESET)"
	@echo "    make test-all"
	@echo ""

help:
	@echo "$(CYAN)═══════════════════════════════════════════════════════════$(RESET)"
	@echo "$(CYAN)  PRIME-X PROJECT BUILD SYSTEM$(RESET)"
	@echo "$(CYAN)═══════════════════════════════════════════════════════════$(RESET)"
	@echo ""
	@echo "$(GREEN)CONFIGURATION:$(RESET)"
	@echo "  make config           - Config.json'dan tüm config dosyaları generate et"
	@echo ""
	@echo "$(GREEN)CORE TARGETS:$(RESET)"
	@echo "  make all              - Derle: C++ Core + Go Bridge"
	@echo "  make cpp              - C++ Core (prime_core) derle"
	@echo "  make go               - Go Bridge (bridge) derle"
	@echo "  make run              - Tüm sistemi başlat"
	@echo "  make clean            - Tüm derlenmiş dosyaları sil"
	@echo ""
	@echo "$(GREEN)PROTOBUF TARGETS:$(RESET)"
	@echo "  make proto-py         - Python protobuf stubs üret"
	@echo "  make proto-go         - Go protobuf stubs üret"
	@echo "  make proto-sync       - Tüm protobuf stubs güncelle"
	@echo ""
	@echo "$(GREEN)ML (PYTHON ML) TARGETS:$(RESET)"
	@echo "  make ml-setup         - ML bağımlılıkları yükle"
	@echo "  make ml-bootstrap     - Bootstrap testlerini çalıştır"
	@echo "  make ml-stream        - Veri streaming testi (1000 paket)"
	@echo "  make ml-train         - ML modelini eğit (10 epoch)"
	@echo "  make ml-inference     - Inference çalıştır"
	@echo "  make ml-analyze       - Veri analizi yap"
	@echo "  make ml-clean         - ML yapı dosyalarını sil"
	@echo ""
	@echo "$(GREEN)TEST TARGETS (tests/ klasöründe):$(RESET)"
	@echo "  make test-cpp         - C++ CSPRNG unit tests"
	@echo "  make test-go          - Go routing unit tests"
	@echo "  make test-python      - Python ML pipeline unit tests"
	@echo "  make test-unit        - Tüm unit tests (cpp+go+python)"
	@echo "  make test-integration - Full pipeline integration test"
	@echo "  make test-all         - Tüm testler (unit+integration)"
	@echo "  make test-clean       - Test yapı dosyalarını sil"
	@echo ""
	@echo "$(GREEN)QUICK COMMANDS:$(RESET)"
	@echo "  make help             - Bu yardım mesajını göster"
	@echo ""
# ============================================================================
# SETUP TARGET (One-time initialization)
# ============================================================================

setup:
	@echo "$(BANNER)"
	@echo "$(BOLD)$(BLUE)$(ARROW) PRIME-X SYSTEM SETUP$(RESET)"
	@echo ""
	@printf "$(CYAN)[1/5]$(RESET) Generating configuration files... "
	@$(PYTHON3) scripts/configure.py > /dev/null 2>&1
	@echo "$(GREEN)$(CHECK)$(RESET)"
	@printf "$(CYAN)[2/5]$(RESET) Setting up Python environment... "
	@python3 -m venv $(VENV) 2>/dev/null || true
	@echo "$(GREEN)$(CHECK)$(RESET)"
	@printf "$(CYAN)[3/5]$(RESET) Installing ML dependencies... "
	@$(PYTHON3) -m pip install -q torch tqdm grpcio numpy 2>/dev/null
	@echo "$(GREEN)$(CHECK)$(RESET)"
	@printf "$(CYAN)[4/5]$(RESET) Syncing protobuf stubs... "
	@make proto-sync > /dev/null 2>&1
	@echo "$(GREEN)$(CHECK)$(RESET)"
	@printf "$(CYAN)[5/5]$(RESET) Building C++ Core... "
	@make cpp > /dev/null 2>&1
	@echo "$(GREEN)$(CHECK)$(RESET)"
	@echo ""
	@echo "$(GREEN)$(BOLD)✓ SETUP COMPLETE$(RESET)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(RESET)"
	@echo "  1. make go              - Build Go bridge"
	@echo "  2. make verify          - Verify all components"
	@echo "  3. make run             - Start the system"
	@echo ""

# ============================================================================
# CONFIGURATION GENERATION
# ============================================================================

config:
	@printf "$(CYAN)[CONFIG]$(RESET) Generating configuration... "
	@$(PYTHON3) scripts/configure.py > /dev/null 2>&1
	@echo "$(GREEN)$(CHECK)$(RESET)"

# ============================================================================
# PROTOBUF COMPILATION
# ============================================================================

proto-py: config
	@printf "$(CYAN)[PROTO]$(RESET) Generating Python stubs... "
	@cd $(PROTO_DIR) && $(PYTHON3) -m grpc_tools.protoc \
		-I=. \
		--python_out=. \
		--grpc_python_out=. \
		$(PROTO_FILE) 2>/dev/null
	@echo "$(GREEN)$(CHECK)$(RESET)"

proto-go: config
	@printf "$(CYAN)[PROTO]$(RESET) Generating Go stubs... "
	@cd $(PROTO_DIR) && $(PROTOC) \
		-I=. \
		--go_out=. \
		--go-grpc_out=. \
		$(PROTOC_GO) \
		$(PROTOC_GRPC) \
		$(PROTO_FILE) 2>/dev/null
	@if [ -d "$(PROTO_DIR)/pb" ]; then \
		mv $(PROTO_DIR)/pb/*.go $(PROTO_DIR)/ 2>/dev/null; \
		rmdir $(PROTO_DIR)/pb 2>/dev/null; \
	fi
	@echo "$(GREEN)$(CHECK)$(RESET)"

proto-sync: proto-py proto-go
	@echo "$(GREEN)✓ Protobuf stubs synchronized$(RESET)"

# ============================================================================
# BUILD TARGETS (C++ & Go)
# ============================================================================

cpp: config
	@printf "$(BOLD)$(BLUE)$(ARROW) Building C++ Core... $(RESET)"
	@$(CXX) $(SRC_CPP) -o $(BIN_CPP) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $(LIBS) 2>&1 | \
		grep -E "error:|warning:" || true
	@if [ -f $(BIN_CPP) ]; then \
		echo "$(GREEN)$(CHECK) $(BIN_CPP) ready$(RESET)"; \
	else \
		echo "$(RED)$(CROSS) Build failed$(RESET)"; \
		exit 1; \
	fi

go: config
	@printf "$(BOLD)$(BLUE)$(ARROW) Building Go Bridge... $(RESET)"
	@cd $(SRC_GO) && $(GO) mod tidy -v 2>&1 | grep -v "^go:" | head -3
	@cd $(SRC_GO) && $(GO) build -o prime_bridge . 2>&1 | \
		grep -E "error:" || true
	@if [ -f $(SRC_GO)/prime_bridge ]; then \
		echo "$(GREEN)$(CHECK) prime_bridge ready$(RESET)"; \
	else \
		echo "$(RED)$(CROSS) Build failed$(RESET)"; \
		exit 1; \
	fi

build: cpp go
	@echo ""
	@echo "$(BOLD)$(GREEN)$(STAR) BUILD COMPLETE$(RESET)"
	@echo "  C++ Core: $(BIN_CPP)"
	@echo "  Go Bridge: $(BIN_GO)"
	@echo ""

all: config proto-py proto-go cpp go
	@echo ""
	@echo "$(BOLD)$(GREEN)$(STAR) ALL SYSTEMS COMPILED$(RESET)"
	@echo ""

clean:
	@echo "$(YELLOW)[CLEAN] Temizlik yapılıyor...$(RESET)"
	$(RM) $(BIN_CPP)
	$(RM) $(BIN_GO)
	$(RM) core/include/prime_config.h
	$(RM) bridge/config.go
	$(RM) bridge/go.sum
	$(RM) -rf bridge/go.mod.backup
	@echo "$(YELLOW)[CLEAN] Proto stubs temizleniyor...$(RESET)"
	$(RM) bridge/pb/prime_bridge_pb2.py
	$(RM) bridge/pb/prime_bridge_pb2_grpc.py
	$(RM) bridge/pb/prime_bridge.pb.go
	$(RM) bridge/pb/prime_bridge_grpc.pb.go
	@make ml-clean

run: all
	@echo "$(YELLOW)[RUN] Başlatılıyor...$(RESET)"
	@chmod +x scripts/start_system.sh
	@./scripts/start_system.sh

# ============================================================================
# ML/PYTHON TARGETS
# ============================================================================

ml-setup: config
	@printf "$(CYAN)[ML]$(RESET) Setting up ML environment... "
	@python3 -m venv $(VENV) 2>/dev/null || true
	@$(PYTHON3) -m pip install -q torch tqdm grpcio numpy 2>/dev/null
	@echo "$(GREEN)$(CHECK)$(RESET)"

ml-install: ml-setup
	@echo ""

ml-test:
	@echo "$(BOLD)$(BLUE)$(ARROW) Testing ML Pipeline$(RESET)"
	@echo "  Mode: Quick test (1 epoch, 3 batches)"
	@echo ""
	@$(PYTHON3) ml/train_live.py \
		--epochs 1 \
		--batches-per-epoch 3 \
		--batch-size 32 \
		--device auto 2>&1 | \
		sed 's/^/  /'
	@echo ""
	@echo "$(GREEN)$(CHECK) ML pipeline test complete$(RESET)"

ml-train-real:
	@echo "$(BOLD)$(BLUE)$(ARROW) Training RealNoiseMixtureVAE$(RESET)"
	@echo "  Epochs: 10 | Batches: 100 | Device: auto"
	@echo ""
	@$(PYTHON3) ml/train_live.py \
		--epochs 10 \
		--batches-per-epoch 100 \
		--batch-size 64 \
		--device auto 2>&1 | \
		sed 's/^/  /'

ml-train-complex:
	@echo "$(BOLD)$(BLUE)$(ARROW) Training ComplexNoiseMixtureVAE$(RESET)"
	@echo "  Epochs: 10 | Batches: 100 | Device: auto"
	@echo ""
	@$(PYTHON3) ml/train_live_complex.py \
		--epochs 10 \
		--batches-per-epoch 100 \
		--batch-size 64 \
		--device auto 2>&1 | \
		sed 's/^/  /'

ml-clean:
	@printf "$(YELLOW)[CLEAN]$(RESET) Cleaning ML artifacts... "
	@find ml -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find ml -type f -name "*.pyc" -delete 2>/dev/null
	@rm -rf ml/checkpoints/ 2>/dev/null || true
	@echo "$(GREEN)$(CHECK)$(RESET)"

# ==============================================================================
# TEST SUITE (Comprehensive testing: C++ → Go → Python → Model)
# ==============================================================================

# Main test command - runs all tests with automatic service management
test: config proto-sync all ml-setup
	@echo "$(BOLD)$(BLUE)$(ARROW) Running Comprehensive Test Suite$(RESET)"
	@echo ""
	@$(PYTHON3) tests/run_all_tests.py --mode all --verbose 2>&1
	@echo ""

# Quick tests (unit only, no services required)
test-quick:
	@echo "$(BOLD)$(BLUE)$(ARROW) Running Quick Unit Tests$(RESET)"
	@$(PYTHON3) tests/run_all_tests.py --mode unit

# Individual component tests
test-cpp:
	@echo "$(CYAN)[TEST] C++ CSPRNG unit tests...$(RESET)"
	@cd tests/cpp_core && g++ -std=c++17 -O2 test_csprng.cpp -o test_csprng \
		-I/opt/homebrew/opt/openssl@3/include \
		-L/opt/homebrew/opt/openssl@3/lib \
		-lssl -lcrypto -lgmp && ./test_csprng

test-go:
	@echo "$(CYAN)[TEST] Go bridge unit tests...$(RESET)"
	@cd tests/go_bridge && go test -v bridge_integration_test.go

test-python:
	@echo "$(CYAN)[TEST] Python ML pipeline unit tests...$(RESET)"
	@$(PYTHON3) tests/python_ml/test_ml_pipeline.py

# Manual integration test (requires services running)
test-integration-manual:
	@echo "$(CYAN)[TEST] Manual Integration Test$(RESET)"
	@echo "$(YELLOW)Prerequisites: Start in separate terminals:$(RESET)"
	@echo "  Terminal 1: make run"
	@echo "  Terminal 2: make test-integration-manual"
	@$(PYTHON3) tests/integration/test_pipeline_v2.py

# Unit tests only (no service startup)
test-unit: test-cpp test-go test-python
	@echo "$(GREEN)$(CHECK) Tüm unit testler tamamlandı$(RESET)"

# Full test suite with automatic service management
test-all:
	@echo "$(BANNER)"
	@echo "$(BOLD)$(BLUE)$(ARROW) Running Full Test Suite with Services$(RESET)"
	@$(PYTHON3) tests/run_all_tests.py --mode all --verbose

# Cleanup test artifacts
test-clean:
	@echo "$(YELLOW)[TEST] Test yapı dosyaları temizleniyor...$(RESET)"
	@find tests -name "*.o" -delete
	@find tests -name "test_csprng" -delete
	@find tests -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)$(CHECK) Test dosyaları temizlendi$(RESET)"

.DEFAULT_GOAL := help
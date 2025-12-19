#!/bin/bash

# ==============================================================================
# PRIME-X SYSTEM LAUNCHER v5.3 (Core & Bridge Integrated)
# Renkli output + progress tracking ile C++ core ve Go bridge'i başlatır
# ==============================================================================

set -e

# --- ANSI COLORS ---
RED='\033[38;5;196m'
GREEN='\033[38;5;46m'
YELLOW='\033[38;5;226m'
BLUE='\033[38;5;39m'
CYAN='\033[38;5;51m'
MAGENTA='\033[38;5;129m'
BOLD='\033[1m'
RESET='\033[0m'

# --- CONFIG ---
SESSION="prime_x"
WIN_NAME="Prime-X_Live_System"
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

# Binaries kontrol et
check_binary() {
    local bin=$1
    local name=$2
    if [ ! -f "$bin" ]; then
        echo -e "${RED}✗ ERROR: $name binary not found at $bin${RESET}"
        echo -e "${YELLOW}  Please run: make cpp && make go${RESET}"
        exit 1
    fi
    echo -e "${GREEN}✓ Found: $name at $bin${RESET}"
}

echo -e "\n${CYAN}${BOLD}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}${BOLD}║  PRIME-X SYSTEM v5.3 - Launching Core & Bridge          ║${RESET}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════════════╝${RESET}\n"

echo -e "${CYAN}[1/4] Checking binaries...${RESET}"
check_binary "prime_core" "C++ Core (prime_core)"
check_binary "bridge/prime_bridge" "Go Bridge (prime_bridge)"

# 2. Config kontrolü
echo -e "\n${CYAN}[2/4] Verifying configuration...${RESET}"
if [ ! -f "config/config.json" ]; then
    echo -e "${RED}✗ config/config.json not found${RESET}"
    exit 1
fi
echo -e "${GREEN}✓ config/config.json found${RESET}"

# 3. Eski session temizle
echo -e "\n${CYAN}[3/4] Cleaning up old sessions...${RESET}"
tmux kill-session -t $SESSION 2>/dev/null || true
echo -e "${GREEN}✓ Session cleaned${RESET}"

# 4. Yeni session oluştur
echo -e "\n${CYAN}[4/4] Creating tmux session...${RESET}"
tmux new-session -d -s $SESSION -n "$WIN_NAME" -x 200 -y 50

# ============================================================
# PANE 0 (LEFT): GO BRIDGE (Secure Proxy & Verification)
# ============================================================
echo -e "\n${YELLOW}Starting GO BRIDGE (left pane)...${RESET}"
tmux select-pane -t 0
tmux send-keys "cd ${PROJECT_ROOT}/bridge && clear" C-m
tmux send-keys "echo -e '${CYAN}${BOLD}╔═══════════════════════════════════════════╗${RESET}'" C-m
tmux send-keys "echo -e '${CYAN}${BOLD}║  PRIME-X BRIDGE v5.3 (Integrity Verified) ║${RESET}'" C-m
tmux send-keys "echo -e '${CYAN}${BOLD}╚═══════════════════════════════════════════╝${RESET}'" C-m
tmux send-keys "echo -e ''" C-m
tmux send-keys "./prime_bridge" C-m

# ============================================================
# SPLIT SCREEN (Side by side)
# ============================================================
tmux split-window -h -p 50

# ============================================================
# PANE 1 (RIGHT): C++ CORE (Prime Number Generator)
# ============================================================
echo -e "${YELLOW}Starting C++ CORE (right pane)...${RESET}"
tmux select-pane -t 1
tmux send-keys "cd ${PROJECT_ROOT} && clear" C-m
tmux send-keys "echo -e '${YELLOW}${BOLD}Waiting for GO Bridge to initialize (2s)...${RESET}'" C-m
tmux send-keys "sleep 2 && ./prime_core" C-m

# ============================================================
# FINAL SETUP
# ============================================================
# Mouse support
tmux set -g mouse on

# Focus on Go Bridge
tmux select-pane -t 0

# Show info
echo -e "\n${GREEN}${BOLD}✓ System started successfully!${RESET}"
echo -e "${CYAN}Architecture:${RESET}"
echo -e "  ${YELLOW}LEFT PANE${RESET}:  GO BRIDGE - Packet verification (BLAKE2s + constant-time compare)"
echo -e "  ${YELLOW}RIGHT PANE${RESET}: C++ CORE - Prime number generation (color progress bars)"
echo -e "\n${MAGENTA}Data Flow:${RESET}"
echo -e "  C++ Core (ZMQ PUSH) → Packets with BLAKE2s trailer"
echo -e "  ↓"
echo -e "  GO Bridge (ZMQ PULL) → Verify integrity + buffer management"
echo -e "  ↓"
echo -e "  Python ML (gRPC) → Verified training data"
echo -e "\n${CYAN}Controls:${RESET}"
echo -e "  Ctrl+C in either pane to stop that component"
echo -e "  Ctrl+B then arrow keys to navigate between panes (tmux)"
echo -e "  Type ${BOLD}exit${RESET} to exit tmux entirely\n"

# Attach to session
tmux attach-session -t $SESSION
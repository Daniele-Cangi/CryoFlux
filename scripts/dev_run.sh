#!/usr/bin/env bash
# CryoFlux v0.1 - Development Run Script (Bash)
# Run from repo root: ./scripts/dev_run.sh

set -e

echo "[dev_run] CryoFlux v0.1 Development Runner"
echo ""

# Check if we're in the repo root
if [ ! -d "joule-agent-rs" ]; then
    echo "[ERROR] Please run this script from the CryoFlux repo root."
    echo "Usage: ./scripts/dev_run.sh"
    exit 1
fi

# Check if setup was run
if [ ! -f "joule-agent-rs/target/release/joule-agent-rs" ]; then
    echo "[ERROR] JouleAgent not built. Please run:"
    echo "  cd joule-agent-rs && cargo build --release"
    exit 1
fi

if [ ! -d "cryo-orchestrator/.venv" ]; then
    echo "[ERROR] Python venv not found. Please run:"
    echo "  cd cryo-orchestrator && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Generate default config.toml if missing
if [ ! -f "config.toml" ]; then
    echo "[INFO] config.toml not found, generating default configuration..."
    cat > config.toml << 'EOF'
# CryoFlux v0.1 Configuration (auto-generated)

[joule_agent]
hz = 2.0
cpu_tdp_w = 65.0
smoothing_alpha = 0.2
idle_learn_w = 5.0
bind_addr = "127.0.0.1:8787"

[orchestrator]
agent_url = "http://127.0.0.1:8787"
receipts_db = "./state/receipts.db"
seed = 42

[orchestrator.model]
encoder_model = "sentence-transformers/all-MiniLM-L6-v2"
clf_base = "distilbert-base-uncased"
lora_rank = 8

[orchestrator.data]
incoming_dir = "./data/incoming"
holdout_csv = "./data/holdout.csv"
embeddings_cache = "./state/embeddings"

[orchestrator.storage]
capsules_dir = "./state/capsules"
base_dir = "./state/base_model"
candidates_dir = "./state/candidates"

[orchestrator.energy]
min_joule_to_run = 1.0
task_index_est_joules = 20.0
task_lora_est_joules = 120.0

[orchestrator.merge]
lora_accept_min_delta = 0.003
merge_every_n_capsules = 1
EOF
    echo "[INFO] Created config.toml with default values"
fi

# Parse bind address from config.toml
BIND_ADDR="127.0.0.1:8787"
if [ -f "config.toml" ]; then
    BIND_ADDR=$(grep 'bind_addr' config.toml | sed 's/.*"\(.*\)".*/\1/' || echo "127.0.0.1:8787")
fi

AGENT_URL="http://${BIND_ADDR}"
echo "[1/3] Starting JouleAgent on ${BIND_ADDR}..."

# Start JouleAgent in background
cd joule-agent-rs
./target/release/joule-agent-rs > ../joule-agent.log 2>&1 &
AGENT_PID=$!
cd ..

echo "  JouleAgent started (PID: ${AGENT_PID})"

# Cleanup function
cleanup() {
    echo ""
    echo "[INFO] Shutting down..."
    echo "[INFO] Stopping JouleAgent (PID: ${AGENT_PID})..."
    kill $AGENT_PID 2>/dev/null || true
    wait $AGENT_PID 2>/dev/null || true
    echo "[INFO] Shutdown complete"
    exit 0
}

# Register cleanup on Ctrl+C
trap cleanup INT TERM

# Wait for JouleAgent to be ready
echo "[2/3] Waiting for JouleAgent to be ready..."
MAX_ATTEMPTS=20
ATTEMPT=0
AGENT_READY=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    sleep 0.5
    if curl -s "${AGENT_URL}/v1/sample" > /dev/null 2>&1; then
        AGENT_READY=1
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    echo "  Attempt ${ATTEMPT}/${MAX_ATTEMPTS}..."
done

if [ $AGENT_READY -eq 0 ]; then
    echo "[ERROR] JouleAgent did not become ready after ${MAX_ATTEMPTS} attempts"
    echo "[INFO] Stopping JouleAgent (PID: ${AGENT_PID})..."
    kill $AGENT_PID 2>/dev/null || true
    exit 1
fi

echo "  JouleAgent is ready!"
echo ""

# Start Orchestrator in foreground
echo "[3/3] Starting Orchestrator..."
echo ""
echo "======================================"
echo "CryoFlux v0.1 is running!"
echo "JouleAgent: ${AGENT_URL}"
echo "Press Ctrl+C to stop"
echo "======================================"
echo ""

cd cryo-orchestrator

# Activate venv and run orchestrator
source .venv/bin/activate
python -u cryo.py || cleanup

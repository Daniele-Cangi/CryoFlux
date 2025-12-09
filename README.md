# CryoFlux — Proof of Learning

**Energy → Intelligence: Verify real learning improvement per joule with cryptographic receipts.**

CryoFlux is a system that directly links real energy consumption (CPU/GPU power) to measurable model improvement (Δ). Every joule spent is traceable, every improvement is audited, and the connection between energy and capability is verifiable.

## The Idea

Modern AI training consumes enormous energy, but there's **no verified link between energy spent and actual capability gained**. CryoFlux answers: **"How much real improvement did we get per joule?"**

The system:
1. **Measures** real energy (CPU/GPU power via NVIDIA NVML)
2. **Spends** energy only on learning (LoRA micro-updates)
3. **Measures** improvement on a fixed holdout set (Δ = loss_before − loss_after)
4. **Accepts** updates only if improvement is real (Δ > threshold)
5. **Records** every decision with cryptographic receipts

**Result**: A verifiable audit trail proving "X joules → Y Δ improvement."

For details, see [WHITEPAPER.md](WHITEPAPER.md).
<img width="1646" height="898" alt="Screenshot 2025-10-26 035630" src="https://github.com/user-attachments/assets/44516fcd-6e11-444f-8beb-5698fdb8d221" />
---

## Dev Quick Start

Get CryoFlux running locally as a complete "CryoFlux Lab" with two commands:

### Windows (PowerShell)
```powershell
# 1. Setup (one-time: build JouleAgent + create venv + install deps)
.\scripts\dev_setup.ps1

# 2. Run (starts JouleAgent + Orchestrator together)
.\scripts\dev_run.ps1
```

### Linux/macOS (Bash)
```bash
# 1. Make script executable (one-time)
chmod +x scripts/dev_run.sh

# 2. Run (builds if needed, starts both components)
./scripts/dev_run.sh
```

**What happens:**
- `dev_setup.ps1` (Windows only): Builds JouleAgent, creates Python venv, installs dependencies, creates placeholder data directories
- `dev_run` scripts: Start JouleAgent in background, wait for `/v1/sample` readiness, then start Orchestrator in foreground with live logs
- Press `Ctrl+C` to stop both components cleanly

This turns your local machine into a **CryoFlux Lab**: a complete energy-to-intelligence verification system running on your hardware.

**Configuration:** All settings centralized in [`config.toml`](config.toml) at repo root. See [Configuration](#configuration-configtoml--env-overrides) section below.

---

## Components Overview

CryoFlux consists of four main components:

### 1. **JouleAgent** (Rust) — Energy Measurement Daemon

Real-time energy monitoring daemon that exposes an HTTP API for the orchestrator.

**Capabilities:**
- CPU power estimation (TDP-based model)
- GPU power measurement (NVIDIA NVML for precise wattage)
- Net power calculation: `net_w = gross_w - idle_baseline_w`
- Joule accumulation: `bucket_j += net_w * Δt`
- Atomic energy withdrawal via `/v1/take`

**API Endpoints:**
- `GET /v1/sample` — Current energy state (bucket_j, net_w, cpu_w, gpu_w, etc.)
- `POST /v1/take {"joules": N}` — Atomically reserve N joules (returns `{"ok": true/false}`)

**Configuration:** Reads from `[joule_agent]` section of `config.toml`, with env var overrides.

### 2. **Orchestrator** (Python) — Task Execution & Learning Loop

The main CryoFlux loop: budget-aware task selection, LoRA training, Δ evaluation, and receipt recording.

**Capabilities:**
- Energy-aware task scheduling (via η-aware scheduler or legacy threshold-based)
- Two task types:
  - **TaskIndex** (`index_refresh`): Semantic embedding refresh (≥20J)
  - **TaskLoRA** (`lora_delta`): LoRA adapter training + evaluation (≥120J)
- Holdout-based Δ measurement with cryptographic hashing
- SQLite receipts database (append-only audit log)

**Task Details:**

*TaskIndex:*
- Reads texts from `data/incoming/*.txt`
- Computes novelty via compression (zlib)
- Embeds new texts with `sentence-transformers/all-MiniLM-L6-v2`
- Updates FAISS index in `state/embeddings/`
- Δ = `embeddings_added / 1000.0`

*TaskLoRA:*
- Base model: `distilbert-base-uncased` (CPU or CUDA)
- LoRA adapter training: ~0.44% trainable params (rank=8)
- Training data: First 256 samples from `data/holdout.csv`
- Evaluation holdout: Up to 512 samples from same file (ensures consistency)
- Computes: `base_loss`, `new_loss`, `Δ = max(0, base_loss - new_loss)`, plus accuracy deltas
- **Acceptance criteria:** `Δ ≥ 0.002 OR Δacc ≥ 0.01`
- On accept: Merges adapter into base, writes new base to `state/base_model/`, records adapter path in receipt metadata

**Configuration:** Reads from `[orchestrator]` section of `config.toml`, with env var overrides.

### 3. **Analysis Layer** (`analysis/`) — CryoFlux Lab Efficiency Tools

Read-only analysis tools for studying energy-to-learning efficiency (η = Δ/J).

**Key Scripts:**
- `analysis/metrics.py` — Core functions: `open_db()`, `compute_global_metrics()`, `load_eta_series()`
- `analysis/report.py` — CLI efficiency report (global + per-task metrics)
- `analysis/update_task_stats.py` — Aggregates receipts into `task_stats` table for scheduler
- `analysis/plot_eta.py` — Optional matplotlib visualization (η over time)

**Database Extensions:**
- `task_stats` table: Per-task aggregates (runs, joules_total, delta_total, eta_avg, accepted_runs)
- `receipts_canonical` view: Normalized receipts with extracted `accepted` flag from JSON metadata

**Design Principles:**
- **Read-only:** Never modifies learning behavior or decision thresholds
- **Backward compatible:** Works with existing receipts
- **Modular:** Core functions reusable for custom analysis

See [CryoFlux Lab – Efficiency Analysis](#cryoflux-lab--efficiency-analysis) for usage examples.

### 4. **η-Aware Scheduler** (`cryo-orchestrator/scheduler.py`) — Bandit-Based Task Selection

Optional UCB (Upper Confidence Bound) bandit scheduler with ε-greedy exploration for dynamic task prioritization based on η.

**Algorithm:**
- **Warmup phase:** Ensure each task gets `warmup_runs` executions to establish baseline η
- **UCB scoring:** `score_i = η_i + c × sqrt(2 × ln(N) / n_i)` balances exploitation vs exploration
- **ε-greedy layer:** With probability `epsilon`, force exploration of under-explored tasks (typically the task with fewest runs)
- **Energy feasibility:** Only considers tasks where `bucket_j >= est_joules × min_bucket_factor`
- **Fallback:** If unavailable or disabled, reverts to legacy threshold-based selection

**Integration:**
- Reads `task_stats` from receipts database (populated by `update_task_stats.py`)
- Strictly read-only: never writes to DB
- Logs decisions with reason: `WARMUP`, `BANDIT`, or `EPSILON`

**Configuration:** Controlled via `[orchestrator.scheduler]` section in `config.toml`.

See [η-Aware Scheduler (Bandit + ε-Greedy)](#η-aware-scheduler-bandit--ε-greedy) for details.

---

## Configuration (config.toml + env overrides)

All configuration is centralized in [`config.toml`](config.toml) at the repo root. Both JouleAgent and Orchestrator read from this file.

### config.toml Structure

```toml
# JouleAgent config
[joule_agent]
hz = 2.0                        # Sampling frequency (Hz)
cpu_tdp_w = 65.0                # CPU TDP for power estimation
smoothing_alpha = 0.2           # Exponential smoothing factor
idle_learn_w = 5.0              # Idle baseline learning threshold (W)
bind_addr = "127.0.0.1:8787"    # HTTP server bind address

# Orchestrator core
[orchestrator]
agent_url = "http://127.0.0.1:8787"
receipts_db = "./state/receipts.db"
seed = 42

[orchestrator.model]
encoder_model = "sentence-transformers/all-MiniLM-L6-v2"
clf_base = "distilbert-base-uncased"
lora_rank = 8

[orchestrator.energy]
min_joule_to_run = 1.0
task_index_est_joules = 20.0    # Estimated energy for index_refresh
task_lora_est_joules = 120.0    # Estimated energy for lora_delta

[orchestrator.data]
incoming_dir = "./data/incoming"
holdout_csv = "./data/holdout.csv"
embeddings_cache = "./state/embeddings"

[orchestrator.storage]
capsules_dir = "./state/capsules"
base_dir = "./state/base_model"
candidates_dir = "./state/candidates"

[orchestrator.merge]
lora_accept_min_delta = 0.003
merge_every_n_capsules = 1

# η-aware scheduler
[orchestrator.scheduler]
bandit_enabled = true           # Enable/disable scheduler (false = legacy thresholds)
ucb_c = 0.3                     # Exploration constant (higher = more exploration)
warmup_runs = 0                 # Minimum runs before bandit (0 = immediate bandit mode)
min_bucket_factor = 1.0         # Task eligible only if bucket_j >= est_joules × factor
epsilon = 0.1                   # ε-greedy exploration probability (10% exploration)
```

### Environment Variable Overrides

**JouleAgent:**
- `JOULE_HZ` — Override sampling frequency
- `JOULE_CPU_TDP_W` — Override CPU TDP
- `JOULE_SMOOTHING_ALPHA` — Override smoothing factor
- `JOULE_IDLE_LEARN_W` — Override idle learning threshold
- `JOULE_BIND_ADDR` — Override bind address

**Orchestrator:**
- `JOULE_AGENT_URL` — Override JouleAgent endpoint URL

**Example (Windows PowerShell):**
```powershell
$env:JOULE_HZ="2.5"
$env:JOULE_IDLE_LEARN_W="8.0"
.\scripts\dev_run.ps1
```

**Example (Linux/Bash):**
```bash
export JOULE_HZ=2.5
export JOULE_IDLE_LEARN_W=8.0
./scripts/dev_run.sh
```

---

## Receipts and Database Schema

CryoFlux maintains an **append-only audit log** of all task executions in SQLite.

### receipts Table (Core Schema)

```sql
CREATE TABLE receipts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL,              -- Unix timestamp
    task TEXT,            -- Task name ("index_refresh" | "lora_delta")
    joule REAL,           -- Energy spent (joules)
    sec REAL,             -- Execution time (seconds)
    delta REAL,           -- Improvement metric (Δ)
    loss REAL,            -- Final loss value
    delta_hash TEXT,      -- Cryptographic hash of result
    meta TEXT             -- JSON metadata (accepted, adapter_path, etc.)
);
```

**Semantics:**
- `delta` for **LoRA**: `Δ = max(0, base_loss - new_loss)`
- `delta` for **Index**: `Δ = embeddings_added / 1000.0`
- `meta.accepted` (bool): Whether update was accepted and merged
- `meta.adapter` (str): Path to LoRA adapter weights (if applicable)
- `meta.base_acc`, `meta.new_acc`, `meta.delta_acc`: Accuracy metrics (LoRA only)

### receipts_canonical View

Normalized view created automatically by analysis tools:

```sql
CREATE VIEW IF NOT EXISTS receipts_canonical AS
SELECT
    id,
    datetime(ts, 'unixepoch') AS timestamp,
    ts AS ts_unix,
    task AS task_name,
    joule AS joules_spent,
    sec AS execution_time_sec,
    delta,
    loss,
    delta_hash,
    meta,
    CASE
        WHEN json_extract(meta, '$.accepted') = 1 THEN 1
        WHEN json_extract(meta, '$.accepted') = 'true' THEN 1
        ELSE 0
    END AS accepted
FROM receipts;
```

### task_stats Table (Scheduler Data)

Aggregated per-task statistics, populated by `analysis/update_task_stats.py`:

```sql
CREATE TABLE task_stats (
    task_name TEXT PRIMARY KEY,
    runs INTEGER NOT NULL,
    joules_total REAL NOT NULL,
    delta_total REAL NOT NULL,
    eta_avg REAL NOT NULL,           -- Average η = delta_total / joules_total
    accepted_runs INTEGER NOT NULL,
    last_run_at TEXT NOT NULL        -- ISO8601 timestamp
);
```

**Purpose:** Provides fast lookups for the η-aware scheduler without scanning all receipts.

**Maintenance:** Run `python analysis/update_task_stats.py` periodically to refresh aggregates.

### Database Path Resolution

Analysis tools and scheduler use fallback logic:

1. Try `./state/receipts.db` (expected location when running from repo root)
2. Fall back to `./cryo-orchestrator/state/receipts.db` (if running from cryo-orchestrator/)
3. Use configured path from `config.toml` as final fallback

**Recommendation:** Always run analysis scripts from repo root for consistent behavior.

---

## CryoFlux Lab – Efficiency Analysis

The **CryoFlux Lab** provides read-only analysis tools for studying energy-to-learning efficiency without modifying core learning behavior.

### Energy Efficiency Metric (η)

Define **η (eta)** as the energy-to-learning efficiency:

$$\eta = \frac{\Delta}{\text{joules spent}}$$

**Interpretation:**
- Higher η = more learning per joule (better efficiency)
- η varies by task type (index_refresh typically ~0.015, lora_delta typically ~0.000003)
- Rejected updates have Δ=0 → η=0

**Example:**
- Task A: 100J → Δ=0.01 → η=0.0001
- Task B: 80J → Δ=0.03 → η=0.000375 ✓ (more efficient)

### Usage Workflow

**1. Run CryoFlux and generate receipts:**
```powershell
.\scripts\dev_run.ps1
# Let it run for a while to accumulate receipts
```

**2. Update task statistics (for scheduler):**
```bash
python analysis/update_task_stats.py
```

Output:
```
============================================================
CryoFlux Lab - Task Statistics Aggregator
============================================================
[INFO] Ensured receipts_canonical view exists
[update_task_stats] task_stats table ready
[update_task_stats] Updated index_refresh: runs=30, η_avg=0.015000
[update_task_stats] Updated lora_delta: runs=12, η_avg=0.000003
[update_task_stats] Successfully updated 2 task(s)
```

**3. Generate efficiency report:**
```bash
# Global report
python analysis/report.py

# Focus on specific task
python analysis/report.py --task lora_delta

# Include rolling η over last 20 receipts
python analysis/report.py --task index_refresh --window 20

# Use custom database path
python analysis/report.py --db ./custom/path/receipts.db
```

**Example output:**
```
============================================================
CryoFlux Lab – Efficiency Report
============================================================

[INFO] Using database: ./state/receipts.db

[GLOBAL METRICS]
  Total receipts:        42
  Accepted:              38 (90.5%)
  Rejected:              4  (9.5%)
  Total joules spent:    2040.00 J
  Total Δ:               0.4500
  Mean η (Δ/J):          0.000221

[BY TASK]
  Task                 Runs     J_total      Δ_total      η_avg           Accept%
  ---------------------------------------------------------------------------------------
  index_refresh        30       600.0        0.4500       0.000750        100.0
  lora_delta           12       1440.0       0.0000       0.000000        0.0

[ROLLING η (window=20)]
  Recent 20 receipts:
    Average η: 0.000650
    Receipt IDs: 23 - 42

============================================================
Report complete
============================================================
```

**4. Visualize η over time (optional):**
```bash
# Install matplotlib (one-time)
pip install matplotlib

# Generate global plot
python analysis/plot_eta.py

# Plot specific task with rolling average
python analysis/plot_eta.py --task lora_delta --window 10

# Custom output path
python analysis/plot_eta.py --output ./my_eta_plot.png
```

Output: Saves plot to `./state/eta_global.png` (or `./state/eta_TASK.png` for task-specific plots)

### Training Ledger (accepted merges)

The **ledger exporter** creates a structured, append-only audit trail of all learning events in JSONL format (newline-delimited JSON). Each receipt is exported as a single JSON object containing: `receipt_id`, `timestamp`, `task_name`, `joules`, `delta`, `accepted`, `eta`, `kwh_eff`, `cost_eur`, `co2_g`, `delta_hash`, and `meta_raw`.

This ledger is designed for external consumption: cryptographic verification, blockchain notarization, token systems, or compliance auditing. The `--accepted-only` flag filters to successful merges only, creating a verifiable chain of learning improvements.

**Usage:**
```bash
# Full ledger (all receipts)
python analysis/export_ledger.py > out/ledger_full.jsonl

# Only accepted merges
python analysis/export_ledger.py --accepted-only > out/ledger_accepted.jsonl

# Export to specific file
python analysis/export_ledger.py --accepted-only --output ledger_accepted.jsonl
```

**Key properties:**
- **Append-only**: Read from receipts database, never modifies
- **Machine-readable**: JSONL format for easy parsing by external tools
- **Cryptographically auditable**: Each record includes `delta_hash` for verification
- **Energy-aware**: Includes joules, kWh, cost (EUR), and CO2 emissions per receipt
- **Signable**: Can be hashed, timestamped, or notarized for immutable audit trails

### Analysis Tool Reference

| Script | Purpose | Key Arguments |
|--------|---------|---------------|
| `update_task_stats.py` | Aggregate receipts into `task_stats` | None (idempotent, safe to rerun) |
| `report.py` | Efficiency report | `--task TASK`, `--window N`, `--db PATH` |
| `cost_report.py` | Cost & impact report | None (reads from `config.toml`) |
| `export_ledger.py` | Export JSONL audit ledger | `--accepted-only`, `--output PATH` |
| `plot_eta.py` | Visualize η over time | `--task TASK`, `--window N`, `--output PATH` |

**Database Path Resolution:**
- All scripts try `./state/receipts.db` first
- Fall back to `./cryo-orchestrator/state/receipts.db` with `[WARN]` message
- Explicitly override with `--db PATH` argument

---

## η-Aware Scheduler (Bandit + ε-Greedy)

CryoFlux includes an **optional η-aware scheduler** that dynamically prioritizes tasks based on their energy-to-learning efficiency.

### How It Works

The scheduler uses a **multi-armed bandit algorithm** (UCB - Upper Confidence Bound) with an **ε-greedy exploration layer** to balance:
- **Exploitation:** Run tasks with proven high η (maximize short-term learning)
- **Exploration:** Try under-explored tasks to discover their true efficiency (maximize long-term knowledge)

### Algorithm Phases

**1. Warmup Phase**
- Each task gets at least `warmup_runs` executions to establish baseline η
- Selection reason: `WARMUP`
- Ensures scheduler has sufficient data before applying bandit logic

**2. Bandit Selection (UCB)**
- Compute UCB score for each energy-feasible task:

```
score_i = η_i + c × sqrt(2 × ln(N) / n_i)

Where:
  η_i = average efficiency for task i (from task_stats.eta_avg)
  n_i = number of runs for task i (from task_stats.runs)
  N   = total runs across all tasks
  c   = exploration constant (config: ucb_c, default 0.3)
```

- Select task with highest score
- Selection reason: `BANDIT`

**3. ε-Greedy Exploration Layer**
- With probability `epsilon` (config: default 0.1 = 10%), override bandit selection
- Choose the task with **fewest runs** among eligible tasks (exploration)
- Selection reason: `EPSILON`
- This prevents permanent starvation of low-η tasks like `lora_delta`

**4. Energy Feasibility Filter**
- A task is considered only if: `bucket_j >= est_joules × min_bucket_factor`
- Ensures scheduler respects energy constraints

**5. Fallback to Legacy**
- If scheduler unavailable, disabled, or returns `None`, fall back to legacy threshold-based selection:
  - `bucket_j >= 120J` → lora_delta
  - `bucket_j >= 20J` → index_refresh

### Configuration

All scheduler parameters are in `[orchestrator.scheduler]` section of `config.toml`:

```toml
[orchestrator.scheduler]
# Enable/disable scheduler (false = revert to legacy threshold-based)
bandit_enabled = true

# Exploration constant (higher = more exploration)
# Typical range: 0.1-1.0
ucb_c = 0.3

# Minimum runs per task before applying bandit
# Set to 0 to use bandit immediately (if task_stats already exists)
warmup_runs = 0

# Task eligible only if bucket_j >= est_joules × min_bucket_factor
# 1.0 = exact threshold, >1.0 = require more energy buffer
min_bucket_factor = 1.0

# Probability of epsilon-greedy exploration (0.0-1.0)
# 0.1 = 10% exploration, 0.0 = pure bandit
epsilon = 0.1
```

### Usage

**1. Enable scheduler (default):**
```toml
[orchestrator.scheduler]
bandit_enabled = true
```

**2. Run orchestrator:**
```powershell
.\scripts\dev_run.ps1
```

**3. Periodically update task_stats (for scheduler data):**
```bash
python analysis/update_task_stats.py
```

**4. Observe scheduler decisions in logs:**
```
[SCHEDULER] Connected to ./state/receipts.db
[SCHEDULER] BANDIT selected task=index_refresh score=0.015234 η=0.015000 runs=10 bucket_j=45.30J
[SCHEDULER] using task=index_refresh reason=BANDIT score=0.015234

[SCHEDULER] EPSILON selected task=lora_delta η=0.000003 runs=5 bandit_task=index_refresh r=0.042 epsilon=0.100
[SCHEDULER] using task=lora_delta reason=EPSILON

[SCHEDULER] WARMUP selected task=lora_delta runs=2/3
[SCHEDULER] using task=lora_delta reason=WARMUP
```

### Example Scenario: Avoiding Starvation

**Typical η values (from real runs):**
- `index_refresh`: η ≈ 0.015 (highly efficient)
- `lora_delta`: η ≈ 0.000003 (low efficiency, low acceptance rate)

**Without ε-greedy:** Pure UCB would almost always select `index_refresh` since its η is ~5000× higher.

**With ε-greedy (epsilon=0.1):**
- ~90% of time: Select `index_refresh` (BANDIT, highest η)
- ~10% of time: Select `lora_delta` (EPSILON, forced exploration)
- Result: `lora_delta` still gets periodic training attempts despite low η

This is critical because:
1. Low η doesn't mean zero value (some LoRA updates are accepted)
2. Early η estimates may be inaccurate (need more data)
3. Exploration maintains option value for future improvements

### Important Notes

**Warmup Staleness:**
- The scheduler's warmup counter is based on `task_stats.runs`, which is only updated when you run `python analysis/update_task_stats.py`
- During a session, warmup counts don't update in real-time
- **Example:** If `task_stats` shows `runs=2` but you've executed 5 more tasks, scheduler still sees `runs=2`

**Workarounds:**
1. **Set `warmup_runs = 0`** if you already have historical data (recommended)
2. Run `update_task_stats.py` between orchestrator sessions to refresh counts
3. Use higher `warmup_runs` initially for cold starts

**Design Rationale:**
- This is an intentional trade-off: scheduler is **read-only** and uses **offline-aggregated data** for simplicity
- Maintains clean separation: scheduler doesn't write to DB, only reads pre-computed stats

**What the Scheduler Does NOT Change:**
- How Δ is computed (still based on holdout evaluation)
- Acceptance thresholds (`lora_accept_min_delta`)
- Task execution logic (LoRA training, index refresh)
- Receipt recording

**The scheduler only decides:** Which task to run next, based on η and energy availability.

### Disabling the Scheduler

To revert to legacy threshold-based selection:

```toml
[orchestrator.scheduler]
bandit_enabled = false
```

Legacy behavior:
- `bucket_j >= task_lora_est_joules (120J)` → lora_delta
- `bucket_j >= task_index_est_joules (20J)` → index_refresh
- Simple, deterministic, no η awareness

---

## Manual Startup (Advanced)

For manual control or debugging, you can start components separately.

### Prerequisites
- Python 3.10+
- Rust 1.70+ (for JouleAgent)
- NVIDIA GPU with CUDA 12.4 (for GPU energy monitoring and LoRA training)
- pip, venv

### Manual Setup Steps

**1. Clone and navigate:**
```bash
git clone https://github.com/Daniele-Cangi/CryoFlux.git
cd CryoFlux
```

**2. Build JouleAgent:**
```powershell
cd joule-agent-rs
cargo build --release
cd ..
```

**3. Set up Python environment:**
```bash
cd cryo-orchestrator
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
cd ..
```

**4. Create config.toml (optional):**

If `config.toml` doesn't exist, the system will use defaults. To customize, create [`config.toml`](config.toml) at repo root (see [Configuration](#configuration-configtoml--env-overrides) section).

**5. Prepare data (optional):**

The setup script creates placeholder data automatically. To use your own:

Create `data/holdout.csv` (sentiment data for evaluation):
```csv
"text",label
"This product is amazing!",1
"Terrible experience.",0
```

Create `data/incoming/news.txt` (data to fine-tune on):
```
This is a positive review.
This is a negative review.
```

### Manual Run

**Terminal 1 (JouleAgent):**
```powershell
cd joule-agent-rs
# Optional: override config with env vars
$env:JOULE_HZ="2.0"; $env:JOULE_IDLE_LEARN_W="5.0"
cargo run --release
```

**Terminal 2 (Orchestrator):**
```bash
cd cryo-orchestrator
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
python -u cryo.py
```

Watch the logs for task execution and receipts.

---

## Example Output

```
[JouleAgent] Listening on 127.0.0.1:8787
[JouleAgent] Sample: bucket_j=45.3, net_w=18.87, gpu_w=12.5, cpu_w=6.4

[CryoFlux] Loaded config from ..\config.toml
[CryoFlux] Orchestrator online — expecting JouleAgent at http://127.0.0.1:8787
[SCHEDULER] Connected to ./state/receipts.db

[SCHEDULER] BANDIT selected task=index_refresh score=0.015234 η=0.015000 runs=10 bucket_j=45.30J
[SCHEDULER] using task=index_refresh reason=BANDIT score=0.015234
[Index] Computing embeddings... done
[CryoFlux] index_refresh → Δ=0.0040 | ok=True | receipt=8d0ae8aa…

[SCHEDULER] EPSILON selected task=lora_delta η=0.000003 runs=5 bandit_task=index_refresh r=0.042 epsilon=0.100
[SCHEDULER] using task=lora_delta reason=EPSILON
[LoRA] Using device: cuda (NVIDIA GeForce RTX 2060)
[LoRA] trainable params: 294912/67249922 (0.439%)
[EVAL] base_loss=0.6505 new_loss=0.6270 Δ=0.0235 | base_acc=0.762 new_acc=0.994
[CryoFlux] lora_delta → Δ=0.0235 | ok=True | receipt=f3b9c1de…
```

---

## Architecture Decisions

### Why LoRA?
- **Non-destructive**: Adapter discarded if rejected, base model never corrupted
- **Efficient**: Only ~0.44% of parameters trainable (294K out of 67M)
- **Fast**: Quick to train and evaluate (critical for real-time energy budgeting)
- **Reversible**: Failed updates have zero impact on system state

### Why frozen idle baseline?
- **Accurate**: Doesn't chase transient load fluctuations, measures true net power
- **Stable**: Locked early during startup, consistent throughout session
- **Conservative**: Prevents over-crediting idle consumption as "learning energy"

### Why holdout evaluation?
- **Reproducible**: Fixed holdout set ensures consistent Δ measurement across runs
- **Unbiased**: Holdout data never seen during adapter training
- **Verifiable**: Same holdout → same Δ for same adapter (cryptographic hashing confirms)

### Why read-only analysis layer?
- **Safety**: Analysis cannot accidentally modify learning behavior
- **Separation of concerns**: Metrics computation decoupled from orchestrator
- **Transparency**: Users can inspect efficiency without affecting system operation

---

## Limitations

- **CPU power estimation**: TDP-based model is approximate; GPU power (NVIDIA NVML) is more reliable
- **Small improvements**: LoRA updates are incremental; many cycles needed for large cumulative Δ
- **Single-node**: Currently local-only; multi-node network verification planned for v0.2+
- **Data quality**: Noisy or imbalanced holdout data leads to noisy Δ signal
- **Scheduler staleness**: Task statistics require manual refresh via `update_task_stats.py` (intentional design trade-off for read-only architecture)

---

## Testing

**System healthcheck:**
```bash
cd cryo-orchestrator
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
python healthcheck.py
```

**Stress test (GPU only):**
```bash
cd cryo-orchestrator
python stress_gpu_only.py
```

**Stress test (CPU+GPU mix):**
```bash
cd cryo-orchestrator
python stress_mix.py
```

These generate controlled load while the orchestrator accumulates joules and executes tasks.

---

## Files

```
CryoFlux/
├── config.toml                  # Centralized configuration (JouleAgent + Orchestrator)
├── scripts/
│   ├── dev_setup.ps1            # Windows: one-time setup
│   ├── dev_run.ps1              # Windows: run JouleAgent + Orchestrator
│   └── dev_run.sh               # Linux/macOS: run JouleAgent + Orchestrator
├── joule-agent-rs/              # Rust energy daemon
│   ├── src/main.rs
│   ├── Cargo.toml
│   └── target/release/joule-agent-rs
├── cryo-orchestrator/           # Python orchestrator
│   ├── cryo.py                  # Main CryoFlux loop
│   ├── scheduler.py             # η-aware UCB+ε-greedy scheduler
│   ├── healthcheck.py           # System healthcheck
│   ├── requirements.txt
│   └── .venv/
├── analysis/                    # CryoFlux Lab - efficiency analysis
│   ├── __init__.py
│   ├── metrics.py               # Core metric computation functions
│   ├── report.py                # CLI efficiency report
│   ├── cost_report.py           # Cost & environmental impact report
│   ├── export_ledger.py         # JSONL ledger exporter (audit trail)
│   ├── plot_eta.py              # η visualization (requires matplotlib)
│   ├── update_task_stats.py     # Task statistics aggregator
│   └── sql/
│       └── views.sql            # receipts_canonical view definition
├── data/                        # Data (not tracked)
│   ├── holdout.csv              # Evaluation dataset (text, label)
│   └── incoming/
│       └── *.txt                # Texts for semantic indexing
├── state/                       # Runtime state (not tracked)
│   ├── receipts.db              # Main receipts + task_stats database
│   ├── capsules/                # LoRA adapter snapshots
│   ├── base_model/              # Current merged base model
│   ├── candidates/              # Staging for merge candidates
│   └── embeddings/
│       └── faiss.index          # Semantic embeddings FAISS index
├── WHITEPAPER.md                # Full technical details
├── README.md                    # This file
└── .gitignore
```

---

## Future Work (v0.2+)

- ✅ η-based scheduler (bandit + ε-greedy for task prioritization)
- ✅ Central config.toml (unified configuration for all components)
- ✅ Analysis layer (read-only efficiency metrics and visualization)
- ⬜ Versioning & rollback (keep last-N base model checkpoints)
- ⬜ Dashboard (real-time energy/Δ/η tracking web UI)
- ⬜ P2P verification (multi-node network with receipt gossiping)
- ⬜ Proof-of-Learning consensus mechanism (cryptographic proof aggregation)

---

## License

MIT

---

## Contributing

Questions or ideas? Open an issue or reach out.

**CryoFlux is a research prototype demonstrating the principle: "If you can measure it, you can improve it. If you can audit it, you can trust it."**

---

**Status**: ✅ **v0.1** (local lab functional with η-aware scheduler) | 🔄 **v0.2** (multi-node, in design)

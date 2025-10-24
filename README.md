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

---

## Architecture

### Two Components

#### **JouleAgent** (Rust) — Energy Measurement
- Real-time CPU/GPU power monitoring
- Net power calculation (gross − idle baseline)
- Joule bucket accumulation
- HTTP API for sampling and withdrawal

**Endpoints:**
- `GET /v1/sample` — Current energy state
- `POST /v1/take {joules}` — Withdraw joules atomically

**Start:**
```powershell
cd joule-agent-rs
$env:JOULE_HZ="2.0"
$env:JOULE_IDLE_LEARN_W="5.0"
cargo run --release
```

#### **Orchestrator** (Python) — Task Execution & Learning
- Budget-aware task selection
- LoRA micro-finetuning on base model
- Holdout-based Δ evaluation
- Receipt system (SQLite)

**Tasks:**
- **Index** (≥20J): Refresh semantic embeddings
- **LoRA Delta** (≥120J): Fine-tune model on subset, evaluate on holdout

**Start:**
```bash
cd cryo-orchestrator
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
python -u cryo.py
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Rust 1.70+ (for JouleAgent)
- NVIDIA GPU with CUDA 12.4 (for GPU energy monitoring and LoRA training)
- pip, venv

### Setup

**1. Clone and navigate:**
```bash
git clone https://github.com/Daniele-Cangi/CryoFlux.git
cd CryoFlux
```

**2. Build JouleAgent:**
```powershell
cd joule-agent-rs
cargo build --release
```

**3. Set up Python environment:**
```bash
cd ../cryo-orchestrator
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**4. Prepare data:**

Create `data/holdout.csv` (sentiment data for evaluation):
```
"text",label
"This product is amazing!",1
"Terrible experience.",0
...
```

Create `data/incoming/news.txt` (data to fine-tune on):
```
This is a positive review.
This is a negative review.
...
```

**5. Run:**

Terminal 1 (JouleAgent):
```powershell
cd joule-agent-rs
$env:JOULE_HZ="2.0"; $env:JOULE_IDLE_LEARN_W="5.0"
cargo run --release
```

Terminal 2 (Orchestrator):
```bash
cd cryo-orchestrator
python -u cryo.py
```

Watch the logs for task execution and receipts.

---

## Configuration

### JouleAgent (`joule-agent-rs/src/main.rs`)
- `JOULE_HZ`: Sampling frequency (default 2.0 = 0.5s period)
- `JOULE_IDLE_LEARN_W`: Threshold for idle baseline learning (default 5.0W)

### Orchestrator (`cryo-orchestrator/cryo.py`)
- `EnergyConfig.agent_url`: JouleAgent endpoint (default `http://127.0.0.1:8787`)
- `ModelConfig.clf_base`: Base model to fine-tune (default `distilbert-base-uncased`)
- `LoraConfig`: LoRA hyperparameters (rank=8, alpha=16)
- `MergeConfig.lora_accept_min_delta`: Acceptance threshold (default 0.002)
- `EnergyConfig.task_index_est_joules`: Energy budget for Index task (default 20J)
- `EnergyConfig.task_lora_est_joules`: Energy budget for LoRA task (default 120J)

---

## Output & Receipts

Every accepted task generates a **receipt** in `receipts.db`:

```json
{
  "task_id": "lora_1729734256",
  "task_name": "lora_delta",
  "joules_spent": 120.0,
  "base_loss": 0.6505,
  "new_loss": 0.6270,
  "delta": 0.0235,
  "accepted": true,
  "adapter_hash": "abc123...",
  "base_model_hash": "def456...",
  "timestamp": 1729734256,
  "receipt_hash": "ghi789..."
}
```

**Query receipts:**
```bash
cd cryo-orchestrator
sqlite3 receipts.db "SELECT task_name, joules_spent, delta, accepted FROM receipts LIMIT 10;"
```

---

## Example Output

```
[JouleAgent] Listening on 127.0.0.1:8787
[JouleAgent] Sample: bucket_j=45.3, net_w=18.87, gpu_w=12.5, cpu_w=6.4

[Orchestrator] Starting energy loop...
[Orchestrator] bucket_j=45.3 | choose: Index (target 20J)
[Index] Computing embeddings... done (18.2J)

[Orchestrator] bucket_j=27.1 | choose: LoRA (target 120J)
[Orchestrator] Insufficient joules (27.1 < 120), waiting...

[GPU Stress] Running stress test...
[Orchestrator] bucket_j=152.4 | choose: LoRA
[LoRA] Using device: cuda (NVIDIA GeForce RTX 2060)
[LoRA] trainable params: 294912/67249922 (0.439%)
[EVAL] base_loss=0.6505 new_loss=0.6270 Δ=0.0235 | base_acc=0.762 new_acc=0.994
[CryoFlux] lora_delta → Δ=0.0235 | ok=True | receipt=8d0ae8aa…
```

---

## Energy Efficiency Metric

Define **η (eta)** as the energy efficiency of learning:

$$\eta = \frac{\Delta}{\text{joules spent}}$$

Example:
- Task A: 100J → Δ=0.01 → η=0.0001
- Task B: 80J → Δ=0.03 → η=0.000375 ✓ (more efficient)

The system optimizes for η by accepting only updates with Δ > threshold.

---

## Architecture Decisions

### Why LoRA?
- **Non-destructive**: Adapter discarded if rejected
- **Efficient**: Only 0.44% of parameters trainable
- **Fast**: Quick to train and evaluate
- **Reversible**: Failed updates don't corrupt base model

### Why frozen idle baseline?
- **Accurate**: Doesn't chase load, measures true net power
- **Stable**: Locked early, consistent throughout

### Why holdout evaluation?
- **Reproducible**: Fixed set = consistent Δ measurement
- **Unbiased**: Holdout never seen during training

---

## Limitations

- **CPU power estimation**: TDP-based model is approximate; GPU power (NVIDIA NVML) more reliable
- **Small improvements**: LoRA updates are incremental; many cycles needed for large Δ
- **Single-node**: Currently local; network verification in v0.2+
- **Data quality**: Noisy/imbalanced holdout → noisy Δ signal

---

## Future Work (v0.2+)

- ✅ η-based scheduler (prioritize high-efficiency tasks)
- ✅ Versioning & rollback (keep last-N merges)
- ✅ Dashboard (real-time energy/Δ/η tracking)
- ✅ P2P verification (multi-node network)
- ✅ Proof-of-Learning consensus mechanism

---

## Files

```
CryoFlux/
├── joule-agent-rs/          # Rust energy daemon
│   ├── src/main.rs
│   └── Cargo.toml
├── cryo-orchestrator/       # Python orchestrator
│   ├── cryo.py              # Main loop
│   ├── requirements.txt
│   └── .venv/
├── data/                    # Data (not tracked)
│   ├── holdout.csv
│   └── incoming/news.txt
├── WHITEPAPER.md            # Full technical details
├── README.md                # This file
└── .gitignore
```

---

## Testing

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

These generate load while the orchestrator accumulates joules and executes tasks.

---

## License

MIT

---

## Contributing

Questions or ideas? Open an issue or reach out.

**CryoFlux is a research prototype demonstrating the principle: "If you can measure it, you can improve it. If you can audit it, you can trust it."**

---

**Status**: ✅ v0.1 (local, functional) | 🔄 v0.2 (multi-node, in design)

# CryoFlux — Energy-Accounted Learning Research

**Energy → learning → evidence. Measure each link before calling it proof.**

CryoFlux is a research prototype for studying how a machine-learning update can be connected to energy measurements, evaluation evidence, and auditable receipts.

The long-term question is simple:

> Can we make a defensible statement of the form **“this learning attempt consumed X joules and produced Y independently measured improvement”**, with enough evidence for another party to verify it?

The current v0.1 repository implements important parts of that loop, but it does **not yet establish a verified `X joules -> Y learning gain` claim**.

Read [`docs/MEASUREMENT_STATUS.md`](docs/MEASUREMENT_STATUS.md) for the current claim boundary. It takes precedence over stronger language in the original [`WHITEPAPER.md`](WHITEPAPER.md), which documents the initial v0.1 thesis and design.

<img width="1646" height="898" alt="CryoFlux prototype" src="https://github.com/user-attachments/assets/44516fcd-6e11-444f-8beb-5698fdb8d221" />

---

## What exists today

CryoFlux contains four working prototype layers:

### JouleAgent — Rust energy sampler

`joule-agent-rs/`

- samples NVIDIA GPU power through NVML when available;
- estimates CPU power from utilization and configured TDP;
- maintains an idle estimate;
- integrates derived net power into a joule bucket;
- exposes `/v1/sample` and atomic `/v1/take` budget reservation.

The bucket is currently best understood as an **energy authorization budget**. The orchestrator records configured task reservations (`20 J`, `120 J`, etc.), not yet a task-bounded integration of the actual power trace.

### Orchestrator — Python learning loop

`cryo-orchestrator/`

- chooses between indexing and LoRA experiments;
- trains LoRA adapters;
- evaluates base vs candidate behavior;
- accepts/rejects candidates;
- merges accepted adapters;
- records task receipts in SQLite.

The present LoRA evaluation is a **development measurement**. Training and evaluation currently draw from the same configured data source and the default setup path can fall back to small embedded examples. It is not yet independent held-out evidence.

### Analysis layer

`analysis/`

- reads receipt history;
- computes task metrics;
- exports ledgers;
- plots local efficiency series;
- aggregates task statistics for the scheduler.

These tools are useful for inspecting prototype behavior, but historical `delta` values have different meanings across task types.

### Experimental scheduler

`cryo-orchestrator/scheduler.py`

- UCB-style selection;
- optional epsilon-greedy exploration;
- energy-feasibility filtering;
- uses historical task-local `eta = delta / joules` values.

The scheduler is experimental because current `delta` semantics are heterogeneous: LoRA uses loss reduction while Index uses `embeddings_added / 1000`. Those values do not yet define a common cross-task utility unit.

---

## The four measurement contracts

CryoFlux is now organized around four separate questions.

### 1. Energy truth

Can the receipt bind the **actual energy attributable to one task execution**, rather than a configured budget reservation?

Required future evidence includes task boundaries, power-source identity, actual integration intervals, baseline semantics, availability state, and uncertainty/coverage.

### 2. Learning truth

Can the candidate improvement be measured on data that was not used to train or select the update, with explicit parent-model lineage and replayable evaluation evidence?

A clean protocol needs immutable train/development/test roles, disjointness checks, model/data/config hashes, and signed deltas that preserve regressions as well as improvements.

### 3. Comparability truth

Can different task outcomes share a meaningful utility scale?

`loss reduction` and `embeddings added` are not naturally the same unit. Until a common utility contract exists, eta is safest as a **task-local** quantity. Global delta/eta across heterogeneous tasks is legacy descriptive output, not an intelligence unit.

### 4. Receipt truth

What exactly does a receipt prove?

The current SQLite row plus `delta_hash` is an auditable run record. It does not yet content-address the complete energy trace, model artifacts, datasets, environment, evaluation outputs, or previous receipt state, and SQLite does not technically enforce append-only storage.

A future Proof-of-Learning receipt needs a canonical evidence schema and an explicit threat model before adding consensus, blockchain, token, or notarization layers.

---

## Current research frontier

The most useful contributions are not feature expansion. They are attempts to close or falsify these measurement contracts:

1. **Task-attributed energy measurement** — measured run energy with calibrated/reference evidence.
2. **Independent learning evidence** — clean data roles, parent-model lineage, signed deltas, replay.
3. **Comparable utility / allocation** — decide how heterogeneous outcomes can or cannot be compared.
4. **Evidence-complete receipts** — canonical artifacts, hashes, lineage, replay and threat model.

Negative results are useful. A contribution showing that an energy estimator is too noisy, that a candidate delta does not generalize, or that two tasks should not share one scalar utility is a successful research outcome.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) before proposing a large experiment.

---

## Important v0.1 boundaries

### Energy accounting

The current orchestrator waits for sufficient bucket energy, calls `/v1/take` with the configured task estimate, then records that configured amount in the receipt. The learning run itself is not yet surrounded by a start/end power integration contract.

### Idle baseline

The current JouleAgent updates its CPU/GPU idle estimates through EMA when derived net power is below the configured idle-learning threshold. It should therefore be described as an **adaptive idle estimate**, not a permanently frozen baseline.

### Sample integration

The sampler currently integrates using the configured nominal sample period. A stronger measurement path should use observed elapsed time and preserve timing evidence.

### Development data

The Windows setup script creates placeholder data under the repository-root `data/` directory, while the default runner launches the orchestrator from `cryo-orchestrator/`. Relative data paths therefore require care. If no local holdout is found, the current code uses embedded fallback examples.

Do not use placeholder/fallback runs as held-out scientific evidence.

### Sequential LoRA state

The adapter training path initializes from the configured base model, while evaluation can load the currently merged model from `state/base_model`. A future continual-learning contract must explicitly bind every candidate to its actual parent model.

### Receipt hashing

Current hashes are useful identifiers, not complete cryptographic attestations of all evidence. New proof work should hash canonical artifact contents and define what an attacker is assumed able to change.

---

## Development quick start

### Windows

```powershell
.\scripts\dev_setup.ps1
.\scripts\dev_run.ps1
```

### Linux/macOS

```bash
chmod +x scripts/dev_run.sh
./scripts/dev_run.sh
```

These commands are appropriate for **prototype development and instrumentation testing**. They are not a qualifying scientific evaluation protocol.

Configuration lives in [`config.toml`](config.toml).

---

## Repository structure

```text
CryoFlux/
├── joule-agent-rs/          Rust energy sampler and budget API
├── cryo-orchestrator/       Learning tasks, receipts, scheduler
├── analysis/                Receipt/eta analysis and export tools
├── scripts/                 Local setup/run helpers
├── docs/
│   └── MEASUREMENT_STATUS.md
├── config.toml
├── WHITEPAPER.md            Original v0.1 design/thesis
├── CONTRIBUTING.md
└── README.md
```

---

## Safe current claim language

CryoFlux can currently be described as:

- an **energy-accounted learning research prototype**;
- a **budget-gated learning loop**;
- an experiment in **linking energy telemetry, model updates, evaluation and provenance**;
- a platform for studying **task-local efficiency and auditable run records**.

It should not yet be described as proving:

- exact task-attributed joules;
- independent held-out capability improvement;
- comparable intelligence gain across heterogeneous task types;
- tamper-proof or independently verifiable Proof of Learning;
- distributed Proof-of-Learning consensus.

Those are the research targets.

---

## License

Apache License 2.0.

## Contributing

Research-method contributions, stronger baselines, measurement audits, reproducibility work, and falsifications are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

# CryoFlux measurement status

This document describes what the current v0.1 implementation actually measures and what remains unproven.

It is the claim boundary for new research and contributor work. Where older README or whitepaper language implies a stronger guarantee, this document takes precedence until a newer measurement contract is implemented and validated.

## Current thesis

CryoFlux asks a useful systems question:

> Can a learning attempt be linked to measured energy, independently measured model improvement, and a receipt that allows another party to verify the claim?

The current repository implements pieces of that loop. It does **not yet establish a verified `X joules -> Y learning gain` claim**.

The remaining work separates into four contracts.

---

## 1. Energy truth

### What exists

`JouleAgent` samples:

- GPU power through NVIDIA NVML when available;
- CPU utilization converted to watts through a configurable TDP model;
- an idle estimate;
- a derived `net_w` value;
- a continuously integrated joule bucket.

The orchestrator uses that bucket as an **authorization budget**. A task becomes eligible when enough joules are present, and `/v1/take` atomically removes a configured amount before the task runs.

### What a v0.1 receipt currently means

For `lora_delta`, the orchestrator records the configured `task_lora_est_joules` value. For `index_refresh`, it records `task_index_est_joules`.

Those values are **budget reservations**, not an integration of the power trace over the task's actual execution interval.

Therefore the current receipt field named `joule` should not be interpreted as independently measured task energy.

### Additional measurement limitations

The current JouleAgent:

- estimates CPU watts from utilization times TDP rather than a package-energy counter;
- returns `0` GPU watts when NVML is unavailable instead of carrying an explicit unavailable state;
- updates the idle baseline through EMA while measured net power is below the idle-learning threshold, so the baseline is adaptive rather than strictly frozen;
- integrates with the nominal sample period rather than the observed elapsed interval between samples.

### Required boundary for a stronger claim

A future task-energy receipt should identify at least:

- measurement source and availability state;
- task start/end boundaries;
- raw or replayable power samples;
- actual integration interval semantics;
- baseline method and version;
- measured task-attributed joules plus uncertainty/coverage information;
- environment and device identity sufficient to interpret the measurement.

External wall-power measurement, RAPL/package counters, NVML, or another source may be used as calibration/reference evidence, but their semantics must remain explicit.

---

## 2. Learning truth

### What exists

`TaskLoRA` trains a LoRA adapter and compares base vs adapter loss/accuracy. The current acceptance rule is based on loss delta and/or accuracy gain.

This is a useful development loop, but it is not currently a clean held-out learning estimate.

### Current data boundary

Training and evaluation both draw from `holdout.csv`. The training code samples up to 256 rows from that source, while evaluation loads up to 512 rows from the same source. There is no enforced disjoint train/development/test manifest.

The Windows setup path also creates `data/holdout.csv` in the repository root, while the default runner launches the orchestrator with `cryo-orchestrator` as the working directory. Unless a separate `cryo-orchestrator/data/holdout.csv` is supplied, the orchestrator can fall back to embedded synthetic examples. The training and evaluation fallbacks contain the same small sentiment examples.

A large delta observed on this path is therefore development evidence, not proof of generalization.

### Sequential-update boundary

The current training path initializes adapters from the configured hub base model. Evaluation, however, attempts to load the current merged model from `state/base_model` before attaching the new adapter.

This means later adapters are not necessarily trained against the same base state on which their incremental delta is evaluated.

### Required boundary for a stronger claim

A future learning receipt should bind:

- immutable train/development/test identities;
- a disjointness check;
- base model artifact hash and parent receipt/model lineage;
- adapter artifact hash;
- tokenizer/model revisions;
- seed and training configuration;
- predeclared evaluation metrics;
- raw predictions or sufficient replay evidence;
- negative deltas as observed, rather than clipping regressions away for evidence storage.

Acceptance policy may still use a threshold, but the evidence record should preserve the signed observation.

---

## 3. Comparability truth

### Current problem

The repository currently uses one field, `delta`, for different task semantics:

- `lora_delta`: loss reduction;
- `index_refresh`: `embeddings_added / 1000`.

The analysis layer sums these values and computes a global `eta = delta / joules`. The scheduler then compares per-task `eta_avg` values directly in its UCB score.

These deltas do not share a unit or utility scale. A numeric comparison such as `index_refresh eta > lora_delta eta` therefore does not currently establish that indexing creates more learning value per joule.

### Required boundary for a stronger scheduler claim

Before cross-task optimization, CryoFlux needs either:

1. a common predeclared utility function with interpretable units; or
2. task-local efficiency metrics plus a higher-level allocation policy that does not treat incomparable deltas as the same quantity.

A useful contribution may also demonstrate that a single scalar utility is the wrong abstraction.

Until then:

- task-local eta may be analyzed within one task definition/version;
- global `delta_total` and global eta across heterogeneous tasks should be treated as descriptive legacy outputs, not a comparable intelligence metric;
- UCB selection based on cross-task raw eta is experimental.

---

## 4. Receipt truth

### What exists

The repository records task rows in SQLite and stores a `delta_hash`. This is useful run provenance.

### What is not yet proven

The current SQLite schema does not technically enforce append-only behavior. The current LoRA `delta_hash` is derived from timestamp text, adapter path, and delta value; it is not a content hash over the complete evidence package.

A current receipt does not bind all of:

- actual energy trace;
- adapter bytes;
- base model bytes/lineage;
- dataset manifests;
- config;
- environment;
- raw evaluation output;
- previous receipt/hash-chain state.

The JouleAgent sample hash similarly hashes timestamp and bucket state; it is not a signature or an attestation of the measurement source.

Therefore current receipts are best described as **auditable run records with hashes**, not independent cryptographic proofs of learning.

### Required boundary for a stronger proof

A future receipt format should define a canonical serialization and content-addressed evidence set. Depending on the intended threat model it may include:

- artifact SHA-256/BLAKE3 hashes;
- previous-receipt hash or Merkle structure;
- task/model/data/config/environment identities;
- measurement trace hash;
- evaluation evidence hash;
- signed attestation when an identity/trust model exists;
- model-free verifier/replay tooling;
- explicit states for missing/unavailable evidence.

The threat model must be written before adding blockchain, consensus, notarization, or token incentives.

---

## Current claim language

Safe current descriptions include:

- energy-accounted learning research prototype;
- budget-gated learning loop;
- experimental energy/learning receipts;
- task-local efficiency analysis;
- prototype for studying how energy, model updates, evaluation, and provenance can be linked.

Do not currently claim that CryoFlux proves:

- exact task-attributed joules;
- independent held-out capability gain;
- comparable intelligence gain across heterogeneous task types;
- tamper-proof or independently verifiable Proof of Learning;
- a valid decentralized consensus or economic value mechanism.

Those are research targets.

---

## Contribution frontier

The highest-value contribution areas are now:

1. **Task-attributed energy measurement** — define and validate a real measurement contract instead of recording configured reservations.
2. **Independent learning evidence** — enforce disjoint data roles, model lineage, signed deltas, and replayable evaluation.
3. **Comparable utility / allocation** — decide how heterogeneous task outcomes can or cannot be compared before optimizing a global eta.
4. **Evidence-complete receipts** — define canonical provenance, hashes, replay, and an explicit threat model.

Negative results are welcome. If a clean experiment shows that a proposed energy attribution method is too noisy, that a delta does not generalize, or that heterogeneous tasks should not share one utility scalar, that is useful evidence.

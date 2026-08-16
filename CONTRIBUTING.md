# Contributing to CryoFlux

CryoFlux welcomes contributions that make the link between **energy, learning, utility, and evidence** easier to measure, falsify, replay, or verify.

Before starting substantial work, read:

1. [`README.md`](README.md)
2. [`docs/MEASUREMENT_STATUS.md`](docs/MEASUREMENT_STATUS.md)
3. [`WHITEPAPER.md`](WHITEPAPER.md) as the original v0.1 design/thesis

The current repository is a research prototype. The phrase **Proof of Learning** is a research target, not a claim that v0.1 already provides independent cryptographic proof of task energy and generalizing capability gain.

## Current contribution areas

The highest-value work falls into four independent contracts.

### 1. Energy measurement

Useful contributions include:

- task-bounded energy traces;
- measured start/end integration rather than fixed budget accounting;
- NVML/RAPL/external-meter calibration;
- explicit measurement availability and uncertainty;
- baseline methods that are versioned and testable;
- deterministic integration/replay tests;
- experiments quantifying error from sample cadence, baseline drift, and unrelated system load.

Do not silently relabel a configured reservation as measured task energy.

### 2. Learning evidence

Useful contributions include:

- immutable train/development/test manifests;
- enforced split disjointness;
- model and adapter lineage;
- signed rather than clipped evaluation deltas in the evidence record;
- raw-prediction or model-free replay artifacts;
- repeated-seed / uncertainty analysis;
- tests for sequential updates against the correct parent model;
- protocols that distinguish development evidence from final held-out evidence.

Do not tune on an evaluation set and then describe that same set as independent holdout evidence.

### 3. Utility and scheduling

Useful contributions include:

- analysis of whether heterogeneous tasks can share one utility scale;
- task-local efficiency metrics;
- normalized or decision-theoretic utility definitions;
- multi-objective scheduling;
- cost-sensitive allocation methods that do not compare incompatible raw deltas;
- strong simple baselines before another learned/bandit policy.

A contribution may conclude that a single global eta is the wrong abstraction.

### 4. Receipts and verification

Useful contributions include:

- canonical receipt schemas;
- content-addressed evidence bundles;
- model/data/config/environment hashes;
- task-energy trace hashes;
- parent-model and previous-receipt lineage;
- model-free receipt verification/replay;
- append-only or tamper-evident storage designs;
- explicit threat models and attestation semantics.

Do not add blockchain, consensus, token, or notarization machinery before defining what evidence is being authenticated and against which attacker model.

## Research principles

### Measurement before optimization

Do not optimize eta, acceptance thresholds, scheduler behavior, or cryptographic packaging until the underlying quantity has a clear meaning.

### Preserve negative evidence

A regression, null result, measurement failure, unavailable counter, or non-comparable task is valid evidence. Do not coerce it into a positive scalar merely to keep the pipeline moving.

### Version semantic changes

If a contribution changes any of these, give the new experiment/receipt/metric a new explicit version:

- energy attribution semantics;
- baseline method;
- data roles/splits;
- delta definition;
- acceptance rule;
- utility scale;
- receipt canonicalization;
- threat model;
- hash/attestation scheme.

Historical receipts should remain interpretable under the semantics that created them.

### Separate authorization from measurement

The joule bucket is currently useful as a task authorization mechanism. A future measured-energy contract may use the same agent, but the two concepts must remain distinct in naming, schema, and tests.

### Separate task-local value from cross-task value

`loss reduction` and `embeddings added` are different quantities. Do not compare their numeric ratios as though they share a natural intelligence unit without an explicit utility model.

## Proposing a research change

For a substantial experiment, open an issue first and include:

- the measurement or scientific question;
- current failure mode;
- proposed contract/change;
- baseline or reference method;
- evidence/data class;
- what is development vs held-out;
- hardware/environment assumptions;
- primary metrics and uncertainty treatment;
- what outcome would count against the proposed method;
- whether historical receipts remain comparable.

Prefer a small experiment that resolves one ambiguity over a broad architecture expansion.

## Development workflow

The repository currently contains a Rust JouleAgent, Python orchestrator, and Python analysis tooling.

For Windows, the existing setup/run scripts remain useful for development:

```powershell
.\scripts\dev_setup.ps1
.\scripts\dev_run.ps1
```

However, do not treat placeholder/fallback datasets or default-development receipts as held-out scientific evidence.

When changing measurement or learning behavior, document:

- exact commands;
- current working directory;
- config path used;
- dataset paths and hashes where applicable;
- GPU/CPU/device information;
- whether NVML or another energy source was available;
- raw outputs needed to reproduce the result.

## Pull request expectations

A PR should state:

- which of the four contracts it affects;
- previous semantics;
- new semantics;
- tests/evidence added;
- known unmeasured conditions;
- whether any schema or metric meaning changed;
- whether old receipts/results are still comparable.

Keep one conceptual change per PR where practical.

## Claim discipline

Please distinguish between:

- implementation capability;
- development observation;
- calibrated measurement;
- held-out learning evidence;
- task-local efficiency;
- cross-task utility;
- auditable record;
- tamper-evident proof;
- signed/externally attestable proof.

Those are not interchangeable.

## Good first contributions

Good small contributions may include:

- deterministic tests for nominal-vs-observed integration time;
- explicit `unavailable` state when NVML is absent;
- receipt schema validation;
- split-disjointness checks;
- hash helpers over canonical files;
- documentation corrections that align a claim with the implemented semantics;
- model-free analysis of historical receipts.

## License

CryoFlux is licensed under Apache License 2.0. Unless explicitly stated otherwise, contributions intentionally submitted for inclusion in the project are provided under the same terms.

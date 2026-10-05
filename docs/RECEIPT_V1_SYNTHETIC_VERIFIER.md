# Synthetic receipt.v1 verifier

This first slice keeps the evidence boundary small and explicit:

- `cryo-orchestrator/receipt_v1.py` verifies canonical JSON, content-addressed artifact hashes, byte lengths, replayable evaluation deltas, and trapezoid energy integration.
- Results are `VALID`, `INVALID`, or `UNVERIFIABLE`; unavailable required evidence is never synthesized.
- `cryo-orchestrator/tests/test_receipt_v1.py` uses only synthetic JSON/JSONL fixtures and Python's standard library.

Run from the repository root:

```bash
python3 -m unittest discover -s cryo-orchestrator/tests -v
```

This authenticates bytes and declared relationships. It does not prove honest hardware, honest measurement, honest execution, or a Proof-of-Learning claim. Existing v0.1 SQLite receipts are untouched and remain legacy run records.

The implementation intentionally leaves signatures, external witnessing, model loading, scheduler behavior, and historical receipt migration outside this slice.

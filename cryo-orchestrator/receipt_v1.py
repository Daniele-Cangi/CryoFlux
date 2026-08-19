"""Model-free verification for synthetic cryoflux.receipt.v1 bundles.

This module authenticates bytes and relationships only.  It does not attest to
honest hardware, honest execution, or scientific sufficiency of a claim.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

VALID = "VALID"
INVALID = "INVALID"
UNVERIFIABLE = "UNVERIFIABLE"


def canonical_json(value: Any) -> str:
    """Return the deterministic JSON representation used by receipt v1."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def receipt_digest(receipt: dict[str, Any]) -> str:
    """Hash a receipt without its self-referential canonical_hash field."""
    unsigned = {k: v for k, v in receipt.items() if k != "canonical_hash"}
    return sha256_bytes(canonical_json(unsigned).encode("utf-8"))


def verify_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Verify a v1 bundle and return ``status`` plus explicit diagnostics.

    ``bundle.artifacts`` maps manifest names to their byte content represented as
    UTF-8 strings in this synthetic, dependency-free fixture format.  A missing
    required artifact is UNVERIFIABLE; a present artifact with wrong bytes is
    INVALID.  Claim sufficiency is intentionally reported separately.
    """
    errors: list[str] = []
    warnings: list[str] = []
    receipt = bundle.get("receipt")
    if not isinstance(receipt, dict) or receipt.get("schema") != "cryoflux.receipt.v1":
        return {"status": INVALID, "errors": ["missing or unsupported receipt schema"], "warnings": []}

    supplied = receipt.get("canonical_hash")
    if not isinstance(supplied, str) or supplied != receipt_digest(receipt):
        errors.append("canonical_hash does not match canonical receipt bytes")

    artifacts = bundle.get("artifacts", {})
    if not isinstance(artifacts, dict):
        errors.append("artifacts must be an object")
        artifacts = {}
    manifest = receipt.get("evidence", [])
    if not isinstance(manifest, list):
        errors.append("evidence manifest must be an array")
        manifest = []
    for entry in manifest:
        name = entry.get("name") if isinstance(entry, dict) else None
        if not name or not isinstance(entry, dict):
            errors.append("evidence manifest contains malformed entry")
            continue
        if name not in artifacts:
            if entry.get("required", True):
                warnings.append(f"required evidence missing: {name}")
            continue
        raw = artifacts[name]
        if not isinstance(raw, str):
            errors.append(f"artifact is not UTF-8 fixture text: {name}")
            continue
        data = raw.encode("utf-8")
        if entry.get("size") != len(data) or entry.get("sha256") != sha256_bytes(data):
            errors.append(f"artifact hash or size mismatch: {name}")

    previous = receipt.get("previous_receipt_hash")
    if previous is not None and (not isinstance(previous, str) or len(previous) != 64):
        errors.append("previous_receipt_hash is not a SHA-256 digest")

    observation = receipt.get("observation", {})
    if not isinstance(observation, dict):
        errors.append("observation must be an object")
    elif observation.get("energy_status") in {"unavailable", "not_yet_qualified"}:
        warnings.append("energy evidence is unavailable; no energy claim is established")
    elif "energy_joules" not in observation:
        warnings.append("energy evidence is absent")

    claim = receipt.get("claim", {})
    claim_sufficient = bool(manifest) and not any("required evidence missing" in w for w in warnings)
    if not claim_sufficient:
        warnings.append("bundle integrity does not establish claim sufficiency")

    status = INVALID if errors else (UNVERIFIABLE if warnings else VALID)
    return {"status": status, "errors": errors, "warnings": warnings, "claim_sufficient": claim_sufficient, "claim": claim}

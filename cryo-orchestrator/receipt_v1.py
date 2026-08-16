"""Model-free verification for synthetic CryoFlux receipt.v1 bundles.

This module deliberately authenticates bytes and declared relationships only. It does
not claim honest hardware, honest execution, or Proof-of-Learning.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "cryoflux.receipt.v1"
CANONICALIZATION = {"algorithm": "sorted-json", "version": "1"}


def canonical_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes for receipt values."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _result(status: str, errors: list[str], warnings: list[str], checked: list[str], receipt_hash: str | None) -> dict[str, Any]:
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "checked_artifacts": checked,
        "receipt_hash": receipt_hash,
    }


def verify_bundle(bundle: str | Path) -> dict[str, Any]:
    """Verify one synthetic bundle and return the stable result contract."""
    root = Path(bundle)
    errors: list[str] = []
    warnings: list[str] = []
    checked: list[str] = []
    receipt_path = root / "receipt.json"
    if not receipt_path.is_file():
        return _result("UNVERIFIABLE", ["missing receipt.json"], [], [], None)

    try:
        receipt = load_json(receipt_path)
    except (OSError, json.JSONDecodeError) as exc:
        return _result("INVALID", [f"receipt parse error: {exc}"], [], [], None)
    receipt_hash = sha256_bytes(canonical_bytes(receipt))

    if receipt.get("schema") != SCHEMA:
        errors.append("unsupported receipt schema")
    if receipt.get("canonicalization") != CANONICALIZATION:
        errors.append("unsupported canonicalization")

    manifest_path = root / "manifest.json"
    try:
        manifest = load_json(manifest_path)
    except FileNotFoundError:
        return _result("UNVERIFIABLE", errors + ["missing manifest.json"], warnings, checked, receipt_hash)
    except (OSError, json.JSONDecodeError) as exc:
        return _result("INVALID", errors + [f"manifest parse error: {exc}"], warnings, checked, receipt_hash)

    entries = {item.get("artifact_id"): item for item in manifest.get("artifacts", [])}
    refs = receipt.get("artifacts", {})
    required_missing = False
    for name, artifact_id in refs.items():
        entry = entries.get(artifact_id)
        if entry is None:
            errors.append(f"manifest missing receipt artifact: {name}")
            continue
        availability = entry.get("availability", "present")
        if availability != "present":
            if availability == "unavailable":
                required_missing = True
            warnings.append(f"artifact unavailable: {name}")
            continue
        rel = Path(entry.get("path", ""))
        if rel.is_absolute() or ".." in rel.parts:
            errors.append(f"unsafe artifact path: {name}")
            continue
        artifact_path = root / rel
        if not artifact_path.is_file():
            required_missing = True
            warnings.append(f"artifact file missing: {name}")
            continue
        raw = artifact_path.read_bytes()
        checked.append(artifact_id)
        if len(raw) != entry.get("byte_length"):
            errors.append(f"byte length mismatch: {name}")
        actual = sha256_bytes(raw)
        if actual != entry.get("sha256") or actual != artifact_id:
            errors.append(f"hash mismatch: {name}")

    if receipt.get("energy", {}).get("status") == "unavailable":
        warnings.append("energy evidence unavailable")
        required_missing = True
    else:
        energy_id = refs.get("energy_trace")
        energy_entry = entries.get(energy_id)
        if energy_entry:
            try:
                points = [json.loads(line) for line in (root / energy_entry["path"]).read_text(encoding="utf-8").splitlines() if line]
                joules = 0.0
                for left, right in zip(points, points[1:]):
                    dt = (right["mono_ns"] - left["mono_ns"]) / 1_000_000_000
                    if dt <= 0:
                        errors.append("energy timestamps are not strictly increasing")
                        break
                    joules += ((left["watts"] + right["watts"]) / 2) * dt
                declared = receipt["energy"].get("joules")
                if declared is None or abs(joules - declared) > 1e-9:
                    errors.append("energy integration mismatch")
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"energy replay error: {exc}")

    evaluation_id = refs.get("evaluation_raw")
    evaluation_entry = entries.get(evaluation_id)
    if evaluation_entry and evaluation_entry.get("availability") == "present":
        try:
            rows = [json.loads(line) for line in (root / evaluation_entry["path"]).read_text(encoding="utf-8").splitlines() if line]
            if rows:
                observed = rows[0]["candidate"] - rows[0]["baseline"]
                declared = receipt["metrics"]["delta"]["value"]
                if abs(observed - declared) > 1e-12:
                    errors.append("metric mismatch")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"metric replay error: {exc}")

    if errors:
        status = "INVALID"
    elif required_missing:
        status = "UNVERIFIABLE"
    else:
        status = "VALID"
    return _result(status, errors, warnings, checked, receipt_hash)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    args = parser.parse_args()
    print(json.dumps(verify_bundle(args.bundle), indent=2, sort_keys=True))

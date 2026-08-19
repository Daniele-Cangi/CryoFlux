#!/usr/bin/env python3
"""Dependency-free adversarial fixtures for receipt_v1."""
import copy
import hashlib
import importlib.util
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("receipt_v1", ROOT / "cryo-orchestrator" / "receipt_v1.py")
receipt_v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(receipt_v1)


def bundle():
    raw = "metric=0.25\n"
    entry = {"name": "evaluation/raw.txt", "sha256": hashlib.sha256(raw.encode()).hexdigest(), "size": len(raw.encode()), "required": True}
    receipt = {
        "schema": "cryoflux.receipt.v1",
        "receipt_id": "r-001",
        "task_id": "synthetic-task",
        "previous_receipt_hash": None,
        "evidence": [entry],
        "observation": {"delta": -0.25, "energy_status": "unavailable"},
        "claim": {"accepted": True, "policy_version": "synthetic-v1"},
    }
    receipt["canonical_hash"] = receipt_v1.receipt_digest(receipt)
    return {"receipt": receipt, "artifacts": {entry["name"]: raw}}


class ReceiptV1Tests(unittest.TestCase):
    def test_valid_bundle_preserves_signed_negative_delta(self):
        result = receipt_v1.verify_bundle(bundle())
        self.assertEqual(result["status"], receipt_v1.UNVERIFIABLE)  # energy is explicitly unavailable
        self.assertEqual(result["claim"]["accepted"], True)
        self.assertIn("energy evidence is unavailable", " ".join(result["warnings"]))

    def test_metric_tampering_invalidates(self):
        value = bundle(); value["artifacts"]["evaluation/raw.txt"] = "metric=0.99\n"
        self.assertEqual(receipt_v1.verify_bundle(value)["status"], receipt_v1.INVALID)

    def test_artifact_substitution_invalidates(self):
        value = bundle(); value["artifacts"]["evaluation/raw.txt"] = "other-run\n"
        self.assertTrue(receipt_v1.verify_bundle(value)["errors"])

    def test_missing_required_evidence_is_unverifiable(self):
        value = bundle(); value["artifacts"] = {}
        result = receipt_v1.verify_bundle(value)
        self.assertEqual(result["status"], receipt_v1.UNVERIFIABLE)
        self.assertIn("required evidence missing", " ".join(result["warnings"]))

    def test_canonicalization_change_invalidates(self):
        value = bundle(); value["receipt"]["task_id"] = "different"
        self.assertEqual(receipt_v1.verify_bundle(value)["status"], receipt_v1.INVALID)

    def test_previous_hash_shape_is_checked(self):
        value = bundle(); value["receipt"]["previous_receipt_hash"] = "not-a-hash"
        value["receipt"]["canonical_hash"] = receipt_v1.receipt_digest(value["receipt"])
        self.assertEqual(receipt_v1.verify_bundle(value)["status"], receipt_v1.INVALID)

    def test_reordering_manifest_does_not_change_artifact_identity(self):
        value = bundle(); extra = "config=true\n"
        value["artifacts"]["config.txt"] = extra
        value["receipt"]["evidence"].append({"name": "config.txt", "sha256": hashlib.sha256(extra.encode()).hexdigest(), "size": len(extra.encode()), "required": False})
        value["receipt"]["canonical_hash"] = receipt_v1.receipt_digest(value["receipt"])
        self.assertEqual(receipt_v1.verify_bundle(value)["status"], receipt_v1.UNVERIFIABLE)
        value["receipt"]["evidence"] = list(reversed(value["receipt"]["evidence"]))
        value["receipt"]["canonical_hash"] = receipt_v1.receipt_digest(value["receipt"])
        self.assertEqual(receipt_v1.verify_bundle(value)["status"], receipt_v1.UNVERIFIABLE)

    def test_json_nan_is_rejected_by_canonicalizer(self):
        with self.assertRaises(ValueError):
            receipt_v1.canonical_json({"delta": float("nan")})


if __name__ == "__main__":
    unittest.main(verbosity=2)

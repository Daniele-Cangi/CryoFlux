import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from receipt_v1 import canonical_bytes, sha256_bytes, verify_bundle


class ReceiptV1Tests(unittest.TestCase):
    def make_bundle(self, *, missing_energy=False):
        root = Path(tempfile.mkdtemp())
        evidence = root / "evidence"
        evidence.mkdir()
        files = {
            "evaluation_raw": ("evaluation.jsonl", b'{"baseline":0.2,"candidate":0.15}\n', "application/jsonl"),
            "energy_trace": ("energy.jsonl", b'{"mono_ns":0,"watts":10}\n{"mono_ns":1000000000,"watts":20}\n', "application/jsonl"),
        }
        entries = []
        refs = {}
        for name, (filename, raw, media_type) in files.items():
            artifact_id = sha256_bytes(raw)
            path = evidence / filename
            path.write_bytes(raw)
            entries.append({"artifact_id": artifact_id, "path": f"evidence/{filename}", "media_type": media_type,
                            "byte_length": len(raw), "sha256": artifact_id, "availability": "present"})
            refs[name] = artifact_id
        if missing_energy:
            energy_id = refs["energy_trace"]
            entries[-1]["availability"] = "unavailable"
            (evidence / "energy.jsonl").unlink()
        receipt = {
            "schema": "cryoflux.receipt.v1", "run_id": "run-demo-1",
            "task": {"name": "synthetic", "definition_version": "1"},
            "parent": {"receipt_hash": None, "model_hash": "sha256:" + "0" * 64},
            "artifacts": refs,
            "metrics": {"delta": {"name": "candidate_minus_baseline", "value": -0.05, "unit": "loss", "status": "observed"}},
            "energy": {"contract_version": "cryoflux.energy.v1", "joules": None if missing_energy else 15.0,
                       "status": "unavailable" if missing_energy else "observed"},
            "canonicalization": {"algorithm": "sorted-json", "version": "1"},
        }
        (root / "receipt.json").write_bytes(canonical_bytes(receipt))
        (root / "manifest.json").write_bytes(canonical_bytes({"schema": "cryoflux.manifest.v1", "artifacts": entries}))
        return root

    def test_valid_bundle(self):
        result = verify_bundle(self.make_bundle())
        self.assertEqual(result["status"], "VALID", result)
        self.assertEqual(result["errors"], [])

    def test_metric_tamper_is_invalid(self):
        root = self.make_bundle()
        receipt = json.loads((root / "receipt.json").read_text())
        receipt["metrics"]["delta"]["value"] = 0.5
        (root / "receipt.json").write_bytes(canonical_bytes(receipt))
        result = verify_bundle(root)
        self.assertEqual(result["status"], "INVALID")
        self.assertIn("metric mismatch", result["errors"])

    def test_artifact_replacement_is_invalid(self):
        root = self.make_bundle()
        (root / "evidence" / "energy.jsonl").write_text('{"mono_ns":0,"watts":99}\n{"mono_ns":1000000000,"watts":99}\n')
        result = verify_bundle(root)
        self.assertEqual(result["status"], "INVALID")
        self.assertIn("hash mismatch: energy_trace", result["errors"])

    def test_missing_required_evidence_is_unverifiable(self):
        result = verify_bundle(self.make_bundle(missing_energy=True))
        self.assertEqual(result["status"], "UNVERIFIABLE", result)

    def test_key_reordering_does_not_change_receipt_hash(self):
        root = self.make_bundle()
        receipt = json.loads((root / "receipt.json").read_text())
        reordered = {key: receipt[key] for key in reversed(list(receipt))}
        (root / "receipt.json").write_text(json.dumps(reordered, indent=2))
        result = verify_bundle(root)
        self.assertEqual(result["status"], "VALID", result)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""
CryoFlux Ledger Summary

Standalone analysis tool that works on exported JSONL ledger files.
Calculates global and per-task metrics, top merges, and accept reasons distribution.

Usage:
    python analysis/summarize_ledger.py
    python analysis/summarize_ledger.py --path out/ledger_all.jsonl
    python analysis/summarize_ledger.py --no-header
    python analysis/summarize_ledger.py --max-records 100
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class Record:
    """Single ledger record."""
    receipt_id: int
    timestamp: str
    task_name: str
    joules: float
    delta: float
    accepted: bool
    eta: float
    kwh_eff: float
    cost_eur: float
    co2_g: float
    delta_hash: str
    meta_raw: str


@dataclass
class TaskMetrics:
    """Aggregated metrics for a task."""
    task_name: str
    count: int
    joules_total: float
    delta_total: float
    kwh_eff_total: float
    cost_eur_total: float
    co2_g_total: float
    eta_avg: float


def parse_record(line: str, line_num: int, no_header: bool) -> Optional[Record]:
    """
    Parse a single JSONL record.

    Args:
        line: JSON line string
        line_num: Line number for error reporting
        no_header: If True, suppress error messages

    Returns:
        Record or None if parsing failed
    """
    try:
        data = json.loads(line)

        # Extract fields with defaults
        return Record(
            receipt_id=data.get("receipt_id", 0),
            timestamp=data.get("timestamp", ""),
            task_name=data.get("task_name", "unknown"),
            joules=float(data.get("joules", 0.0)),
            delta=float(data.get("delta", 0.0)),
            accepted=bool(data.get("accepted", False)),
            eta=float(data.get("eta", 0.0)),
            kwh_eff=float(data.get("kwh_eff", 0.0)),
            cost_eur=float(data.get("cost_eur", 0.0)),
            co2_g=float(data.get("co2_g", 0.0)),
            delta_hash=data.get("delta_hash", ""),
            meta_raw=data.get("meta_raw", "{}")
        )
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        if not no_header:
            print(f"[WARN] Skipping malformed record at line {line_num}: {e}", file=sys.stderr)
        return None


def extract_accept_reason(meta_raw: str) -> str:
    """
    Extract accept_reason from meta_raw JSON string.

    Args:
        meta_raw: JSON string containing metadata

    Returns:
        Accept reason string or "unknown"
    """
    try:
        meta = json.loads(meta_raw)
        return meta.get("accept_reason", "unknown")
    except (json.JSONDecodeError, KeyError):
        return "unknown"


def load_records(path: Path, max_records: Optional[int], no_header: bool) -> List[Record]:
    """
    Load records from JSONL file.

    Args:
        path: Path to JSONL file
        max_records: Maximum number of records to load (None = all)
        no_header: If True, suppress progress messages

    Returns:
        List of Record objects
    """
    if not path.exists():
        if not no_header:
            print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)

    records = []
    line_num = 0

    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line_num += 1
                line = line.strip()

                if not line:
                    continue

                record = parse_record(line, line_num, no_header)
                if record:
                    records.append(record)

                    if max_records and len(records) >= max_records:
                        break
    except IOError as e:
        if not no_header:
            print(f"[ERROR] Could not read file: {e}", file=sys.stderr)
        sys.exit(1)

    return records


def compute_global_metrics(records: List[Record]) -> Dict[str, Any]:
    """
    Compute global aggregate metrics.

    Args:
        records: List of Record objects

    Returns:
        Dictionary of global metrics
    """
    total_records = len(records)
    accepted_count = sum(1 for r in records if r.accepted)
    rejected_count = total_records - accepted_count

    total_joules = sum(r.joules for r in records)
    total_delta = sum(r.delta for r in records)
    total_kwh_eff = sum(r.kwh_eff for r in records)
    total_cost_eur = sum(r.cost_eur for r in records)
    total_co2_g = sum(r.co2_g for r in records)

    global_eta = (total_delta / total_joules) if total_joules > 0 else 0.0

    return {
        "total_records": total_records,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "total_joules": total_joules,
        "total_delta": total_delta,
        "global_eta": global_eta,
        "total_kwh_eff": total_kwh_eff,
        "total_cost_eur": total_cost_eur,
        "total_co2_g": total_co2_g,
    }


def compute_task_metrics(records: List[Record]) -> List[TaskMetrics]:
    """
    Compute per-task aggregate metrics.

    Args:
        records: List of Record objects

    Returns:
        List of TaskMetrics objects
    """
    task_data = defaultdict(lambda: {
        "count": 0,
        "joules_total": 0.0,
        "delta_total": 0.0,
        "kwh_eff_total": 0.0,
        "cost_eur_total": 0.0,
        "co2_g_total": 0.0,
    })

    for record in records:
        task = task_data[record.task_name]
        task["count"] += 1
        task["joules_total"] += record.joules
        task["delta_total"] += record.delta
        task["kwh_eff_total"] += record.kwh_eff
        task["cost_eur_total"] += record.cost_eur
        task["co2_g_total"] += record.co2_g

    # Convert to TaskMetrics objects
    task_metrics = []
    for task_name, data in task_data.items():
        eta_avg = (data["delta_total"] / data["joules_total"]) if data["joules_total"] > 0 else 0.0

        task_metrics.append(TaskMetrics(
            task_name=task_name,
            count=data["count"],
            joules_total=data["joules_total"],
            delta_total=data["delta_total"],
            kwh_eff_total=data["kwh_eff_total"],
            cost_eur_total=data["cost_eur_total"],
            co2_g_total=data["co2_g_total"],
            eta_avg=eta_avg
        ))

    # Sort by task_name
    task_metrics.sort(key=lambda x: x.task_name)

    return task_metrics


def get_top_records(records: List[Record], key: str, n: int = 3) -> List[Record]:
    """
    Get top N records by specified key.

    Args:
        records: List of Record objects
        key: Attribute name to sort by ("delta" or "eta")
        n: Number of top records to return

    Returns:
        List of top N Record objects
    """
    if key == "delta":
        sorted_records = sorted(records, key=lambda r: r.delta, reverse=True)
    elif key == "eta":
        sorted_records = sorted(records, key=lambda r: r.eta, reverse=True)
    else:
        return []

    return sorted_records[:n]


def compute_accept_reasons(records: List[Record]) -> Dict[str, int]:
    """
    Count accept reasons distribution.

    Args:
        records: List of Record objects

    Returns:
        Dictionary mapping accept_reason to count
    """
    reasons = defaultdict(int)

    for record in records:
        reason = extract_accept_reason(record.meta_raw)
        reasons[reason] += 1

    # Sort by count descending
    return dict(sorted(reasons.items(), key=lambda x: x[1], reverse=True))


def print_summary(records: List[Record], no_header: bool):
    """
    Print formatted summary report.

    Args:
        records: List of Record objects
        no_header: If True, suppress human-readable formatting
    """
    # Compute metrics
    global_metrics = compute_global_metrics(records)
    task_metrics = compute_task_metrics(records)
    top_delta = get_top_records(records, "delta", 3)
    top_eta = get_top_records(records, "eta", 3)
    accept_reasons = compute_accept_reasons(records)

    if no_header:
        # Machine-readable format (JSON)
        output = {
            "global": global_metrics,
            "tasks": [
                {
                    "task_name": tm.task_name,
                    "count": tm.count,
                    "joules_total": tm.joules_total,
                    "delta_total": tm.delta_total,
                    "eta_avg": tm.eta_avg,
                    "kwh_eff_total": tm.kwh_eff_total,
                    "cost_eur_total": tm.cost_eur_total,
                    "co2_g_total": tm.co2_g_total,
                }
                for tm in task_metrics
            ],
            "top_delta": [
                {
                    "receipt_id": r.receipt_id,
                    "timestamp": r.timestamp,
                    "task_name": r.task_name,
                    "delta": r.delta,
                    "eta": r.eta,
                }
                for r in top_delta
            ],
            "top_eta": [
                {
                    "receipt_id": r.receipt_id,
                    "timestamp": r.timestamp,
                    "task_name": r.task_name,
                    "delta": r.delta,
                    "eta": r.eta,
                }
                for r in top_eta
            ],
            "accept_reasons": accept_reasons,
        }
        print(json.dumps(output, indent=2))
    else:
        # Human-readable format
        print("=" * 80)
        print("CryoFlux Ledger Summary")
        print("=" * 80)
        print()

        # Global metrics
        print("[GLOBAL]")
        print(f"  Total records:         {global_metrics['total_records']:,}")
        print(f"  Accepted:              {global_metrics['accepted_count']:,}")
        print(f"  Rejected:              {global_metrics['rejected_count']:,}")
        print(f"  Total joules:          {global_metrics['total_joules']:,.2f} J")
        print(f"  Total delta:           {global_metrics['total_delta']:.6f}")
        print(f"  Global eta:            {global_metrics['global_eta']:.9f}")
        print(f"  Total kWh (eff):       {global_metrics['total_kwh_eff']:.6f} kWh")
        print(f"  Total cost:            EUR {global_metrics['total_cost_eur']:.4f}")
        print(f"  Total emissions:       {global_metrics['total_co2_g']:.2f} gCO2")
        print()

        # Per-task breakdown
        print("[BY TASK]")
        header = (
            f"{'Task':<20} {'Count':>6} {'Joules':>12} {'Delta':>10} {'eta':>12} "
            f"{'kWh_eff':>10} {'Cost (EUR)':>12} {'CO2 (g)':>10}"
        )
        print(header)
        print("-" * len(header))

        for tm in task_metrics:
            print(
                f"{tm.task_name:<20} "
                f"{tm.count:>6} "
                f"{tm.joules_total:>12,.2f} "
                f"{tm.delta_total:>10.6f} "
                f"{tm.eta_avg:>12.9f} "
                f"{tm.kwh_eff_total:>10.6f} "
                f"{tm.cost_eur_total:>12.4f} "
                f"{tm.co2_g_total:>10.2f}"
            )
        print()

        # Top 3 by delta
        print("[TOP 3 BY DELTA]")
        for i, r in enumerate(top_delta, 1):
            print(
                f"  {i}. receipt_id={r.receipt_id:<6} task={r.task_name:<15} "
                f"delta={r.delta:.6f}  eta={r.eta:.9f}  ts={r.timestamp}"
            )
        print()

        # Top 3 by eta
        print("[TOP 3 BY ETA]")
        for i, r in enumerate(top_eta, 1):
            print(
                f"  {i}. receipt_id={r.receipt_id:<6} task={r.task_name:<15} "
                f"eta={r.eta:.9f}  delta={r.delta:.6f}  ts={r.timestamp}"
            )
        print()

        # Accept reasons
        print("[ACCEPT REASONS]")
        for reason, count in accept_reasons.items():
            pct = (count / global_metrics['total_records'] * 100) if global_metrics['total_records'] > 0 else 0.0
            print(f"  {reason:<30} {count:>6}  ({pct:>5.1f}%)")
        print()

        print("=" * 80)
        print("Summary complete")
        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze CryoFlux JSONL ledger files (standalone, no database access)"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="out/ledger_accepted.jsonl",
        help="Path to JSONL ledger file (default: out/ledger_accepted.jsonl)"
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Output machine-readable JSON (suppress human-readable formatting)"
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limit analysis to first N records (default: all)"
    )

    args = parser.parse_args()

    try:
        # Load records
        path = Path(args.path)
        if not args.no_header:
            print(f"[INFO] Loading records from {path}", file=sys.stderr)

        records = load_records(path, args.max_records, args.no_header)

        if not records:
            if not args.no_header:
                print("[WARN] No valid records found in file", file=sys.stderr)
            sys.exit(0)

        if not args.no_header:
            print(f"[INFO] Loaded {len(records)} records", file=sys.stderr)
            print()

        # Print summary
        print_summary(records, args.no_header)

    except KeyboardInterrupt:
        if not args.no_header:
            print("\n[INFO] Analysis interrupted by user", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        if not args.no_header:
            print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

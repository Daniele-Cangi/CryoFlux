#!/usr/bin/env python3
"""
CryoFlux Ledger Export

Exports receipts as structured JSON lines (NDJSON) including energy, cost, CO2,
delta, and acceptance info. Creates a "learning & energy ledger" for external use.

Usage:
    python analysis/export_ledger.py
    python analysis/export_ledger.py --accepted-only
    python analysis/export_ledger.py --output ledger.jsonl

Each line is a JSON object with:
    - receipt_id, timestamp, task_name
    - joules, delta, accepted, eta
    - kwh_eff, cost_eur, co2_g (from cost model)
    - delta_hash, meta_raw
"""

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Import open_db from metrics module
try:
    from analysis.metrics import open_db
except ImportError:
    # Try relative import if running from analysis directory
    try:
        from metrics import open_db
    except ImportError:
        print("[ERROR] Could not import analysis.metrics. Run from repo root.")
        sys.exit(1)


# Constants
JOULES_PER_KWH = 3_600_000.0


@dataclass
class CostConfig:
    """Cost model configuration."""
    enabled: bool = True
    kwh_price_eur: float = 0.30
    pue: float = 1.20
    carbon_intensity_g_per_kwh: float = 300.0


def load_cost_config() -> Optional[CostConfig]:
    """
    Load cost model configuration from config.toml.

    Returns:
        CostConfig or None if not found/disabled
    """
    config_paths = [
        Path("config.toml"),
        Path("../config.toml"),
        Path("../../config.toml"),
    ]

    for path in config_paths:
        if path.exists():
            try:
                import toml
                data = toml.load(path)

                if "cost_model" not in data:
                    print("[WARN] No [cost_model] section in config.toml")
                    return None

                cm = data["cost_model"]
                config = CostConfig()

                if "enabled" in cm:
                    config.enabled = cm["enabled"]
                if "kwh_price_eur" in cm:
                    config.kwh_price_eur = cm["kwh_price_eur"]
                if "pue" in cm:
                    config.pue = cm["pue"]
                if "carbon_intensity_g_per_kwh" in cm:
                    config.carbon_intensity_g_per_kwh = cm["carbon_intensity_g_per_kwh"]

                return config

            except ImportError:
                print("[ERROR] 'toml' package not found. Install with: pip install toml")
                return None
            except Exception as e:
                print(f"[WARN] Could not parse config.toml: {e}")
                return None

    print("[WARN] config.toml not found")
    return None


def ensure_receipts_canonical_view(conn: sqlite3.Connection):
    """
    Ensure receipts_canonical view exists.

    Args:
        conn: Database connection
    """
    cursor = conn.cursor()

    # Check if view exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='view' AND name='receipts_canonical'
    """)

    if cursor.fetchone():
        return  # View already exists

    # Create view
    try:
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS receipts_canonical AS
            SELECT
                id,
                datetime(ts, 'unixepoch') AS timestamp,
                ts AS ts_unix,
                task AS task_name,
                joule AS joules_spent,
                sec AS execution_time_sec,
                delta,
                loss,
                delta_hash,
                meta,
                CASE
                    WHEN json_extract(meta, '$.accepted') = 1 THEN 1
                    WHEN json_extract(meta, '$.accepted') = 'true' THEN 1
                    ELSE 0
                END AS accepted
            FROM receipts
        """)
        conn.commit()
        print("[INFO] Created receipts_canonical view")
    except sqlite3.OperationalError as e:
        print(f"[WARN] Could not create receipts_canonical view: {e}")


def compute_cost_metrics(joules: float, config: CostConfig) -> tuple:
    """
    Convert joules to kWh, cost, and CO2 emissions.

    Args:
        joules: Energy in joules
        config: Cost model configuration

    Returns:
        (kwh_eff, cost_eur, co2_g)
    """
    if joules <= 0:
        return (0.0, 0.0, 0.0)

    kwh = joules / JOULES_PER_KWH
    kwh_eff = kwh * config.pue
    cost_eur = kwh_eff * config.kwh_price_eur
    co2_g = kwh_eff * config.carbon_intensity_g_per_kwh

    return (kwh_eff, cost_eur, co2_g)


def export_ledger(accepted_only: bool, output_path: Optional[str]):
    """
    Export receipts as NDJSON ledger.

    Args:
        accepted_only: If True, only export accepted receipts
        output_path: Output file path (None = stdout)
    """
    print("=" * 60)
    print("CryoFlux Ledger Export")
    print("=" * 60)
    print()

    # Load cost config
    config = load_cost_config()
    if not config:
        print("[ERROR] Could not load cost model configuration")
        sys.exit(1)

    if not config.enabled:
        print("[INFO] Cost model is disabled (cost_model.enabled = false)")
        sys.exit(0)

    # Open database
    try:
        conn = open_db()
        print("[INFO] Connected to receipts database")
    except Exception as e:
        print(f"[ERROR] Could not connect to database: {e}")
        sys.exit(1)

    # Ensure receipts_canonical view exists
    ensure_receipts_canonical_view(conn)

    # Build query
    query = """
        SELECT
            id, timestamp, task_name, joules_spent, delta,
            accepted, delta_hash, meta
        FROM receipts_canonical
    """

    if accepted_only:
        query += " WHERE accepted = 1"

    query += " ORDER BY id ASC"

    # Execute query
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"[ERROR] Could not query receipts_canonical: {e}")
        conn.close()
        sys.exit(1)

    if not rows:
        print("[WARN] No receipts found in database")
        conn.close()
        sys.exit(0)

    # Open output destination
    if output_path:
        try:
            output_file = open(output_path, 'w', encoding='utf-8')
            print(f"[INFO] Exporting to {output_path}")
        except IOError as e:
            print(f"[ERROR] Could not open output file: {e}")
            conn.close()
            sys.exit(1)
    else:
        output_file = sys.stdout
        print("[INFO] Exporting to stdout")

    print()

    # Export records
    total_count = 0
    accepted_count = 0
    rejected_count = 0

    for row in rows:
        receipt_id = row[0]
        timestamp = row[1]
        task_name = row[2]
        joules_spent = row[3] or 0.0
        delta = row[4] or 0.0
        accepted = bool(row[5])
        delta_hash = row[6] or ""
        meta_raw = row[7] or "{}"

        # Compute cost metrics
        kwh_eff, cost_eur, co2_g = compute_cost_metrics(joules_spent, config)

        # Compute eta
        eta = (delta / joules_spent) if joules_spent > 0 else 0.0

        # Build record
        record = {
            "receipt_id": receipt_id,
            "timestamp": timestamp,
            "task_name": task_name,
            "joules": joules_spent,
            "delta": delta,
            "accepted": accepted,
            "eta": eta,
            "kwh_eff": kwh_eff,
            "cost_eur": cost_eur,
            "co2_g": co2_g,
            "delta_hash": delta_hash,
            "meta_raw": meta_raw
        }

        # Write as JSON line
        json_line = json.dumps(record, ensure_ascii=False)
        output_file.write(json_line + "\n")

        # Update counts
        total_count += 1
        if accepted:
            accepted_count += 1
        else:
            rejected_count += 1

    # Close output file if not stdout
    if output_path:
        output_file.close()

    # Print summary
    print()
    print("=" * 60)
    destination = output_path if output_path else "stdout"
    print(f"[INFO] Exported {total_count} receipts ({accepted_count} accepted, {rejected_count} rejected) to {destination}")
    print("=" * 60)

    conn.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export CryoFlux receipts as NDJSON ledger with energy, cost, and CO2 metrics"
    )
    parser.add_argument(
        "--accepted-only",
        action="store_true",
        help="Export only accepted receipts"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout)"
    )

    args = parser.parse_args()

    try:
        export_ledger(args.accepted_only, args.output)
    except KeyboardInterrupt:
        print("\n[INFO] Export interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CryoFlux Lab - Task Statistics Aggregator

This script reads all receipts from the database and computes per-task
aggregated statistics, storing them in the task_stats table.

Safe to re-run multiple times; it recomputes aggregates from scratch.

Usage:
    python analysis/update_task_stats.py
    python -m analysis.update_task_stats
"""

import sys
import sqlite3
from pathlib import Path
from datetime import datetime

try:
    from analysis.metrics import load_config, open_db
except ImportError:
    # Support running directly from analysis/ directory
    from metrics import load_config, open_db


def create_task_stats_table(conn):
    """Create task_stats table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS task_stats (
            task_name TEXT PRIMARY KEY,
            runs INTEGER NOT NULL,
            joules_total REAL NOT NULL,
            delta_total REAL NOT NULL,
            eta_avg REAL NOT NULL,
            accepted_runs INTEGER NOT NULL,
            last_run_at TEXT NOT NULL
        )
    """)
    conn.commit()
    print("[update_task_stats] task_stats table ready")


def update_aggregates(conn):
    """Compute per-task aggregates and upsert into task_stats."""
    cursor = conn.cursor()

    # Compute aggregates from receipts
    cursor.execute("""
        SELECT
            task AS task_name,
            COUNT(*) AS runs,
            SUM(joule) AS joules_total,
            SUM(delta) AS delta_total,
            SUM(CASE
                WHEN json_extract(meta, '$.accepted') = 1 THEN 1
                WHEN json_extract(meta, '$.accepted') = 'true' THEN 1
                ELSE 0
            END) AS accepted_runs,
            MAX(ts) AS last_run_ts
        FROM receipts
        WHERE task IS NOT NULL
        GROUP BY task
    """)

    rows = cursor.fetchall()

    if not rows:
        print("[update_task_stats] No receipts found in database")
        return

    # Upsert aggregates into task_stats
    for row in rows:
        task_name, runs, joules_total, delta_total, accepted_runs, last_run_ts = row

        # Compute eta_avg
        eta_avg = delta_total / joules_total if joules_total > 0 else 0.0

        # Convert last_run_ts to ISO8601
        last_run_at = datetime.fromtimestamp(last_run_ts).isoformat()

        # Upsert
        cursor.execute("""
            INSERT OR REPLACE INTO task_stats
            (task_name, runs, joules_total, delta_total, eta_avg, accepted_runs, last_run_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (task_name, runs, joules_total, delta_total, eta_avg, accepted_runs, last_run_at))

        print(f"[update_task_stats] Updated {task_name}: runs={runs}, η_avg={eta_avg:.6f}")

    conn.commit()
    print(f"[update_task_stats] Successfully updated {len(rows)} task(s)")


def main():
    """Main entry point."""
    print("=" * 60)
    print("CryoFlux Lab - Task Statistics Aggregator")
    print("=" * 60)

    # Load configuration
    cfg = load_config()
    
    # Open database (resolves path automatically & applies views)
    try:
        conn = open_db(cfg)
        # Note: open_db logs the view creation steps
    except Exception as e:
        print(f"[ERROR] Could not connect to database: {e}")
        # Try to print helpful hint if it might be path related, 
        # though resolve_db_path should have handled fallsback warnings.
        sys.exit(1)

    try:
        # Create task_stats table if needed
        create_task_stats_table(conn)

        # Update aggregates
        update_aggregates(conn)

        print("\n" + "=" * 60)
        print("Task statistics updated successfully")
        print("=" * 60)

    except Exception as e:
        print(f"[ERROR] Failed to update task stats: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()


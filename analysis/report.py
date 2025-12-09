#!/usr/bin/env python3
"""
CryoFlux Lab - Efficiency Report CLI

Generate structured reports on energy-to-learning efficiency (η = Δ/J).

Usage:
    python analysis/report.py
    python analysis/report.py --task lora_delta
    python analysis/report.py --db ./state/receipts.db --window 20
    python -m analysis.report --task index_refresh
"""

import argparse
import sys
from pathlib import Path

try:
    from analysis.metrics import (
        load_config,
        open_db,
        compute_global_metrics,
        compute_task_metrics,
        load_eta_series,
        compute_rolling_eta,
        resolve_db_path,
    )
except ImportError:
    # Support running from analysis/ directory
    from metrics import (
        load_config,
        open_db,
        compute_global_metrics,
        compute_task_metrics,
        load_eta_series,
        compute_rolling_eta,
        resolve_db_path,
    )


def print_global_report(global_metrics):
    """Print global metrics section."""
    gm = global_metrics
    
    print("\n[GLOBAL METRICS]")
    print(f"  Total receipts:        {gm.total_receipts}")
    print(f"  Accepted:              {gm.accepted} ({gm.accept_rate*100:.1f}%)")
    print(f"  Rejected:              {gm.rejected} ({(1-gm.accept_rate)*100:.1f}%)")
    print(f"  Total joules spent:    {gm.joules_total:.2f} J")
    print(f"  Total Δ:               {gm.delta_total:.4f}")
    print(f"  Mean η (Δ/J):          {gm.eta_mean:.6f}")


def print_task_report(task_metrics_list):
    """Print per-task metrics table."""
    if not task_metrics_list:
        print("\n[BY TASK]")
        print("  No task data available")
        return

    print("\n[BY TASK]")

    # Table header
    header = f"  {'Task':<20} {'Runs':<8} {'J_total':<12} {'Δ_total':<12} {'η_avg':<14} {'Accept%':<10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Table rows
    for tm in task_metrics_list:
        task_short = tm.task_name[:19] if len(tm.task_name) > 19 else tm.task_name
        print(f"  {task_short:<20} {tm.runs:<8} {tm.joules_total:<12.1f} "
              f"{tm.delta_total:<12.4f} {tm.eta_avg:<14.6f} {tm.accept_rate*100:<10.1f}")


def print_rolling_eta_note(conn, task_name, window):
    """Print note about rolling η over recent receipts."""
    samples = load_eta_series(conn, task_name=task_name)

    if len(samples) < window:
        print(f"\n[ROLLING η (window={window})]")
        print(f"  Insufficient data: only {len(samples)} receipt(s) available")
        return

    # Get most recent window
    recent_samples = samples[-window:]
    avg_eta = sum(s.eta for s in recent_samples) / len(recent_samples)

    print(f"\n[ROLLING η (window={window})]")
    if task_name:
        print(f"  Task: {task_name}")
    print(f"  Recent {window} receipts:")
    print(f"    Average η: {avg_eta:.6f}")
    print(f"    Receipt IDs: {recent_samples[0].receipt_id} - {recent_samples[-1].receipt_id}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CryoFlux Lab - Efficiency Report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db",
        type=str,
        help="Path to receipts database (overrides config.toml)",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Focus on a single task (e.g., 'lora_delta')",
    )
    parser.add_argument(
        "--window",
        type=int,
        help="Compute rolling η over last N receipts",
    )

    args = parser.parse_args()

    # Print header
    print("=" * 60)
    print("CryoFlux Lab – Efficiency Report")
    print("=" * 60)

    # Load configuration
    if args.db:
        db_path = args.db
        print(f"\n[INFO] Using database: {db_path}")
    else:
        cfg = load_config()
        # Resolve path here for reporting and checking
        db_path = resolve_db_path(cfg.receipts_db)
        print(f"\n[INFO] Using database: {db_path}")

    # Check if database exists
    if not Path(db_path).exists():
        print(f"\n[ERROR] Receipts database not found at {db_path}")
        print("[INFO] Run the orchestrator first to generate receipts")
        sys.exit(1)

    # Open database
    try:
        conn = open_db(db_path=db_path)
    except Exception as e:
        print(f"\n[ERROR] Could not connect to database: {e}")
        sys.exit(1)

    try:
        # Compute metrics
        global_metrics = compute_global_metrics(conn)
        task_metrics_list = compute_task_metrics(conn)

        # Filter by task if requested
        if args.task:
            task_metrics_list = [tm for tm in task_metrics_list if tm.task_name == args.task]
            if not task_metrics_list:
                print(f"\n[ERROR] No receipts found for task: {args.task}")
                sys.exit(1)

        # Print reports
        if args.task:
            print(f"\n[FOCUS] Task: {args.task}")
            print(f"  Global context: {global_metrics.total_receipts} total receipts")

        print_global_report(global_metrics)
        print_task_report(task_metrics_list)

        # Rolling eta if requested
        if args.window:
            print_rolling_eta_note(conn, args.task, args.window)

        # Footer
        print("\n" + "=" * 60)
        print("Report complete")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Failed to generate report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CryoFlux Lab - η Visualization Tool

Generate plots of energy-to-learning efficiency (η = Δ/J) over time.

Usage:
    python analysis/plot_eta.py
    python analysis/plot_eta.py --task lora_delta
    python analysis/plot_eta.py --window 10 --output ./eta_plot.png
    python -m analysis.plot_eta
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

try:
    from analysis.metrics import (
        load_config,
        open_db,
        load_eta_series,
        compute_rolling_eta,
        resolve_db_path,
    )
except ImportError:
    # Support running from analysis/ directory
    from metrics import (
        load_config,
        open_db,
        load_eta_series,
        compute_rolling_eta,
        resolve_db_path,
    )

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_eta_series(samples, task_name=None, window=None, output_path="./state/eta_plot.png"):
    """
    Generate η time series plot with optional rolling average.

    Args:
        samples: List of EtaSample
        task_name: Optional task name for title
        window: Optional rolling window size
        output_path: Where to save the plot
    """
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib not installed. Install with: pip install matplotlib")
        sys.exit(1)

    if not samples:
        print("[ERROR] No data to plot")
        sys.exit(1)

    # Extract data
    timestamps = [datetime.fromtimestamp(s.timestamp) for s in samples]
    etas = [s.eta for s in samples]
    accepted = [s.accepted for s in samples]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot accepted/rejected with different colors
    accepted_times = [t for t, a in zip(timestamps, accepted) if a]
    accepted_etas = [e for e, a in zip(etas, accepted) if a]
    rejected_times = [t for t, a in zip(timestamps, accepted) if not a]
    rejected_etas = [e for e, a in zip(etas, accepted) if not a]

    if accepted_times:
        ax.scatter(accepted_times, accepted_etas, c='green', alpha=0.6, s=50,
                   label='Accepted', marker='o', edgecolors='darkgreen')
    if rejected_times:
        ax.scatter(rejected_times, rejected_etas, c='red', alpha=0.6, s=50,
                   label='Rejected', marker='x', linewidths=2)

    # Plot rolling average if requested
    if window and len(samples) >= window:
        rolling = compute_rolling_eta(samples, window)
        rolling_times = [datetime.fromtimestamp(samples[i].timestamp)
                        for i in range(len(samples)) if i >= window - 1]
        rolling_etas = [eta for _, eta in rolling]

        ax.plot(rolling_times, rolling_etas, 'b-', linewidth=2, alpha=0.7,
                label=f'Rolling avg (window={window})')

    # Formatting
    title = "CryoFlux Lab: η (Energy-to-Learning Efficiency) Over Time"
    if task_name:
        title += f" - {task_name}"
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.set_xlabel("Timestamp", fontsize=12)
    ax.set_ylabel("η (Δ / Joule)", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45, ha='right')

    # Tight layout
    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[plot_eta] Plot saved to: {output_path.absolute()}")

    # Print stats
    print(f"\n[PLOT STATS]")
    print(f"  Total samples:     {len(samples)}")
    print(f"  Accepted:          {sum(accepted)} ({sum(accepted)/len(samples)*100:.1f}%)")
    print(f"  Rejected:          {len(samples) - sum(accepted)}")
    print(f"  Mean η:            {sum(etas)/len(etas):.6f}")
    print(f"  Min η:             {min(etas):.6f}")
    print(f"  Max η:             {max(etas):.6f}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CryoFlux Lab - η Visualization Tool",
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
        default=10,
        help="Rolling average window size (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: ./state/eta_plot.png or ./state/eta_TASK.png)",
    )

    args = parser.parse_args()

    # Print header
    print("=" * 60)
    print("CryoFlux Lab – η Visualization")
    print("=" * 60)

    # Check matplotlib
    if not HAS_MATPLOTLIB:
        print("\n[ERROR] matplotlib not installed")
        print("[INFO] Install with: pip install matplotlib")
        sys.exit(1)

    # Load configuration
    if args.db:
        db_path = args.db
        print(f"\n[INFO] Using database: {db_path}")
    else:
        cfg = load_config()
        # Resolve path here for reporting and checking
        db_path = resolve_db_path(cfg.receipts_db)
        print(f"\n[INFO] Using database: {db_path} (from config.toml)")

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
        # Load eta series
        print(f"\n[INFO] Loading η time series...")
        if args.task:
            print(f"[INFO] Filtering by task: {args.task}")

        samples = load_eta_series(conn, task_name=args.task)

        if not samples:
            print(f"\n[ERROR] No receipts found")
            if args.task:
                print(f"[INFO] No data for task: {args.task}")
            sys.exit(1)

        print(f"[INFO] Loaded {len(samples)} receipt(s)")

        # Determine output path
        if args.output:
            output_path = args.output
        elif args.task:
            output_path = f"./state/eta_{args.task}.png"
        else:
            output_path = "./state/eta_global.png"

        # Generate plot
        print(f"\n[INFO] Generating plot...")
        plot_eta_series(samples, task_name=args.task, window=args.window, output_path=output_path)

        # Footer
        print("\n" + "=" * 60)
        print("Visualization complete")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Failed to generate plot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

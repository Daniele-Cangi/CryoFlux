#!/usr/bin/env python3
"""
CryoFlux Cost & Impact Report

Computes energy costs and environmental impact from receipts database.
Expresses energy usage in kWh, EUR, and CO₂ (globally and per task).

Usage:
    python analysis/cost_report.py

Configuration:
    Reads [cost_model] section from config.toml:
    - kwh_price_eur: Electricity price (EUR per kWh)
    - pue: Power Usage Effectiveness (datacenter overhead multiplier)
    - carbon_intensity_g_per_kwh: Carbon intensity (gCO2 per kWh)
"""

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

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


@dataclass
class TaskCostMetrics:
    """Cost and impact metrics for a single task."""
    task_name: str
    runs: int
    joules_total: float
    delta_total: float
    eta_avg: float
    kwh_eff: float
    cost_eur: float
    co2_g: float


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


def compute_cost_metrics(joules: float, config: CostConfig) -> Tuple[float, float, float]:
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


def load_task_metrics(conn: sqlite3.Connection) -> List[Tuple[str, int, float, float, float]]:
    """
    Load per-task metrics from database.

    Tries task_stats first, falls back to aggregating receipts_canonical.

    Args:
        conn: Database connection

    Returns:
        List of (task_name, runs, joules_total, delta_total, eta_avg)
    """
    cursor = conn.cursor()

    # Try task_stats first
    try:
        cursor.execute("""
            SELECT task_name, runs, joules_total, delta_total, eta_avg
            FROM task_stats
        """)
        rows = cursor.fetchall()

        if rows:
            return [(row[0], row[1], row[2], row[3], row[4]) for row in rows]
        else:
            print("[INFO] task_stats is empty, falling back to receipts_canonical")

    except sqlite3.OperationalError:
        print("[INFO] task_stats table not found, falling back to receipts_canonical")

    # Fallback: aggregate from receipts_canonical
    try:
        cursor.execute("""
            SELECT
                task_name,
                COUNT(*) AS runs,
                SUM(joules_spent) AS joules_total,
                SUM(delta) AS delta_total
            FROM receipts_canonical
            GROUP BY task_name
        """)
        rows = cursor.fetchall()

        if not rows:
            return []

        # Compute eta_avg manually
        result = []
        for row in rows:
            task_name = row[0]
            runs = row[1]
            joules_total = row[2] or 0.0
            delta_total = row[3] or 0.0
            eta_avg = (delta_total / joules_total) if joules_total > 0 else 0.0
            result.append((task_name, runs, joules_total, delta_total, eta_avg))

        return result

    except sqlite3.OperationalError as e:
        print(f"[ERROR] Could not query receipts_canonical: {e}")
        return []


def generate_report():
    """Generate and print cost & impact report."""

    print("=" * 60)
    print("CryoFlux Cost & Impact Report")
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

    # Print config
    print("[CONFIG]")
    print(f"  kWh price (EUR):       {config.kwh_price_eur:.4f}")
    print(f"  PUE:                   {config.pue:.2f}")
    print(f"  Carbon intensity:      {config.carbon_intensity_g_per_kwh:.1f} gCO2/kWh")
    print()

    # Open database
    try:
        conn = open_db()
        print(f"[INFO] Connected to receipts database")
        print()
    except Exception as e:
        print(f"[ERROR] Could not connect to database: {e}")
        sys.exit(1)

    # Load task metrics
    task_data = load_task_metrics(conn)

    if not task_data:
        print("[WARN] No task metrics available. Run the orchestrator to generate receipts.")
        conn.close()
        sys.exit(0)

    # Compute per-task cost metrics
    task_metrics: List[TaskCostMetrics] = []
    global_joules = 0.0
    global_delta = 0.0

    for task_name, runs, joules_total, delta_total, eta_avg in task_data:
        kwh_eff, cost_eur, co2_g = compute_cost_metrics(joules_total, config)

        task_metrics.append(TaskCostMetrics(
            task_name=task_name,
            runs=runs,
            joules_total=joules_total,
            delta_total=delta_total,
            eta_avg=eta_avg,
            kwh_eff=kwh_eff,
            cost_eur=cost_eur,
            co2_g=co2_g
        ))

        global_joules += joules_total
        global_delta += delta_total

    # Compute global cost metrics
    global_kwh_eff, global_cost_eur, global_co2_g = compute_cost_metrics(global_joules, config)
    global_eta = (global_delta / global_joules) if global_joules > 0 else 0.0

    # Print global metrics
    print("[GLOBAL]")
    print(f"  Total joules:          {global_joules:,.2f} J")
    print(f"  Total kWh (eff):       {global_kwh_eff:.6f} kWh")
    print(f"  Total cost:            EUR {global_cost_eur:.4f}")
    print(f"  Total emissions:       {global_co2_g:.2f} gCO2")
    print(f"  Total Delta:           {global_delta:.6f}")
    print(f"  Global eta (Delta/J):  {global_eta:.9f}")
    print()

    # Print per-task breakdown
    print("[BY TASK]")

    # Header
    header = (
        f"{'Task':<20} {'Runs':>6} {'Joules':>12} {'kWh_eff':>10} "
        f"{'Cost (EUR)':>12} {'CO2 (g)':>10} {'Delta':>10} {'eta':>12}"
    )
    print(header)
    print("-" * len(header))

    # Task rows
    for tm in task_metrics:
        print(
            f"{tm.task_name:<20} "
            f"{tm.runs:>6} "
            f"{tm.joules_total:>12,.2f} "
            f"{tm.kwh_eff:>10.6f} "
            f"{tm.cost_eur:>12.4f} "
            f"{tm.co2_g:>10.2f} "
            f"{tm.delta_total:>10.6f} "
            f"{tm.eta_avg:>12.9f}"
        )

    print()
    print("=" * 60)
    print("Report complete")
    print("=" * 60)

    conn.close()


if __name__ == "__main__":
    try:
        generate_report()
    except KeyboardInterrupt:
        print("\n[INFO] Report interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

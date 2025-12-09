#!/usr/bin/env python3
"""
CryoFlux Lab - Core Metrics Module

Provides dataclasses and functions for computing energy-to-learning
efficiency metrics from the receipts database.

Usage:
    from analysis.metrics import compute_global_metrics, compute_task_metrics
    conn = open_db(cfg)
    global_metrics = compute_global_metrics(conn)
    task_metrics = compute_task_metrics(conn)
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple


# Reuse config structure from orchestrator
# (simplified version to avoid circular imports)
@dataclass
class Cfg:
    """Minimal configuration for analysis tools."""
    receipts_db: str = "./state/receipts.db"


def load_config() -> Cfg:
    """Load configuration from config.toml if available."""
    cfg = Cfg()

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
                if "orchestrator" in data and "receipts_db" in data["orchestrator"]:
                    cfg.receipts_db = data["orchestrator"]["receipts_db"]
                break
            except (ImportError, Exception):
                pass

    return cfg


def resolve_db_path(configured_path: str) -> str:
    """
    Resolve database path with fallback logic.
    
    Priority:
    1. configured_path (if exists)
    2. ./cryo-orchestrator/state/receipts.db (if exists)
    3. configured_path (default)
    """
    path_obj = Path(configured_path)
    if path_obj.exists():
        return configured_path

    # Fallback check
    # Assuming running from repo root, fallback is typically inside cryo-orchestrator
    fallback_path = Path("cryo-orchestrator") / "state" / "receipts.db"
    
    if fallback_path.exists():
        print(f"[WARN] receipts.db not found at {configured_path}, using fallback {fallback_path}")
        return str(fallback_path)

    return configured_path


def ensure_views(conn: sqlite3.Connection):
    """
    Apply views from sql/views.sql to the database.
    Idempotent (CREATE VIEW IF NOT EXISTS).
    """
    try:
        # Locate views.sql relative to this module
        # analysis/metrics.py -> analysis/sql/views.sql
        base_dir = Path(__file__).parent
        sql_path = base_dir / "sql" / "views.sql"
        
        if not sql_path.exists():
            # Try falling back to current dir if __file__ resolution fails or different cwd
            sql_path = Path("analysis/sql/views.sql")
            
        if sql_path.exists():
            with open(sql_path, "r", encoding="utf-8") as f:
                sql_script = f.read()
                conn.executescript(sql_script)
                print("[INFO] Ensured receipts_canonical view exists")
        else:
            print(f"[WARN] Could not find views.sql at {sql_path}, skipping view creation")
            
    except Exception as e:
        print(f"[WARN] Failed to apply views: {e}")


def open_db(cfg: Optional[Cfg] = None, db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Open SQLite connection to receipts database.
    Resolves path and applies views automatically.

    Args:
        cfg: Configuration object with receipts_db path
        db_path: Direct path override

    Returns:
        sqlite3.Connection
    """
    if db_path:
        path = db_path
    elif cfg:
        path = resolve_db_path(cfg.receipts_db)
    else:
        path = resolve_db_path("./state/receipts.db")

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    
    # Ensure views exist
    ensure_views(conn)
    
    return conn


@dataclass
class GlobalMetrics:
    """Global aggregate metrics across all receipts."""
    total_receipts: int
    accepted: int
    rejected: int
    joules_total: float
    delta_total: float
    eta_mean: float  # delta_total / joules_total

    @property
    def accept_rate(self) -> float:
        """Acceptance rate as a fraction [0, 1]."""
        return self.accepted / self.total_receipts if self.total_receipts > 0 else 0.0


@dataclass
class TaskMetrics:
    """Per-task aggregate metrics."""
    task_name: str
    runs: int
    joules_total: float
    delta_total: float
    eta_avg: float
    accepted_runs: int

    @property
    def accept_rate(self) -> float:
        """Acceptance rate as a fraction [0, 1]."""
        return self.accepted_runs / self.runs if self.runs > 0 else 0.0


def compute_global_metrics(conn: sqlite3.Connection) -> GlobalMetrics:
    """
    Compute global aggregate metrics from all receipts.

    Args:
        conn: SQLite connection to receipts database

    Returns:
        GlobalMetrics dataclass
    """
    cursor = conn.cursor()

    # Query global aggregates
    cursor.execute("""
        SELECT
            COUNT(*) AS total_receipts,
            SUM(joule) AS joules_total,
            SUM(delta) AS delta_total,
            SUM(CASE
                WHEN json_extract(meta, '$.accepted') = 1 THEN 1
                WHEN json_extract(meta, '$.accepted') = 'true' THEN 1
                ELSE 0
            END) AS accepted
        FROM receipts
    """)

    row = cursor.fetchone()

    total_receipts = row["total_receipts"] or 0
    accepted = row["accepted"] or 0
    rejected = total_receipts - accepted
    joules_total = row["joules_total"] or 0.0
    delta_total = row["delta_total"] or 0.0
    eta_mean = delta_total / joules_total if joules_total > 0 else 0.0

    return GlobalMetrics(
        total_receipts=total_receipts,
        accepted=accepted,
        rejected=rejected,
        joules_total=joules_total,
        delta_total=delta_total,
        eta_mean=eta_mean,
    )


def compute_task_metrics(conn: sqlite3.Connection) -> List[TaskMetrics]:
    """
    Compute per-task aggregate metrics.

    Args:
        conn: SQLite connection to receipts database

    Returns:
        List of TaskMetrics dataclasses, sorted by task_name
    """
    cursor = conn.cursor()

    # Query per-task aggregates
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
            END) AS accepted_runs
        FROM receipts
        WHERE task IS NOT NULL
        GROUP BY task
        ORDER BY task
    """)

    results = []
    for row in cursor.fetchall():
        joules_total = row["joules_total"] or 0.0
        delta_total = row["delta_total"] or 0.0
        eta_avg = delta_total / joules_total if joules_total > 0 else 0.0

        results.append(TaskMetrics(
            task_name=row["task_name"],
            runs=row["runs"],
            joules_total=joules_total,
            delta_total=delta_total,
            eta_avg=eta_avg,
            accepted_runs=row["accepted_runs"] or 0,
        ))

    return results


@dataclass
class EtaSample:
    """Single η measurement with metadata."""
    receipt_id: int
    timestamp: float
    task_name: str
    joules: float
    delta: float
    eta: float
    accepted: bool


def load_eta_series(
    conn: sqlite3.Connection,
    task_name: Optional[str] = None,
    window: Optional[int] = None,
) -> List[EtaSample]:
    """
    Load time series of η measurements from receipts.

    Args:
        conn: SQLite connection
        task_name: Filter by task name (None = all tasks)
        window: If specified, return only last N receipts

    Returns:
        List of EtaSample, ordered by timestamp ascending
    """
    cursor = conn.cursor()

    # Build query
    query = """
        SELECT
            id,
            ts,
            task,
            joule,
            delta,
            meta
        FROM receipts
        WHERE joule > 0
    """

    params = []
    if task_name:
        query += " AND task = ?"
        params.append(task_name)

    query += " ORDER BY ts ASC"

    if window:
        query += " LIMIT ?"
        params.append(window)

    cursor.execute(query, params)

    results = []
    for row in cursor.fetchall():
        joules = row["joule"]
        delta = row["delta"]
        eta = delta / joules if joules > 0 else 0.0

        # Extract accepted flag from meta JSON
        try:
            meta = json.loads(row["meta"])
            accepted = meta.get("accepted", False)
        except (json.JSONDecodeError, TypeError):
            accepted = False

        results.append(EtaSample(
            receipt_id=row["id"],
            timestamp=row["ts"],
            task_name=row["task"],
            joules=joules,
            delta=delta,
            eta=eta,
            accepted=accepted,
        ))

    return results


def compute_rolling_eta(
    samples: List[EtaSample],
    window_size: int = 10,
) -> List[Tuple[int, float]]:
    """
    Compute rolling average of η over a sliding window.

    Args:
        samples: List of EtaSample ordered by timestamp
        window_size: Size of rolling window

    Returns:
        List of (receipt_id, rolling_eta) tuples
    """
    if len(samples) < window_size:
        return []

    results = []
    for i in range(window_size - 1, len(samples)):
        window = samples[i - window_size + 1:i + 1]
        avg_eta = sum(s.eta for s in window) / window_size
        results.append((samples[i].receipt_id, avg_eta))

    return results

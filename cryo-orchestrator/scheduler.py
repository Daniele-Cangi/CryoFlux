#!/usr/bin/env python3
"""
CryoFlux η-Aware Scheduler

Implements bandit-style task selection based on energy-to-learning efficiency (η).
Uses UCB (Upper Confidence Bound) formula to balance exploitation vs exploration.

This module is read-only on the database - it never modifies receipts or task_stats.
"""

import math
import random
import sqlite3
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class TaskEstimate:
    """Efficiency estimate for a single task from task_stats."""
    name: str
    est_joules: float       # Energy cost from config
    runs: int               # Total runs from task_stats
    joules_total: float     # Total energy spent from task_stats
    delta_total: float      # Total Δ from task_stats
    eta_avg: float          # Average η = delta_total / joules_total
    accepted_runs: int      # Number of accepted runs


@dataclass
class TaskChoice:
    """Result of task selection."""
    name: str
    reason: str             # "WARMUP", "BANDIT", or "FALLBACK"
    score: Optional[float]  # UCB score (None for warmup/fallback)


def load_task_stats(conn: sqlite3.Connection) -> Dict[str, TaskEstimate]:
    """
    Read task_stats table and return a dict keyed by task_name.

    Returns empty dict if table doesn't exist or has no data.

    Args:
        conn: SQLite connection to receipts database

    Returns:
        Dict mapping task_name -> TaskEstimate (without est_joules set yet)
    """
    try:
        cursor = conn.cursor()

        # Check if task_stats table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='task_stats'
        """)

        if not cursor.fetchone():
            print("[SCHEDULER][WARN] task_stats table not found")
            return {}

        # Load task stats
        cursor.execute("""
            SELECT
                task_name,
                runs,
                joules_total,
                delta_total,
                eta_avg,
                accepted_runs
            FROM task_stats
        """)

        results = {}
        for row in cursor.fetchall():
            task_name = row[0]
            results[task_name] = TaskEstimate(
                name=task_name,
                est_joules=0.0,  # Will be filled by caller from config
                runs=row[1],
                joules_total=row[2],
                delta_total=row[3],
                eta_avg=row[4],
                accepted_runs=row[5],
            )

        return results

    except sqlite3.Error as e:
        print(f"[SCHEDULER][WARN] Failed to load task_stats: {e}")
        return {}


def compute_scores(
    task_stats: Dict[str, TaskEstimate],
    ucb_c: float
) -> Dict[str, float]:
    """
    Compute UCB-style scores for each task.

    Formula: score_i = η_i + c * sqrt(2 * ln(N) / n_i)

    Where:
        η_i = eta_avg for task i
        n_i = runs for task i
        N = sum of runs over all tasks
        c = exploration constant

    Args:
        task_stats: Dict of TaskEstimate by task name
        ucb_c: Exploration constant

    Returns:
        Dict mapping task_name -> UCB score
    """
    if not task_stats:
        return {}

    # Calculate total runs across all tasks
    total_runs = sum(est.runs for est in task_stats.values())

    if total_runs == 0:
        # No runs yet - return zero scores (warmup will handle this)
        return {name: 0.0 for name in task_stats.keys()}

    scores = {}
    for task_name, est in task_stats.items():
        if est.runs == 0:
            # Infinite UCB for unexplored tasks (handled by warmup)
            scores[task_name] = float('inf')
        else:
            # UCB formula
            exploitation = est.eta_avg
            exploration = ucb_c * math.sqrt(2 * math.log(total_runs) / est.runs)
            scores[task_name] = exploitation + exploration

    return scores


def choose_task(
    cfg,
    bucket_j: float,
    conn: sqlite3.Connection
) -> Optional[TaskChoice]:
    """
    Select a task using η-aware bandit logic.

    Returns None to signal fallback to legacy choose(j) logic.

    Algorithm:
    1. Check if bandit is enabled in config
    2. Load task_stats from database
    3. Filter by energy feasibility (bucket_j >= est_joules * min_factor)
    4. Warmup phase: if any task has runs < warmup_runs, select it
    5. Otherwise: compute UCB scores and select highest

    Args:
        cfg: Configuration object with scheduler settings
        bucket_j: Available energy in joules
        conn: SQLite connection to receipts database

    Returns:
        TaskChoice if scheduler makes a selection, None for fallback
    """

    # Check if scheduler is enabled
    if not hasattr(cfg, 'scheduler') or not cfg.scheduler.bandit_enabled:
        print("[SCHEDULER] bandit disabled - falling back to legacy choose(j)")
        return None

    # Load task statistics from database
    try:
        task_stats = load_task_stats(conn)
    except Exception as e:
        print(f"[SCHEDULER][WARN] Failed to load task stats: {e}")
        return None

    if not task_stats:
        print("[SCHEDULER] no task_stats available - using legacy choose(j)")
        return None

    # Map task names to energy costs from config
    energy_map = {
        "index_refresh": cfg.energy.task_index_est_joules,
        "lora_delta": cfg.energy.task_lora_est_joules,
    }

    # Update TaskEstimate objects with energy costs
    for task_name, est in task_stats.items():
        if task_name in energy_map:
            est.est_joules = energy_map[task_name]
        else:
            # Unknown task - skip it
            print(f"[SCHEDULER][WARN] Unknown task in task_stats: {task_name}")
            continue

    # Filter tasks by energy feasibility
    min_factor = cfg.scheduler.min_bucket_factor
    eligible_tasks = {
        name: est
        for name, est in task_stats.items()
        if bucket_j >= est.est_joules * min_factor and est.est_joules > 0
    }

    if not eligible_tasks:
        print(f"[SCHEDULER] no eligible tasks (bucket_j={bucket_j:.2f}J)")
        return None

    # Warmup phase: ensure each task gets minimum runs
    warmup_runs = cfg.scheduler.warmup_runs
    for task_name, est in eligible_tasks.items():
        if est.runs < warmup_runs:
            print(f"[SCHEDULER] WARMUP selected task={task_name} runs={est.runs}/{warmup_runs}")
            return TaskChoice(
                name=task_name,
                reason="WARMUP",
                score=None
            )

    # Bandit selection: compute UCB scores
    ucb_c = cfg.scheduler.ucb_c
    scores = compute_scores(eligible_tasks, ucb_c)

    if not scores:
        print("[SCHEDULER] no scores computed - fallback")
        return None

    # Select task with highest UCB score
    bandit_task_name, bandit_score = max(scores.items(), key=lambda x: x[1])
    bandit_est = eligible_tasks[bandit_task_name]

    # Epsilon-greedy exploration
    epsilon = getattr(cfg.scheduler, "epsilon", 0.0)

    if epsilon > 0.0 and len(eligible_tasks) > 1:
        r = random.random()
        if r < epsilon:
            # EPSILON exploration: choose a task with fewer runs
            # among eligible tasks, excluding the current bandit task if possible
            exploration_candidates = {
                name: est for name, est in eligible_tasks.items()
                if name != bandit_task_name
            } or eligible_tasks

            # pick the task with the minimum runs (ties arbitrary)
            exp_task_name, exp_est = min(
                exploration_candidates.items(),
                key=lambda kv: kv[1].runs
            )
            exp_score = scores.get(exp_task_name, None)

            print(f"[SCHEDULER] EPSILON selected task={exp_task_name} "
                  f"η={exp_est.eta_avg:.6f} runs={exp_est.runs} "
                  f"bandit_task={bandit_task_name} r={r:.3f} epsilon={epsilon:.3f}")

            return TaskChoice(
                name=exp_task_name,
                reason="EPSILON",
                score=exp_score,
            )

    # Otherwise: stick to BANDIT choice
    print(f"[SCHEDULER] BANDIT selected task={bandit_task_name} score={bandit_score:.6f} "
          f"η={bandit_est.eta_avg:.6f} runs={bandit_est.runs} bucket_j={bucket_j:.2f}J")

    return TaskChoice(
        name=bandit_task_name,
        reason="BANDIT",
        score=bandit_score,
    )


def get_task_by_name(tasks, task_name: str):
    """
    Helper to find a task object by name from the task list.

    Args:
        tasks: List of task objects (TaskIndex, TaskLoRA, etc.)
        task_name: Name to match (e.g., "index_refresh", "lora_delta")

    Returns:
        Task object or None if not found
    """
    for task in tasks:
        if task.name == task_name:
            return task
    return None

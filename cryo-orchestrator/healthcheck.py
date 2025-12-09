#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CryoFlux v0.1 - System Healthcheck
Verifies JouleAgent connectivity and receipt database status.
"""
import sys
import sqlite3
from pathlib import Path

try:
    import requests
except ImportError:
    print("[ERROR] 'requests' package not found. Please install requirements.txt")
    sys.exit(1)

def load_config():
    """Load config.toml if available."""
    config = {
        "agent_url": "http://127.0.0.1:8787",
        "receipts_db": "./state/receipts.db"
    }

    # Try to find config.toml
    config_paths = [
        Path("../config.toml"),
        Path("config.toml"),
        Path("../../config.toml"),
    ]

    for path in config_paths:
        if path.exists():
            try:
                import toml
                data = toml.load(path)
                if "orchestrator" in data:
                    orch = data["orchestrator"]
                    if "agent_url" in orch:
                        config["agent_url"] = orch["agent_url"]
                    if "receipts_db" in orch:
                        config["receipts_db"] = orch["receipts_db"]
                print(f"[INFO] Loaded config from {path}")
                break
            except ImportError:
                print("[WARN] 'toml' package not found. Using defaults.")
                break
            except Exception as e:
                print(f"[WARN] Could not parse config.toml: {e}")
                break

    return config

def check_joule_agent(agent_url: str) -> bool:
    """Check if JouleAgent is reachable and responding correctly."""
    print(f"\n[CHECK] JouleAgent at {agent_url}")
    print("-" * 50)

    try:
        response = requests.get(f"{agent_url}/v1/sample", timeout=2)
        response.raise_for_status()

        data = response.json()

        # Verify expected fields
        expected_fields = ["bucket_j", "gpu_w", "cpu_w", "net_w", "ts", "hash"]
        missing_fields = [f for f in expected_fields if f not in data]

        if missing_fields:
            print(f"[WARN] Missing fields in response: {missing_fields}")
            print(f"[WARN] Received: {list(data.keys())}")
            return False

        # Print current state
        print(f"[OK] JouleAgent reachable")
        print(f"  Timestamp:    {data.get('ts', 'N/A')}")
        print(f"  Bucket:       {data.get('bucket_j', 0.0):.2f} J")
        print(f"  Net Power:    {data.get('net_w', 0.0):.2f} W")
        print(f"  GPU Power:    {data.get('gpu_w', 0.0):.2f} W")
        print(f"  CPU Power:    {data.get('cpu_w', 0.0):.2f} W")
        print(f"  Idle GPU:     {data.get('idle_gpu_w', 0.0):.2f} W")
        print(f"  Idle CPU:     {data.get('idle_cpu_w', 0.0):.2f} W")
        print(f"  Hash:         {data.get('hash', 'N/A')[:16]}...")
        return True

    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Could not connect to JouleAgent at {agent_url}")
        print(f"[ERROR] Is JouleAgent running?")
        return False
    except requests.exceptions.Timeout:
        print(f"[ERROR] JouleAgent request timed out")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] HTTP error: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

def check_receipts_db(db_path: str) -> bool:
    """Check if receipts database exists and print stats."""
    print(f"\n[CHECK] Receipts database at {db_path}")
    print("-" * 50)

    db_file = Path(db_path)

    if not db_file.exists():
        print(f"[WARN] Receipts database not found at {db_path}")
        print(f"[INFO] This is normal if the orchestrator hasn't run yet")
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if receipts table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='receipts'")
        if not cursor.fetchone():
            print(f"[ERROR] 'receipts' table not found in database")
            conn.close()
            return False

        # Get row count
        cursor.execute("SELECT COUNT(*) FROM receipts")
        count = cursor.fetchone()[0]

        if count == 0:
            print(f"[OK] Receipts database found: 0 rows")
            print(f"[INFO] No tasks have been executed yet")
        else:
            print(f"[OK] Receipts database found: {count} rows")

            # Get summary stats
            cursor.execute("""
                SELECT
                    task,
                    COUNT(*) as count,
                    SUM(joule) as total_joules,
                    AVG(delta) as avg_delta,
                    SUM(CASE WHEN delta > 0 THEN 1 ELSE 0 END) as accepted
                FROM receipts
                GROUP BY task
            """)

            print(f"\n  Task Summary:")
            print(f"  {'Task':<20} {'Count':<8} {'Joules':<12} {'Avg Δ':<12} {'Accepted':<10}")
            print(f"  {'-'*20} {'-'*8} {'-'*12} {'-'*12} {'-'*10}")

            for row in cursor.fetchall():
                task, cnt, joules, avg_delta, accepted = row
                joules_str = f"{joules:.1f}" if joules else "0.0"
                delta_str = f"{avg_delta:.4f}" if avg_delta else "0.0000"
                print(f"  {task:<20} {cnt:<8} {joules_str:<12} {delta_str:<12} {accepted:<10}")

        conn.close()
        return True

    except sqlite3.Error as e:
        print(f"[ERROR] Database error: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

def main():
    """Run all healthchecks."""
    print("=" * 50)
    print("CryoFlux v0.1 - System Healthcheck")
    print("=" * 50)

    # Load configuration
    config = load_config()

    # Run checks
    agent_ok = check_joule_agent(config["agent_url"])
    db_ok = check_receipts_db(config["receipts_db"])

    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)

    if agent_ok and db_ok:
        print("[OK] All systems operational")
        sys.exit(0)
    elif agent_ok:
        print("[OK] JouleAgent operational")
        print("[INFO] Receipts database not yet initialized (this is normal)")
        sys.exit(0)
    else:
        print("[ERROR] JouleAgent not reachable")
        print("[INFO] Please start JouleAgent first")
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
measure_rate.py

Poll http://127.0.0.1:8788/v1/sample repeatedly and display a terminal
progress view showing current bucket_j, average J/s, ETA for targets
and a simple ASCII progress bar. Also logs detailed samples to CSV.

Usage:
  python measure_rate.py --duration 120 --interval 1

This is intentionally lightweight (depends only on `requests`).
"""
import time
import sys
import argparse
import csv
from collections import deque
from datetime import datetime, timezone
import shutil

try:
    import requests
except Exception as e:
    print("This script requires the 'requests' package. Install it in your venv:")
    print("pip install requests")
    raise


def human_time(sec):
    if sec is None or sec != sec:
        return 'n/a'
    if sec < 0:
        return 'già raggiunto'
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    if h:
        return f"{h}h {m}m {s:.1f}s"
    if m:
        return f"{m}m {s:.1f}s"
    return f"{s:.1f}s"


def make_bar(progress, width=40):
    filled = int(round(progress * width))
    bar = '█' * filled + '-' * (width - filled)
    return f"[{bar}]"


def sample_loop(uri, duration, interval, window, targets, csv_path):
    samples = deque(maxlen=window)
    t0 = time.time()
    prev_bucket = None
    # open CSV and write header
    csv_file = open(csv_path, 'a', newline='')
    writer = csv.writer(csv_file)
    if csv_file.tell() == 0:
        writer.writerow(['ts_iso', 'elapsed_s', 'bucket_j', 'delta_j', 'avg_j_s', 'cpu_watts', 'gpu_watts'])

    try:
        while True:
            loop_start = time.time()
            try:
                r = requests.get(uri, timeout=5.0)
                r.raise_for_status()
                j = r.json()
            except Exception as e:
                print(f"\nErrore request: {e}")
                time.sleep(interval)
                continue

            bucket = float(j.get('bucket_j', 0.0))
            cpu_w = j.get('cpu_watts', None)
            gpu_w = j.get('gpu_watts', None)

            def fmt_watts(x):
                try:
                    if x is None:
                        return 'n/a'
                    fx = float(x)
                    return f"{fx:.3f}W"
                except Exception:
                    return str(x)

            if prev_bucket is None:
                delta = 0.0
            else:
                delta = bucket - prev_bucket

            prev_bucket = bucket
            samples.append(delta)

            # avg rate in J/s over last window seconds (samples are per-interval)
            avg_j_s = (sum(samples) / len(samples)) if samples else 0.0

            cols, _ = shutil.get_terminal_size((80, 20))

            # ETA calculations for targets
            etas = {}
            for tgt in targets:
                rem = tgt - bucket
                if avg_j_s <= 1e-12:
                    etas[tgt] = None
                else:
                    etas[tgt] = rem / avg_j_s

            # Progress for first target (for bar)
            main_target = targets[0]
            progress = min(max(bucket / main_target, 0.0), 1.0)

            # Render line
            bar = make_bar(progress, width=min(40, cols - 50))
            line = (
                f"Bucket: {bucket:.4f} J | avg: {avg_j_s:.6f} J/s | {bar} "
                f"ETA40: {human_time(etas.get(40))} | ETA120: {human_time(etas.get(120))} "
                f"(cpu:{fmt_watts(cpu_w)} gpu:{fmt_watts(gpu_w)})"
            )

            # Print updating in place
            sys.stdout.write('\r' + ' ' * (cols - 1))
            sys.stdout.flush()
            sys.stdout.write('\r' + line[:cols - 1])
            sys.stdout.flush()

            # Log to CSV
            writer.writerow([datetime.now(timezone.utc).isoformat(), f"{time.time() - t0:.1f}", f"{bucket:.8f}", f"{delta:.8f}", f"{avg_j_s:.8f}", cpu_w, gpu_w])
            csv_file.flush()

            # Duration check
            if duration is not None and (time.time() - t0) >= duration:
                break

            # Sleep until next interval (account for time spent)
            elapsed = time.time() - loop_start
            to_sleep = max(0.0, interval - elapsed)
            time.sleep(to_sleep)

    finally:
        csv_file.close()
        print('\nDone.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--uri', default='http://127.0.0.1:8788/v1/sample', help='JouleAgent sample endpoint')
    p.add_argument('--duration', type=int, default=120, help='Total seconds to run the sampler (0 for indefinite)')
    p.add_argument('--interval', type=float, default=1.0, help='Seconds between samples')
    p.add_argument('--window', type=int, default=20, help='Samples for moving average (seconds)')
    p.add_argument('--csv', default='measurements.csv', help='CSV file to append samples')
    p.add_argument('--targets', nargs='*', type=float, default=[40.0, 120.0], help='Energy targets in J (first is used for bar)')
    args = p.parse_args()

    duration = None if args.duration == 0 else args.duration
    try:
        sample_loop(args.uri, duration, args.interval, args.window, args.targets, args.csv)
    except KeyboardInterrupt:
        print('\nInterrupted by user')


if __name__ == '__main__':
    main()

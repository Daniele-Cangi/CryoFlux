import sqlite3
from datetime import datetime

c = sqlite3.connect('state/receipts.db')
cur = c.cursor()
rows = cur.execute("""
    SELECT id, ts, task, joule
    FROM receipts ORDER BY id DESC LIMIT 10
""").fetchall()

print("Ultimi task eseguiti:")
print("-" * 60)
prev_ts = None
for r in rows:
    dt = datetime.fromtimestamp(r[1])
    if prev_ts:
        gap = prev_ts - r[1]
        gap_min = gap / 60.0
        print(f"ID={r[0]} {dt.strftime('%H:%M:%S')} {r[2]:15} {r[3]:5.1f}J  [gap: {gap_min:.1f} min]")
    else:
        print(f"ID={r[0]} {dt.strftime('%H:%M:%S')} {r[2]:15} {r[3]:5.1f}J")
    prev_ts = r[1]

print("\n" + "=" * 60)
# calcola media gap tra task recenti (ultimi 5)
if len(rows) >= 2:
    gaps = []
    for i in range(len(rows) - 1):
        gaps.append(rows[i][1] - rows[i+1][1])
    avg_gap = sum(gaps) / len(gaps)
    print(f"Gap medio tra task: {avg_gap/60:.1f} minuti ({avg_gap:.0f} secondi)")
c.close()

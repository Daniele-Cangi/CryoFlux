import sqlite3
c = sqlite3.connect('state/receipts.db')
cur = c.cursor()
rows = cur.execute("""
    SELECT id, datetime(ts,'unixepoch'), task, joule, round(delta,4), loss
    FROM receipts ORDER BY id DESC LIMIT 10
""").fetchall()
for r in rows:
    print(f"ID={r[0]} ts={r[1]} task={r[2]} joule={r[3]} delta={r[4]} loss={r[5]}")
c.close()

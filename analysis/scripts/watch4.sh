#!/bin/bash
cd /home/arnaud/devel/src/data_manipulation_tools
SP="$(dirname "$0")"
while pgrep -f "kinetics_tu[n]e" >/dev/null; do
  .venv/bin/python - >> "$SP/watch4.log" 2>&1 <<'PY'
import csv, datetime, statistics, os
p="results/kinetics-cell-20260905/trials.csv"
if not os.path.exists(p): raise SystemExit
rows=[r for r in csv.DictReader(open(p)) if r["status"]=="ok"]
if not rows: raise SystemExit
rows.sort(key=lambda r:int(r["trial"]))
gens=[]
for i in range(0,len(rows),24):
    ch=[float(r["objective"]) for r in rows[i:i+24]]
    if len(ch)>=10: gens.append(min(ch))
best=min(float(r["objective"]) for r in rows)
print(f"{datetime.datetime.now():%H:%M}  {len(rows)}/440  best {best:+.4f}  gens " +
      " ".join(f"{g:+.2f}" for g in gens))
PY
  sleep 1800
done
echo "$(date +%H:%M) finished" >> "$SP/watch4.log"

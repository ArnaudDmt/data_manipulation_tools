#!/bin/bash
# One line per hour into a log, plus a divergence check: if the best has not improved for
# three generations the search is walking away and needs attention rather than more time.
cd /home/arnaud/devel/src/data_manipulation_tools
SP="$(dirname "$0")"
while pgrep -f "python.*kinetics_tune\.py --tracking" >/dev/null; do
  date +"---- %H:%M ----" >> "$SP/hourly.log"
  .venv/bin/python "$SP/hourly.py" >> "$SP/hourly.log" 2>&1
  .venv/bin/python - >> "$SP/hourly.log" 2>&1 <<'PY'
import csv, statistics
import os
if not os.path.exists("results/kinetics-focus3-20260904/trials.csv"): raise SystemExit
rows=[r for r in csv.DictReader(open("results/kinetics-focus3-20260904/trials.csv")) if r["status"]=="ok"]
rows.sort(key=lambda r:int(r["trial"]))
gens=[]
for i in range(0,len(rows),24):
    ch=[float(r["objective"]) for r in rows[i:i+24]]
    if len(ch)>=8: gens.append((min(ch), statistics.median(ch)))
print("        gens best:", "  ".join(f"{b:+.3f}" for b,_ in gens))
if len(gens)>=4 and all(gens[i][0] >= gens[0][0] for i in range(1,len(gens))):
    print("        *** DIVERGING: no generation has beaten the first ***")
PY
  sleep 3600
done
echo "search finished at $(date +%H:%M)" >> "$SP/hourly.log"

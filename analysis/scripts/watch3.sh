#!/bin/bash
cd /home/arnaud/devel/src/data_manipulation_tools
SP="$(dirname "$0")"
while pgrep -f "python.*kinetics_tune\.py --tracking" >/dev/null; do
  .venv/bin/python - >> "$SP/watch3.log" 2>&1 <<'PY'
import csv, datetime
rows=[r for r in csv.DictReader(open("results/kinetics-focus3-20260904/trials.csv")) if r["status"]=="ok"]
rows.sort(key=lambda r:int(r["trial"]))
gens=[]
for i in range(0,len(rows),24):
    ch=[float(r["objective"]) for r in rows[i:i+24]]
    if len(ch)>=8: gens.append((min(ch), len(ch)))
best=min(float(r["objective"]) for r in rows)
flag = "BEAT SEED" if best < 1.3694 else ""
print(f"{datetime.datetime.now():%H:%M}  {len(rows)} trials  best {best:+.4f}  "
      + " ".join(f"{b:+.3f}({n})" for b,n in gens) + "  " + flag)
PY
  sleep 1800
done
echo "$(date +%H:%M) search finished" >> "$SP/watch3.log"

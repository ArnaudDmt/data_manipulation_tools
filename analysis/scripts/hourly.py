#!/usr/bin/env python3
"""Hourly progress line. Counts work in flight, so a mid-batch sample is not read as idle."""
import csv, json, math, collections, statistics, shutil, subprocess
T = 'results/kinetics-focus3-20260904/trials.csv'
CAP = {'vel_xy': 1.10, 'vel_z': 1.10, 'tilt': 1.10, 'trans_z': 1.05}
geo = lambda v: math.exp(sum(math.log(max(x, 1e-9)) for x in v) / len(v))
try:
    rows = list(csv.DictReader(open(T)))
except OSError:
    print("HOURLY  no trials yet"); raise SystemExit
ok = [r for r in rows if r['status'] == 'ok' and r['ratios'].strip() not in ('', '{}')]
if not ok:
    print(f"HOURLY  {len(rows)} trials, none scored yet"); raise SystemExit
pid = subprocess.run(["pgrep", "-f", "python.*kinetics_tune.py --tracking"],
                     capture_output=True, text=True).stdout.split()
el = int(subprocess.run(["ps", "-o", "etimes=", "-p", pid[0]], capture_output=True,
                        text=True).stdout) if pid else 1
kids = subprocess.run(["pgrep", "-f", "scripts/kinetics_eval.py"],
                      capture_output=True, text=True).stdout.split()
inflight = sum(int(subprocess.run(["ps", "-o", "etimes=", "-p", p], capture_output=True,
                                  text=True).stdout or 0) for p in kids)
busy = sum(float(r['seconds']) for r in rows)
js = [float(r['objective']) for r in ok]
best = min(ok, key=lambda r: float(r['objective']))
per = collections.defaultdict(list)
for k, v in json.loads(best['ratios']).items():
    per[k.split('|')[1]].append(v)
g = {m: geo(per[m]) for m in per}
left = 260 - len(rows)
util = (busy + inflight) / max(12 * el, 1)
mem = int(open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]) / 2**20
print(f"HOURLY  {len(rows)}/260   {dict(collections.Counter(r['status'] for r in rows))}"
      f"   median J {statistics.median(js):+.3f}   best {min(js):+.4f}   seed +1.3694")
print("        best #%s  " % best['trial'] + "  ".join(
    f"{m}={g[m]:.3f}" for m in ('trans_xy', 'yaw', 'trans_z', 'tilt', 'vel_xy', 'vel_z')))
print(f"        caps inside: {[m for m, c in CAP.items() if g.get(m, 9) <= c] or 'none'}"
      f"   util {100*util:.0f}%   ~{left*(busy/max(len(rows),1))/12/max(util,.01)/3600:.1f}h left"
      f"   disk {shutil.disk_usage('/').free/2**30:.0f}GB   mem {mem:.0f}GB")

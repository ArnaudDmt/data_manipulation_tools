#!/usr/bin/env python3
"""Print a per-metric comparison whenever the search finds a new best config.

Emits one block per improvement, so a Monitor tailing this reports only real progress.
"""
import csv, json, math, os, sys, time, collections

TRIALS = "/home/arnaud/devel/src/data_manipulation_tools/results/kinetics-yaw-20260904/trials.csv"
STATE  = os.path.join(os.path.dirname(__file__), ".best_reported")
METRICS = ("trans_xy", "trans_z", "yaw", "tilt", "vel_xy", "vel_z")
# installed tuning, all 13 datasets, measured from the baseline run
BASE = {"trans_xy":0.793,"trans_z":1.073,"yaw":0.941,"tilt":1.007,"vel_xy":1.345,"vel_z":1.298}
BASE_J, BASE_WINS = 1.9557, 27

REF = json.load(open(os.path.join(os.path.dirname(__file__), "riekf_absolute.json")))
RI, SEG, UNIT = REF["riekf"], REF["segment"], REF["units"]
ORDER = ["HRP5_MultiContact_1","HRP5_MultiContact_2","HRP5_MultiContact_3","HRP5_MultiContact_4",
         "HRP5P_LongWalk","KO_TRO2024_RHPS1_1","KO_TRO2024_RHPS1_2","KO_TRO2024_RHPS1_3",
         "KO_TRO2024_RHPS1_4","KO_TRO2024_RHPS1_5","KO_TRO_2024_RHPS1_SLIPPAGE_1",
         "KO_TRO_2024_RHPS1_SLIPPAGE_2","KO_TRO_2024_RHPS1_SLIPPAGE_3"]

def totals(byp, m):
    """Sum of Kinetics and RI-EKF absolute errors over datasets sharing a segment length."""
    out = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for p, mm in byp.items():
        if m in mm:
            g = out[SEG[p]]
            g[0] += RI[p][m] * mm[m]; g[1] += RI[p][m]; g[2] += 1
    return out

def absolute_table(byp, m):
    fmt = ".4f" if m.startswith(("trans", "vel")) else ".3f"
    lines = [f"   {m} [{UNIT[m]}]   {'dataset':28}{'seg':>4}{'Kinetics':>10}{'RI-EKF':>10}{'ratio':>8}"]
    for p in ORDER:
        if p not in byp or m not in byp[p]: continue
        r = byp[p][m]; ri = RI[p][m]
        lines.append(f"   {'':>{len(m)+len(UNIT[m])+6}}{p:28}{SEG[p]:4g}{ri*r:10{fmt}}{ri:10{fmt}}{r:8.3f}"
                     + ("  WIN" if r < 1 else ""))
    for seg, (ko, ri, n) in sorted(totals(byp, m).items()):
        lines.append(f"   {'':>{len(m)+len(UNIT[m])+6}}{'sum over '+str(n)+' runs @ '+str(seg)+'m':28}{'':4}{ko:10{fmt}}{ri:10{fmt}}{ko/ri:8.3f}")
    return lines

# The installed configuration is the one the paper used, so a dimension far from it is a claim
# that the published value was wrong -- worth surfacing explicitly rather than leaving buried in
# an overlay file. A decade is the threshold: these are log-scale parameters, and less than that
# is within the range the earlier ablations showed to be inconsequential.
DEPARTURE_DECADES = 1.0
_INSTALLED_POINT = {}

def installed_point():
    if not _INSTALLED_POINT:
        import sys as _sys
        _sys.path.insert(0, "/home/arnaud/devel/src/data_manipulation_tools/scripts")
        import kinetics_tune as _kt
        base, _widths, flex = _kt.base_covariances()
        _INSTALLED_POINT.update(_kt.base_point(base, flex))
    return _INSTALLED_POINT

def departures(point):
    """Dimensions the search moved a decade or more away from the published configuration."""
    ref = installed_point()
    out = []
    for name, value in sorted(point.items()):
        if name not in ref:
            continue
        delta = value - ref[name]
        if abs(delta) >= DEPARTURE_DECADES:
            out.append((name, 10.0 ** ref[name], 10.0 ** value, delta))
    return sorted(out, key=lambda row: -abs(row[3]))

def print_departures(row):
    moved = departures(json.loads(row["point"]))
    if not moved:
        print("   no parameter moved a decade or more from the published configuration")
        return
    print(f"   this win moves {len(moved)} parameter(s) a decade or more from the paper's value:")
    print(f"     {'parameter':34}{'paper':>12}{'found':>12}{'change':>12}")
    for name, was, now, delta in moved:
        print(f"     {name:34}{was:>12.3g}{now:>12.3g}{delta:>+11.1f} dec")


def geo(v): return math.exp(sum(math.log(max(x,1e-9)) for x in v)/len(v))

def best_row():
    try:
        rows=[r for r in csv.DictReader(open(TRIALS)) if r["status"]=="ok" and r["ratios"].strip() not in ("","{}")]
    except OSError:
        return None
    return min(rows, key=lambda r: float(r["objective"])) if rows else None

def report(r):
    per=collections.defaultdict(list)
    for k,v in json.loads(r["ratios"]).items(): per[k.split("|")[1]].append(v)
    j=float(r["objective"])
    print(f"NEW BEST  trial #{r['trial']}  J={j:+.4f}  (baseline {BASE_J:+.4f}, {'better' if j<BASE_J else 'WORSE'})"
          f"  wins {r['wins']}/78 vs {BASE_WINS}/78")
    print(f"  {'metric':<9} {'ratio':>7} {'installed':>10} {'change':>8}  wins")
    for m in METRICS:
        if m not in per: continue
        g=geo(per[m]); b=BASE[m]
        print(f"  {m:<9} {g:7.3f} {b:10.3f} {100*(g/b-1):+7.1f}%  {sum(x<1 for x in per[m])}/{len(per[m])}")
    print_departures(r)
    byp={}
    for k,v in json.loads(r["ratios"]).items():
        pj,m=k.split("|"); byp.setdefault(pj,{})[m]=v
    print("  errors  Kinetics | RI-EKF   (translation/yaw/tilt relative per segment; velocity RMS over run)")
    for group in (("trans_xy","trans_z","yaw"), ("tilt","vel_xy","vel_z")):
        print("  " + f"{'dataset':10}{'seg':>4}" + "".join(f"{m+' ['+UNIT[m]+']':>21}" for m in group))
        for pj in ORDER:
            if pj not in byp: continue
            line=f"  {pj.replace('HRP5_MultiContact_','MC_').replace('KO_TRO2024_RHPS1_','RHPS1_').replace('KO_TRO_2024_RHPS1_SLIPPAGE_','SLIP_').replace('HRP5P_LongWalk','LongWalk'):10}{SEG[pj]:>4g}"
            for m in group:
                if m not in byp[pj]: continue
                ri=RI[pj][m]; ko=ri*byp[pj][m]; f=".4f" if m.startswith(("trans","vel")) else ".3f"
                line+=f"{ko:>10{f}}|{ri:<10{f}}"
            print(line)
    sys.stdout.flush()

def main():
    last=float(open(STATE).read()) if os.path.exists(STATE) else float("inf")
    while True:
        r=best_row()
        if r is not None:
            j=float(r["objective"])
            if j < last - 1e-9:
                report(r); last=j
                open(STATE,"w").write(repr(j))
        time.sleep(120)

main()

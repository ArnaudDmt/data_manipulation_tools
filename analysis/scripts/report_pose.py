#!/usr/bin/env python3
"""Report configs that make a clear gain on translation or yaw, regardless of J.

The J-based reporter stays silent on a config that wins the pose metrics but loses on velocity,
which is exactly the trade-off worth seeing. This one watches trans_xy / trans_z / yaw only.
"""
import csv, json, math, os, sys, time, collections

TRIALS="/home/arnaud/devel/src/data_manipulation_tools/results/kinetics-yaw-20260904/trials.csv"
STATE=os.path.join(os.path.dirname(__file__), ".pose_records")
POSE=("trans_xy","trans_z","yaw")
ALL=("trans_xy","trans_z","yaw","tilt","vel_xy","vel_z")
BASE={"trans_xy":0.793,"trans_z":1.073,"yaw":0.941,"tilt":1.007,"vel_xy":1.345,"vel_z":1.298}
BASE_WINS={"trans_xy":9,"trans_z":5,"yaw":8}
MC=[f"HRP5_MultiContact_{i}" for i in range(1,5)]

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
    print(f"   {len(moved)} parameter(s) a decade or more from the paper's value:")
    print(f"     {'parameter':34}{'paper':>12}{'found':>12}{'change':>12}")
    for name, was, now, delta in moved:
        print(f"     {name:34}{was:>12.3g}{now:>12.3g}{delta:>+11.1f} dec")


def geo(v): return math.exp(sum(math.log(max(x,1e-9)) for x in v)/len(v))

def load():
    try: return [r for r in csv.DictReader(open(TRIALS)) if r["status"]=="ok" and r["ratios"].strip() not in ("","{}")]
    except OSError: return []

def main():
    rec=json.load(open(STATE)) if os.path.exists(STATE) else {}
    seen=set(rec.get("seen",[]))
    best_geo={m:rec.get("geo",{}).get(m, BASE[m]) for m in POSE}
    best_win={m:rec.get("win",{}).get(m, BASE_WINS[m]) for m in POSE}
    while True:
        for r in load():
            if r["trial"] in seen: continue
            seen.add(r["trial"])
            rt=json.loads(r["ratios"]); per=collections.defaultdict(list); byp={}
            for k,v in rt.items():
                p,m=k.split("|"); per[m].append(v); byp.setdefault(p,{})[m]=v
            notes=[]
            for m in POSE:
                if m not in per: continue
                g=geo(per[m]); w=sum(x<1 for x in per[m])
                if g < best_geo[m]*0.99: notes.append(f"{m} geomean {g:.3f} (was {best_geo[m]:.3f}, base {BASE[m]:.3f})"); best_geo[m]=g
                if w > best_win[m]:      notes.append(f"{m} wins {w}/13 (was {best_win[m]})"); best_win[m]=w
            sweep=[m for m in POSE if m in per and sum(x<1 for x in per[m])==13]
            if sweep: notes.append("*** BEATS RI-EKF ON ALL 13: " + ", ".join(sweep) + " ***")
            if notes:
                print(f"POSE RECORD  trial #{r['trial']}  J={float(r['objective']):+.4f}")
                for n in notes: print("   " + n)
                print("   full picture: " + "  ".join(
                    f"{m}={geo(per[m]):.3f}({sum(x<1 for x in per[m])}/13)" for m in ALL if m in per))
                print("   MultiContact trans_xy: " + " ".join(
                    f"{p.split('_')[-1]}={byp[p]['trans_xy']:.3f}" for p in MC if p in byp))
                # Departures from the published configuration are only reported alongside a
                # win -- a new record can still be a config that loses overall, and listing the
                # parameters it moved would invite reading a loss as a finding.
                for m in POSE:
                    if any(m in n for n in notes):
                        for line in absolute_table(byp, m): print(line)
                sys.stdout.flush()
        json.dump({"seen":sorted(seen),"geo":best_geo,"win":best_win}, open(STATE,"w"))
        time.sleep(120)

main()

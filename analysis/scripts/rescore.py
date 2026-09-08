#!/usr/bin/env python3
"""Re-score every recorded trial under a goal-aligned objective.

The trials store full per-dataset ratios, so changing what we optimise for costs nothing:
no replay is re-run, the ranking is simply recomputed.

Goal, in the user's terms: win against the RI-EKF on most metrics; win *by far* on horizontal
translation and yaw; degrade less than the RI-EKF does under slippage.
"""
import csv, json, math, collections, glob, sys

METRICS = ("trans_xy", "trans_z", "yaw", "tilt", "vel_xy", "vel_z")
# Priority is explicit rather than emergent: the two metrics the thesis claim rests on carry
# four times the weight of a velocity channel that only has to keep up.
WEIGHT = {"trans_xy": 3.0, "yaw": 3.0, "trans_z": 1.0, "tilt": 1.0, "vel_xy": 0.75, "vel_z": 0.75}
WORST_WEIGHT = 0.15      # was 0.4 -- enough to catch divergence, not enough to steer
WORST_BETA = 6.0
LOSS_PENALTY = 0.60      # breadth: charged on the weighted fraction of datasets lost
SLIP_WEIGHT = 1.00       # charged only when the KO degrades more than the RI-EKF does
SLIP_METRICS = ("trans_xy", "yaw")

NOMINAL = [f"KO_TRO2024_RHPS1_{i}" for i in range(1, 6)]
SLIP = [f"KO_TRO_2024_RHPS1_SLIPPAGE_{i}" for i in range(1, 4)]

def geo(v):
    return math.exp(sum(math.log(max(x, 1e-9)) for x in v) / len(v))

def split(ratios):
    per, byp = collections.defaultdict(list), {}
    for k, v in ratios.items():
        p, m = k.split("|")
        per[m].append(v); byp.setdefault(p, {})[m] = v
    return per, byp

def slippage_terms(byp):
    """geomean(ratio on slippage) / geomean(ratio on nominal), per metric.

    Both are Kinetics/RI-EKF, so the quotient is exactly how much more the Kinetics Observer
    degrades under slippage than the RI-EKF does. Below 1 means it degrades less.
    """
    out = {}
    for m in SLIP_METRICS:
        nom = [byp[p][m] for p in NOMINAL if p in byp and m in byp[p]]
        sl = [byp[p][m] for p in SLIP if p in byp and m in byp[p]]
        if nom and sl:
            out[m] = geo(sl) / geo(nom)
    return out

def score(ratios):
    per, byp = split(ratios)
    costs, weights = [], []
    for m in METRICS:
        for v in per.get(m, []):
            costs.append(WEIGHT[m] * math.log(max(v, 1e-9))); weights.append(WEIGHT[m])
    if not costs:
        return None
    total = sum(weights)
    mean = sum(costs) / total
    soft = math.log(sum(math.exp(WORST_BETA * c / w) for c, w in zip(costs, weights)) / len(costs)) / WORST_BETA
    lost = sum(w for c, w in zip(costs, weights) if c > 0) / total
    slip = slippage_terms(byp)
    slip_cost = sum(max(0.0, math.log(v)) for v in slip.values())
    return {"J": (1 - WORST_WEIGHT) * mean + WORST_WEIGHT * soft
                 + LOSS_PENALTY * lost + SLIP_WEIGHT * slip_cost,
            "mean": mean, "lost": lost, "slip": slip,
            "geo": {m: geo(per[m]) for m in METRICS if m in per},
            "wins": {m: sum(x < 1 for x in per[m]) for m in METRICS if m in per}}

def main():
    rows = []
    for f in sorted(glob.glob("results/kinetics-retuning-*/trials.csv")):
        for r in csv.DictReader(open(f)):
            if r["status"] != "ok" or r["ratios"].strip() in ("", "{}"): continue
            rt = json.loads(r["ratios"])
            if len(rt) < 70: continue           # full-set trials only
            s = score(rt)
            if s: rows.append((s, r, f.split("/")[1]))
    rows.sort(key=lambda x: x[0]["J"])
    print(f"re-scored {len(rows)} full-set trials under the goal-aligned objective\n")
    hdr = f"{'rank':>4} {'trial':>18} {'J':>8} {'lost':>6} " + "".join(f"{m:>10}" for m in METRICS) + f"{'slip_xy':>9}{'slip_yaw':>9}"
    print(hdr)
    for i, (s, r, run) in enumerate(rows[:15], 1):
        g = "".join(f"{s['geo'].get(m, float('nan')):7.3f}({s['wins'].get(m,0):>2})" for m in METRICS)
        print(f"{i:>4} {run.split('-')[-1]+'#'+r['trial']:>18} {s['J']:+8.4f} {s['lost']:6.2f} {g}"
              f"{s['slip'].get('trans_xy',float('nan')):9.3f}{s['slip'].get('yaw',float('nan')):9.3f}")
    return rows

if __name__ == "__main__":
    main()

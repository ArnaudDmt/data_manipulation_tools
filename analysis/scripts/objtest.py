import sys, glob; sys.path.insert(0,'scripts')
from pathlib import Path
import kinetics_tune as kt
PROJ=["HRP5_MultiContact_1","HRP5_MultiContact_2","HRP5_MultiContact_3","HRP5_MultiContact_4",
 "KO_TRO2024_RHPS1_1","KO_TRO2024_RHPS1_2","KO_TRO2024_RHPS1_3","KO_TRO2024_RHPS1_4","KO_TRO2024_RHPS1_5",
 "KO_TRO_2024_RHPS1_SLIPPAGE_1","KO_TRO_2024_RHPS1_SLIPPAGE_2","KO_TRO_2024_RHPS1_SLIPPAGE_3"]
r=kt.read_ratios(Path(glob.glob("results/PW_cal-*/summary.csv")[0]),PROJ)
# velocity ratios measured earlier by the harness, per category
VEL={"MultiContact":(1.4854,1.2347),"RHPS1 walk":(1.3441,1.5269),"RHPS1 slip":(1.2180,1.1819)}
for p in PROJ:
    c=kt.category_of(p); r[f"{p}|vel_xy"]=VEL[c][0]; r[f"{p}|vel_z"]=VEL[c][1]
print("category map:", {p.split('_')[0]+"...": kt.category_of(p) for p in (PROJ[0],PROJ[4],PROJ[9])})
for m in ("trans_xy","yaw","trans_z","tilt","vel_xy"):
    print(f"  {m:9s} " + "  ".join(f"{c}={v:.3f}" for c,v in sorted(kt.by_category(r,m).items())))
print(f"\n  requirement_violation = {kt.requirement_violation(r):.4f}   (trans_xy floor 0.80 on RHPS1 only, yaw 0.95 all)")
print(f"  cap_violation         = {kt.cap_violation(r):.4f}   (trans_z/tilt 1.05, vel_xy 1.10, per category)")
print(f"  slippage_penalty      = {kt.slippage_penalty(r):.4f}")
print(f"  objective             = {kt.objective(r):.4f}")
# a control: pooled-geomean would have hidden these
import collections, math
per=collections.defaultdict(list)
for case,v in r.items(): per[kt.metric_of(case)].append(v)
print("\n  pooled geomean (what the OLD code checked):")
for m in ("trans_xy","yaw","tilt","vel_xy"):
    print(f"    {m:9s} {kt.geomean(per[m]):.3f}")

# --- term-by-term decomposition, so "major target" can be judged rather than assumed ----------
import math, collections
costs, weights = [], []
for case, value in r.items():
    w = kt.METRIC_WEIGHT.get(kt.metric_of(case), 1.0); lg = math.log(max(value, 1e-9))
    costs.append(w * (lg + kt.HARD_REGRESSION_PENALTY * max(0.0, lg))); weights.append(w)
tot = sum(weights); mean = sum(costs)/tot
sw = math.log(sum(math.exp(kt.WORST_BETA*c/w) for c,w in zip(costs,weights))/len(costs))/kt.WORST_BETA
lost = sum(w for c,w in zip(costs,weights) if c>0)/tot
terms = [("weighted mean", (1-kt.WORST_WEIGHT)*mean), ("soft worst", kt.WORST_WEIGHT*sw),
         ("datasets lost", kt.LOSS_PENALTY*lost),
         ("slippage", kt.SLIP_WEIGHT*kt.slippage_penalty(r)),
         ("caps (vel_xy)", kt.CAP_BARRIER*kt.cap_violation(r)),
         ("floors (yaw)", kt.REQUIRED_BARRIER*kt.requirement_violation(r))]
print("\n  objective terms on the current best config:")
for n,v in terms: print(f"    {n:16s}{v:+8.3f}")
print(f"    {'TOTAL':16s}{sum(v for _,v in terms):+8.3f}")

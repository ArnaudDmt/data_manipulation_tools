"""Velocity ratios via the harness' own code path (local frame, mocap grid, time offset)."""
import sys,glob,collections,numpy as np; sys.path.insert(0,'scripts')
from pathlib import Path
from kinetics_tune import reference_velocities, velocity_ratios, VELOCITY_METRICS
PROJECTS=["HRP5_MultiContact_1","HRP5_MultiContact_2","HRP5_MultiContact_3","HRP5_MultiContact_4",
 "KO_TRO2024_RHPS1_1","KO_TRO2024_RHPS1_2","KO_TRO2024_RHPS1_3","KO_TRO2024_RHPS1_4","KO_TRO2024_RHPS1_5",
 "KO_TRO_2024_RHPS1_SLIPPAGE_1","KO_TRO_2024_RHPS1_SLIPPAGE_2","KO_TRO_2024_RHPS1_SLIPPAGE_3"]
def cat(p):
    if "MultiContact" in p: return "MultiContact"
    if "SLIPPAGE" in p.upper(): return "RHPS1 slip"
    return "RHPS1 walk"
cache=reference_velocities(PROJECTS)
print(f"{'run':22s}"+"".join(f"{m+' '+c:>24s}" for m in VELOCITY_METRICS
                              for c in ("MC","walk","slip"))[:0] or "")
rows={}
for lab,pref in (("baseline","PW_base"),("wrench calibration","PW_cal")):
    d=glob.glob(f"results/{pref}-*/")
    if not d: print(f"{lab}: no results"); continue
    r=velocity_ratios(Path(d[0]),PROJECTS,cache)
    acc=collections.defaultdict(list)
    for k,v in r.items():
        name,metric=k.split("|"); acc[(cat(name),metric)].append(v)
    rows[lab]={k:float(np.mean(v)) for k,v in acc.items()}
cats=["MultiContact","RHPS1 walk","RHPS1 slip"]
for metric in VELOCITY_METRICS:
    print(f"\n--- {metric}: KO / RI-EKF velocity RMSE ratio")
    print(f"{'run':22s}"+"".join(f"{c:>16s}" for c in cats))
    for lab in rows:
        print(f"{lab:22s}"+"".join(f"{rows[lab].get((c,metric),float('nan')):16.4f}" for c in cats))

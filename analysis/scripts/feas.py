"""Per-category KO/RI-EKF ratios against the stated constraint set."""
import csv,glob,os,collections,numpy as np
CAT=lambda p: ("MultiContact" if "MultiContact" in p else "LongWalk" if "LongWalk" in p
               else "RHPS1 slip" if "SLIPPAGE" in p.upper() else "RHPS1 walk")
POSE=("trans_xy","yaw","trans_z","tilt")
def pose(prefs):
    acc=collections.defaultdict(lambda: collections.defaultdict(list))
    for pref in prefs:
        for f in glob.glob(f"results/{pref}-*/summary.csv"):
            for r in csv.DictReader(open(f)):
                if r["statistic"]!="rmse" or r["metric"] not in POSE: continue
                acc[(CAT(r["project"]),r["metric"])][r["estimator"]].append((r["project"],float(r["value"])))
    out={}
    for k,d in acc.items():
        if "Kinetics" not in d or "RI-EKF" not in d: continue
        ko=dict(d["Kinetics"]); ri=dict(d["RI-EKF"]); c=sorted(set(ko)&set(ri))
        if c: out[k]=np.mean([ko[p] for p in c])/np.mean([ri[p] for p in c])
    return out
main=pose(["PW_cal"]); lw=pose(["W_LW","V_tuned_a0.0"])
for k,v in lw.items():
    if k[0]=="LongWalk": main[k]=v
VEL={("MultiContact","vel_xy"):1.4854,("RHPS1 walk","vel_xy"):1.3441,("RHPS1 slip","vel_xy"):1.2180,
     ("LongWalk","vel_xy"):1.070,
     ("MultiContact","vel_z"):1.2347,("RHPS1 walk","vel_z"):1.5269,("RHPS1 slip","vel_z"):1.1819,
     ("LongWalk","vel_z"):1.227}
main.update(VEL)
LIM={"trans_xy":("<=",0.80),"yaw":("<=",0.95),"trans_z":("<=",1.05),
     "tilt":("<=",1.05),"vel_xy":("<=",1.10),"vel_z":(None,None)}
cats=["MultiContact","LongWalk","RHPS1 walk","RHPS1 slip"]
print(f"{'metric':10s}{'limit':>8s}"+"".join(f"{c:>16s}" for c in cats))
nfail=0
for m,(op,lim) in LIM.items():
    row=f"{m:10s}{(str(lim) if lim else '-'):>8s}"
    for c in cats:
        v=main.get((c,m))
        if v is None: row+=f"{'-':>16s}"; continue
        bad = lim is not None and v>lim
        nfail+= 1 if bad else 0
        row+=f"{v:14.3f}{'X' if bad else ' '} "
    print(row)
print(f"\n  X = violates the stated constraint.  {nfail} violations across 4 categories.")
print("  (LongWalk rows come from an older config -- it is not in the 12-dataset runs;")
print("   velocity rows are the measured harness ratios.)")

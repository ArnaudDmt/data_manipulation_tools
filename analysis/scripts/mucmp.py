import csv,glob,collections,numpy as np,sys
sys.path.insert(0,'scripts'); import kinetics_tune as kt
CAT=lambda p: ("MultiContact" if "MultiContact" in p else "RHPS1 slip" if "SLIPPAGE" in p.upper() else "RHPS1 walk")
def load(pref):
    d=glob.glob(f"results/{pref}-*/summary.csv")
    if not d: return None
    acc=collections.defaultdict(lambda: collections.defaultdict(list))
    for f in d:
        for r in csv.DictReader(open(f)):
            if r["statistic"]!="mean" or r["metric"] not in ("trans_xy","yaw"): continue
            acc[(CAT(r["project"]),r["metric"])][r["estimator"]].append((r["project"],float(r["value"])))
    out={}
    for k,v in acc.items():
        ko=dict(v.get("Kinetics",[])); ri=dict(v.get("RI-EKF",[])); c=sorted(set(ko)&set(ri))
        if c: out[k]=np.mean([ko[p] for p in c])/np.mean([ri[p] for p in c])
    return out
runs=[("baseline (mu off)","PW_cal"),("mu = 0.8","MU_0.8"),("mu = 0.5","MU_0.5")]
cats=["MultiContact","RHPS1 walk","RHPS1 slip"]
for m in ("trans_xy","yaw"):
    print(f"\n--- {m}  (KO/RI)")
    print(f"{'run':20s}"+"".join(f"{c:>15s}" for c in cats)+f"{'degradation':>14s}")
    for lab,pref in runs:
        r=load(pref)
        if not r: print(f"{lab:20s} (no results)"); continue
        row=f"{lab:20s}"+"".join(f"{r.get((c,m),float('nan')):15.3f}" for c in cats)
        w,s=r.get(("RHPS1 walk",m)),r.get(("RHPS1 slip",m))
        row+=f"{(s/w if w and s else float('nan')):14.3f}"
        print(row)

import csv,glob,re,collections,numpy as np
RUNS=[("calibrated (cw 10)","PW_cal"),("cw 1.0","CW_1.0"),("cw 3.0","CW_3.0")]
def cat(p):
    if "MultiContact" in p: return "MultiContact"
    if "LongWalk" in p: return "LongWalk"
    if "SLIPPAGE" in p.upper(): return "RHPS1 slip"
    return "RHPS1 walk"
MET={"trans_xy":"txy","yaw":"yaw","tilt":"tilt"}
def load(pref):
    d=glob.glob(f"results/{pref}-*/summary.csv")
    if not d: return None
    acc=collections.defaultdict(lambda: collections.defaultdict(list))
    for f in d:
        for r in csv.DictReader(open(f)):
            if r["statistic"]!="rmse" or r["metric"] not in MET: continue
            acc[(cat(r["project"]),MET[r["metric"]])][r["estimator"]].append((r["project"],float(r["value"])))
    out={}
    for k,est in acc.items():
        names=["Kinetics"] if "Kinetics" in est else []; ri=["RI-EKF"] if "RI-EKF" in est else []
        if not names or not ri: continue
        ko=dict(est[names[0]]); rk=dict(est[ri[0]]); common=sorted(set(ko)&set(rk))
        if common: out[k]=(np.mean([ko[p] for p in common]),np.mean([rk[p] for p in common]))
    return out
res={lab:load(p) for lab,p in RUNS}
cats=["MultiContact","RHPS1 walk","RHPS1 slip"]
for met,unit,sc in (("txy","mm",1000),("yaw","deg",1),("tilt","deg",1)):
    print(f"\n--- {met} [{unit}]   KO / RI-EKF   (ratio)")
    print(f"{'run':26s}"+"".join(f"{c:>26s}" for c in cats))
    for lab,_ in RUNS:
        r=res.get(lab)
        if not r: print(f"{lab:26s}  (no results)"); continue
        row=f"{lab:26s}"
        for c in cats:
            v=r.get((c,met))
            row+= f"{v[0]*sc:9.2f}/{v[1]*sc:7.2f}({v[0]/v[1]:5.3f})" if v else f"{'-':>26s}"
        print(row)

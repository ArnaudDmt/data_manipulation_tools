"""Slippage degradation ratio: (KO_slip/KO_walk) / (RI_slip/RI_walk), RHPS1 only.
<1 means the KO degrades less than the baseline; the target is <=0.90."""
import csv,glob,collections,numpy as np
MET={"trans_xy":"trans_xy","yaw":"yaw","tilt":"tilt"}
def grp(p):
    if "MultiContact" in p or "LongWalk" in p: return None
    return "slip" if "SLIPPAGE" in p.upper() else "walk"
def load(pref):
    d=glob.glob(f"results/{pref}-*/summary.csv")
    if not d: return None
    acc=collections.defaultdict(list)
    for f in d:
        for r in csv.DictReader(open(f)):
            g=grp(r["project"])
            if g is None or r["statistic"]!="rmse" or r["metric"] not in MET: continue
            acc[(g,r["metric"],r["estimator"])].append(float(r["value"]))
    return {k:float(np.mean(v)) for k,v in acc.items()}
for lab,pref in (("baseline","PW_base"),("wrench calibration","PW_cal")):
    a=load(pref)
    if not a: print(f"{lab}: no results"); continue
    print(f"\n=== {lab}")
    print(f"  {'metric':10s}{'KO walk':>10s}{'KO slip':>10s}{'KO degr':>10s}"
          f"{'RI walk':>10s}{'RI slip':>10s}{'RI degr':>10s}{'ratio':>9s}")
    for m in MET:
        kw,ks=a[("walk",m,"Kinetics")],a[("slip",m,"Kinetics")]
        rw,rs=a[("walk",m,"RI-EKF")],a[("slip",m,"RI-EKF")]
        kd,rd=ks/kw,rs/rw; sc=1000 if m=="trans_xy" else 1
        u="mm" if m=="trans_xy" else "deg"
        print(f"  {m:10s}{kw*sc:10.3f}{ks*sc:10.3f}{kd:10.4f}{rw*sc:10.3f}{rs*sc:10.3f}{rd:10.4f}{kd/rd:9.4f}"
              f"   {'PASS' if kd/rd<=0.90 else 'fail'}  [{u}]")

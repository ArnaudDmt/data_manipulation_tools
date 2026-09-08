import csv,glob,collections,numpy as np,sys
CAT=lambda p: ("MultiContact" if "MultiContact" in p else "LongWalk" if "LongWalk" in p
               else "RHPS1 slip" if "SLIPPAGE" in p.upper() else "RHPS1 walk")
def load(d):
    acc=collections.defaultdict(lambda: collections.defaultdict(list))
    for f in glob.glob(f"{d}/summary.csv"):
        for r in csv.DictReader(open(f)):
            if r["statistic"]!="mean" or r["metric"] not in ("trans_xy","trans_z","yaw","tilt"): continue
            acc[(CAT(r["project"]),r["metric"])][r["estimator"]].append((r["project"],float(r["value"])))
    out={}
    for k,d2 in acc.items():
        if "Kinetics" not in d2 or "RI-EKF" not in d2: continue
        ko=dict(d2["Kinetics"]); ri=dict(d2["RI-EKF"]); c=sorted(set(ko)&set(ri))
        if c: out[k]=(np.mean([ko[p] for p in c]),np.mean([ri[p] for p in c]))
    return out
b=load("results/kinetics-retuning/best-base"); w=load("results/kinetics-retuning/best-best")
if not b or not w:
    print("  summary.csv not found in those dirs; contents:")
    for d in ("best-base","best-best"):
        print("   ",d,sorted(x.split('/')[-1] for x in glob.glob(f"results/kinetics-retuning/{d}/*"))[:6])
    sys.exit()
cats=["MultiContact","RHPS1 walk","RHPS1 slip"]
for m in ("trans_xy","yaw","trans_z","tilt"):
    print(f"\n--- {m}   (KO/RI ratio)   baseline -> winner")
    for c in cats:
        if (c,m) in b and (c,m) in w:
            rb=b[(c,m)][0]/b[(c,m)][1]; rw=w[(c,m)][0]/w[(c,m)][1]
            print(f"  {c:14s} {rb:6.3f} -> {rw:6.3f}   {'better' if rw<rb else 'worse '}  ({(rw/rb-1)*100:+5.1f}%)")

import csv,glob,os,collections,numpy as np
def grp(p):
    if "LongWalk" in p: return "LongWalk"
    if "MultiContact" in p: return "MC"
    return "RHPS1slip" if "SLIPPAGE" in p.upper() else "RHPS1walk"
rows=[]
for f in sorted(glob.glob("results/*/summary.csv"), key=os.path.getmtime):
    acc=collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(f)):
        if r["statistic"]!="rmse" or r["metric"]!="yaw": continue
        acc[grp(r["project"])][r["estimator"]].append(float(r["value"]))
    if not acc: continue
    lab=os.path.basename(os.path.dirname(f)).rsplit("-",1)[0]
    cell={}
    for g,d in acc.items():
        if "Kinetics" in d and "RI-EKF" in d:
            cell[g]=(np.mean(d["Kinetics"]),np.mean(d["RI-EKF"]),len(d["Kinetics"]))
    rows.append((lab,cell))
print(f"{'run':26s}{'RHPS1 walk KO/RI':>26s}{'RHPS1 slip':>22s}{'LongWalk':>22s}")
for lab,c in rows[-22:]:
    def fmt(g):
        if g not in c: return f"{'-':>22s}"
        k,r,n=c[g]; return f"{k:7.3f}/{r:6.3f}({k/r:5.3f})n{n}"
    print(f"{lab:26s}{fmt('RHPS1walk'):>26s}{fmt('RHPS1slip'):>22s}{fmt('LongWalk'):>22s}")

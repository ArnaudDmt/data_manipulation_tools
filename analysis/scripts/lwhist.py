import csv,glob,os,collections,numpy as np
rows=[]
for f in sorted(glob.glob("results/*/summary.csv"), key=os.path.getmtime):
    d=collections.defaultdict(dict)
    for r in csv.DictReader(open(f)):
        if r["statistic"]!="rmse" or "LongWalk" not in r["project"]: continue
        d[r["metric"]][r["estimator"]]=float(r["value"])
    if not d: continue
    lab=os.path.basename(os.path.dirname(f)).rsplit("-",1)[0]
    out=[]
    for m in ("trans_xy","trans_z","yaw","tilt"):
        if m in d and "Kinetics" in d[m] and "RI-EKF" in d[m]:
            k,r=d[m]["Kinetics"],d[m]["RI-EKF"]
            out.append(f"{m} {k*1000:7.1f}/{r*1000:7.1f}({k/r:5.3f})" if m.startswith("trans")
                       else f"{m} {k:6.3f}/{r:6.3f}({k/r:5.3f})")
    print(f"{lab:24s} " + "  ".join(out))

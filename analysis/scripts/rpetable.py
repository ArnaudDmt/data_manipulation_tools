import csv,glob,collections,numpy as np
ORDER=["HRP5_MultiContact_1","HRP5_MultiContact_2","HRP5_MultiContact_3","HRP5_MultiContact_4",
 "KO_TRO2024_RHPS1_1","KO_TRO2024_RHPS1_2","KO_TRO2024_RHPS1_3","KO_TRO2024_RHPS1_4","KO_TRO2024_RHPS1_5",
 "KO_TRO_2024_RHPS1_SLIPPAGE_1","KO_TRO_2024_RHPS1_SLIPPAGE_2","KO_TRO_2024_RHPS1_SLIPPAGE_3"]
def cat(p):
    if "MultiContact" in p: return "MultiContact (0.3 m)"
    return "RHPS1 slippage (1 m)" if "SLIPPAGE" in p.upper() else "RHPS1 walk (1 m)"
def load(pref):
    v={}
    for f in glob.glob(f"results/{pref}-*/summary.csv"):
        for r in csv.DictReader(open(f)):
            if r["statistic"]!="rmse" or r["metric"] not in ("trans_xy","yaw"): continue
            v[(r["project"],r["metric"],r["estimator"])]=float(r["value"])
    return v
for lab,pref in (("BASELINE","PW_base"),("WITH WRENCH CALIBRATION","PW_cal")):
    v=load(pref)
    print(f"\n########## {lab}")
    print(f"{'dataset':32s}{'txy KO':>10s}{'txy RI':>10s}{'yaw KO':>10s}{'yaw RI':>10s}")
    agg=collections.defaultdict(lambda: collections.defaultdict(list))
    for p in ORDER:
        try:
            tk,tr=v[(p,"trans_xy","Kinetics")]*1000,v[(p,"trans_xy","RI-EKF")]*1000
            yk,yr=v[(p,"yaw","Kinetics")],v[(p,"yaw","RI-EKF")]
        except KeyError: continue
        print(f"{p:32s}{tk:10.3f}{tr:10.3f}{yk:10.3f}{yr:10.3f}")
        for k,val in (("tk",tk),("tr",tr),("yk",yk),("yr",yr)): agg[cat(p)][k].append(val)
    print()
    for c,d in agg.items():
        print(f"{c:32s}{np.mean(d['tk']):10.3f}{np.mean(d['tr']):10.3f}"
              f"{np.mean(d['yk']):10.3f}{np.mean(d['yr']):10.3f}")

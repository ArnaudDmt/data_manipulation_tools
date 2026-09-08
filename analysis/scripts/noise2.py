import sys, numpy as np, os; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
def hf_unloaded(log, sensor, ch, thr=10.0):
    fz=np.asarray(log[f"{sensor}_fz"],float)
    v=np.asarray(log[f"{sensor}_{ch}"],float)
    d=np.diff(v)/np.sqrt(2.0)                 # white part only: rejects slow gravity variation
    m=(np.abs(fz[1:])<thr)&(np.abs(fz[:-1])<thr)   # unloaded: rejects impacts and contact dynamics
    return (np.std(d[m]), int(m.sum())) if m.sum()>200 else (float('nan'),int(m.sum()))
jobs=[("RHPS1","KO_TRO2024_RHPS1_1",["RightFootForceSensor","LeftFootForceSensor"]),
      ("HRP5P","HRP5_MultiContact_1",["RightFootForceSensor","LeftFootForceSensor","LeftHandForceSensor"])]
print("sensor noise: white component, unloaded only (N and N.m)\n")
print(f"{'robot':7s} {'sensor':22s} " + "".join(f"{c:>9}" for c in ("fx","fy","fz","cx","cy","cz")))
for robot,proj,sensors in jobs:
    log=read_log(f"Projects/{proj}/output_data/kinetics_eval/logReplay_full.bin")
    for s in sensors:
        try: vals=[hf_unloaded(log,s,c)[0] for c in ("fx","fy","fz","cx","cy","cz")]
        except KeyError: continue
        print(f"{robot:7s} {s:22s} " + "".join(f"{v:9.4f}" for v in vals))
    del log

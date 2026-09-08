"""Sensor noise FLOOR: std within 0.2 s windows while the sensor is unloaded, taking the 10th
percentile over windows. Swing-phase inertia inflates the mean; the floor is the electrical
noise the measurement model is meant to describe."""
import sys, numpy as np; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
SETS=[

      ("hrp5_p",["HRP5_MultiContact_1","HRP5_MultiContact_2","HRP5_MultiContact_3","HRP5_MultiContact_4"],
       ["LeftFootForceSensor","RightFootForceSensor","LeftHandForceSensor"])]
for robot,projs,sensors in SETS:
    acc={}
    for p in projs:
        try: log=read_log(f"Projects/{p}/output_data/kinetics_eval/logReplay_full.bin")
        except Exception as e: print(f"  {p}: {type(e).__name__}"); continue
        dt=float(np.median(np.diff(np.asarray(log["t"],float)))); W=max(20,int(0.2/dt))
        for s in sensors:
            if f"{s}_fz" not in log: continue
            v={k:np.asarray(log[f"{s}_{k}"],float) for k in ("fx","fy","fz","cx","cy","cz")}
            fin=np.all([np.isfinite(v[k]) for k in v],axis=0)
            idx=np.flatnonzero(fin&(np.abs(v["fz"])<15))
            if len(idx)<5*W: continue
            # contiguous unloaded runs, split into windows
            brk=np.flatnonzero(np.diff(idx)>1); runs=np.split(idx,brk+1)
            for k in v:
                sds=[]
                for r in runs:
                    for a in range(0,len(r)-W,W):
                        sds.append(np.std(v[k][r[a:a+W]]))
                if len(sds)<10: continue
                acc.setdefault((s,k),[]).append(np.percentile(sds,10))
        del log
    print(f"\n=== {robot}     noise floor [std]     force N, moment N.m")
    print(f"{'sensor':22s}{'axis':6s}{'floor std':>11s}{'variance':>13s}")
    for (s,k),vals in sorted(acc.items()):
        f=float(np.mean(vals)); print(f"{s:22s}{k:6s}{f:11.4f}{f**2:13.3e}")

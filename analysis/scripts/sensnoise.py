"""Force/torque sensor noise, measured on the raw sensor while unloaded (|fz| < 15 N).
Reported as std. 'raw' includes slow drift; 'hp' is high-passed (deviation from a 1 s
moving median) and is the white-noise part the measurement model actually describes."""
import sys, numpy as np; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
from scipy.ndimage import median_filter
SETS=[("rhps1",["KO_TRO2024_RHPS1_1","KO_TRO2024_RHPS1_3","KO_TRO2024_RHPS1_5"],
       ["LeftFootForceSensor","RightFootForceSensor"]),
      ("hrp5_p",["HRP5_MultiContact_1","HRP5_MultiContact_3"],
       ["LeftFootForceSensor","RightFootForceSensor","LeftHandForceSensor"])]
for robot,projs,sensors in SETS:
    acc={}
    for p in projs:
        try: log=read_log(f"Projects/{p}/output_data/kinetics_eval/logReplay_full.bin")
        except Exception as e: print(f"{p}: {type(e).__name__}"); continue
        for s in sensors:
            if f"{s}_fz" not in log: continue
            v={k:np.asarray(log[f"{s}_{k}"],float) for k in ("fx","fy","fz","cx","cy","cz")}
            fin=np.all([np.isfinite(v[k]) for k in v],axis=0)
            m=fin&(np.abs(v["fz"])<15)
            if m.sum()<2000: continue
            for k in v:
                x=v[k][m]
                hp=x-median_filter(x,size=min(501,len(x)//4*2+1),mode="nearest")
                acc.setdefault((s,k),[[],[]])
                acc[(s,k)][0].append(np.std(x)); acc[(s,k)][1].append(np.std(hp))
        del log
    print(f"\n=== {robot}     unloaded sensor noise [std]      force N, moment N.m")
    print(f"{'sensor':22s}{'axis':6s}{'raw std':>10s}{'high-passed std':>18s}{'-> variance':>14s}")
    for (s,k),(raw,hp) in sorted(acc.items()):
        r=float(np.mean(raw)); h=float(np.mean(hp))
        print(f"{s:22s}{k:6s}{r:10.4f}{h:18.4f}{h**2:14.3e}")

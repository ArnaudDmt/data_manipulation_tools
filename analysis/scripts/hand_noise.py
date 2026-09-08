import sys, numpy as np, os; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
log=read_log("Projects/HRP5_MultiContact_1/output_data/kinetics_eval/logReplay_full.bin")
sensors=sorted({k.rsplit("_",1)[0] for k in log.keys() if "ForceSensor_" in k and k.endswith(("_fx","_cx"))})
print(f"{'sensor':26s} {'chan':5s} {'unloaded sd':>12} {'n':>8}")
for s in sensors:
    try: fz=np.asarray(log[f"{s}_fz"],float)
    except KeyError: continue
    idle = np.abs(fz) < 10.0
    if idle.sum() < 200: 
        print(f"{s:26s} (never unloaded, n={idle.sum()})"); continue
    for ch in ("fx","fy","fz","cx","cy","cz"):
        try: v=np.asarray(log[f"{s}_{ch}"],float)
        except KeyError: continue
        print(f"{s:26s} {ch:5s} {np.std(v[idle]):12.4f} {idle.sum():8d}")

"""The clean test: what does the raw foot sensor read while the foot is IN THE AIR?
An unloaded sensor must read zero. Anything else is an offset, and is subtractable."""
import sys, numpy as np; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
SETS=[("rhps1",["KO_TRO2024_RHPS1_1","KO_TRO2024_RHPS1_2","KO_TRO2024_RHPS1_3",
                "KO_TRO2024_RHPS1_4","KO_TRO2024_RHPS1_5"]),
      ("hrp5_p",["HRP5_MultiContact_1","HRP5_MultiContact_2"])]
for robot,projs in SETS:
    print(f"\n=== {robot}    raw sensor, unloaded (fz < 15 N)      [force N, moment N.m]")
    print(f"{'dataset':30s}{'sensor':22s}{'n':>7s}{'fx':>8s}{'fy':>8s}{'fz':>8s}"
          f"{'cx':>9s}{'cy':>9s}{'cz':>9s}")
    for p in projs:
        try: log=read_log(f"Projects/{p}/output_data/kinetics_eval/logReplay_full.bin")
        except Exception as e: print(f"{p:30s} -- {type(e).__name__}"); continue
        for s in ("LeftFootForceSensor","RightFootForceSensor"):
            if f"{s}_fz" not in log: continue
            v={k:np.asarray(log[f"{s}_{k}"],float) for k in ("fx","fy","fz","cx","cy","cz")}
            fin=np.all([np.isfinite(v[k]) for k in v],axis=0)
            m=fin&(np.abs(v["fz"])<15)
            if m.sum()<200: print(f"{p:30s}{s:22s}{m.sum():7d}   never unloaded"); continue
            print(f"{p:30s}{s:22s}{m.sum():7d}"+"".join(f"{np.median(v[k][m]):8.3f}" for k in ("fx","fy","fz"))
                  +"".join(f"{np.median(v[k][m]):9.3f}" for k in ("cx","cy","cz")))
        del log

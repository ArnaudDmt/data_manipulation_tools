import sys, numpy as np, os; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
runs=[("RHPS1  ","KO_TRO2024_RHPS1_1"),("HRP5P-MC","HRP5_MultiContact_1")]
print("force/torque sensor noise, measured two ways")
print("  unloaded: std while the foot carries no load (fz < 10 N)")
print("  hf: std of consecutive differences / sqrt(2), i.e. the white part only\n")
print(f"{'run':10s} {'foot':10s} {'chan':4s} {'unloaded sd':>12} {'hf sd':>10} {'n_unloaded':>11}")
for lab,name in runs:
    p=f"Projects/{name}/output_data/kinetics_eval/logReplay_full.bin"
    if not os.path.exists(p): print(f"{lab}: no log"); continue
    log=read_log(p)
    for foot in ("RightFoot","LeftFoot"):
        try:
            fz=np.asarray(log[f"{foot}ForceSensor_fz"],float)
        except KeyError: continue
        idle = fz < 10.0
        for ch in ("cx","cy","cz","fx","fy","fz"):
            try: v=np.asarray(log[f"{foot}ForceSensor_{ch}"],float)
            except KeyError: continue
            hf=np.std(np.diff(v))/np.sqrt(2.0)
            un=np.std(v[idle]) if idle.sum()>200 else float('nan')
            print(f"{lab:10s} {foot:10s} {ch:4s} {un:12.4f} {hf:10.4f} {idle.sum():11d}")
    del log
print("\nconfig says: forceSensorVariance 1e0 -> sigma 1 N ; torqueSensorVariance 9e-4 -> sigma 0.03 N.m")

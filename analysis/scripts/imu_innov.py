import sys, numpy as np, os; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
P="Observers_MainObserverPipeline_MCKineticsObserver_MEKF_"
runs=[("nom ",f"KO_TRO2024_RHPS1_{i}") for i in (1,3)]+\
     [("SLIP",f"KO_TRO_2024_RHPS1_SLIPPAGE_{i}") for i in (1,2,3)]
def v3(log,name):
    return np.array([np.asarray(log[f"{P}innovation_{name}_{a}"],float) for a in "xyz"])
print("state corrections per step, median while loaded")
print("If the accelerometer disagreed with the contact-driven base motion, the correction applied")
print("to linear velocity would rise during slip alongside the position correction.\n")
print(f"{'run':30s} {'base pos (mm)':>14} {'linVel (mm/s)':>14} {'gyroBias (rad/s)':>18} {'ratio v/p':>10}")
for lab,name in runs:
    p=f"Projects/{name}/output_data/kinetics_eval/logReplay_full.bin"
    if not os.path.exists(p): continue
    log=read_log(p)
    pos=np.linalg.norm(v3(log,"positionW_")[:2],axis=0)*1000.0
    vel=np.linalg.norm(v3(log,"linVelW_")[:2],axis=0)*1000.0
    gb=np.linalg.norm(v3(log,"gyroBias_Accelerometer"),axis=0)
    m=np.isfinite(pos)&np.isfinite(vel)&(pos>0)
    p50,v50,g50=np.median(pos[m]),np.median(vel[m]),np.median(gb[m])
    print(f"{lab} {name:25s} {p50:14.4f} {v50:14.4f} {g50:18.2e} {v50/max(p50,1e-9):10.2f}")
    del log
print("DONE")

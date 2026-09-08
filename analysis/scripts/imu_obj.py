import sys, numpy as np, os; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
M="Observers_MainObserverPipeline_MCKineticsObserver_MEKF_measurements_"
runs=[("nom ",f"KO_TRO2024_RHPS1_{i}") for i in (1,3,5)]+\
     [("SLIP",f"KO_TRO_2024_RHPS1_SLIPPAGE_{i}") for i in (1,2,3)]
def innov(log, sensor):
    meas=np.array([np.asarray(log[f"{M}{sensor}_Accelerometer_measured_{a}"],float) for a in "xyz"])
    pred=np.array([np.asarray(log[f"{M}{sensor}_Accelerometer_predicted_{a}"],float) for a in "xyz"])
    return np.linalg.norm(meas-pred,axis=0)
print("IMU measurement innovation |measured - predicted|, median over the run")
print("accelerometer in m/s^2, gyro in rad/s\n")
print(f"{'run':30s} {'accelero':>12} {'gyro':>12} {'contact force (N)':>19}")
for lab,name in runs:
    p=f"Projects/{name}/output_data/kinetics_eval/logReplay_full.bin"
    if not os.path.exists(p): continue
    log=read_log(p)
    a=innov(log,"accelerometer"); g=innov(log,"gyro")
    # contact wrench innovation for scale
    cf=[]
    for c in ("LeftFootCenter","RightFootCenter"):
        try:
            ms=np.array([np.asarray(log[f"{M}contacts_force_{c}_measured_{x}"],float) for x in "xyz"])
            pr=np.array([np.asarray(log[f"{M}contacts_force_{c}_predicted_{x}"],float) for x in "xyz"])
            cf.append(np.linalg.norm(ms-pr,axis=0))
        except KeyError: pass
    cfm=np.median(np.max(cf,axis=0)) if cf else float('nan')
    print(f"{lab} {name:25s} {np.median(a):12.4f} {np.median(g):12.5f} {cfm:19.2f}")
    del log
print("DONE")

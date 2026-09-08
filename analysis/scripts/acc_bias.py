import sys, numpy as np, os; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
M="Observers_MainObserverPipeline_MCKineticsObserver_MEKF_measurements_accelerometer_Accelerometer_"
for lab,name in [("nom ","KO_TRO2024_RHPS1_1"),("SLIP","KO_TRO_2024_RHPS1_SLIPPAGE_1")]:
    p=f"Projects/{name}/output_data/kinetics_eval/logReplay_full.bin"
    if not os.path.exists(p): continue
    log=read_log(p)
    meas=np.array([np.asarray(log[f"{M}measured_{a}"],float) for a in "xyz"])
    pred=np.array([np.asarray(log[f"{M}predicted_{a}"],float) for a in "xyz"])
    d=meas-pred
    print(f"{lab} {name}")
    print(f"   |measured| median {np.median(np.linalg.norm(meas,axis=0)):.3f} m/s^2")
    for i,a in enumerate("xyz"):
        print(f"   axis {a}: innovation mean {np.mean(d[i]):+8.4f}  std {np.std(d[i]):7.4f}"
              f"   -> {'BIAS dominates' if abs(np.mean(d[i]))>np.std(d[i]) else 'noise dominates'}")
    print()
    del log

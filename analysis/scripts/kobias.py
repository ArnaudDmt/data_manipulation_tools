import sys, numpy as np; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
B="Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_gyroBias_Accelerometer"
for n in ("KO_TRO2024_RHPS1_1","KO_TRO_2024_RHPS1_SLIPPAGE_1"):
    log=read_log(f"Projects/{n}/output_data/kinetics_eval/logReplay_full.bin")
    b=np.array([np.asarray(log[f"{B}_{a}"],float) for a in "xyz"]).T
    print(f"{n}: bias start {b[0]*1e6} urad/s   end {b[-1]*1e6}   "
          f"total excursion {np.linalg.norm(b[-1]-b[0])*1e6:.3f} urad/s   "
          f"max|b| {np.max(np.linalg.norm(b,axis=1))*1e6:.3f}")
    del log

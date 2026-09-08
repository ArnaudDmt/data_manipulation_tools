import sys, numpy as np; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
P="Observers_MainObserverPipeline_MCKineticsObserver_"; B=P+"MEKF_"; V=P+"MCKineticsObserverBackupValinor_"
def vec(log,b,c="xyz"): return np.array([np.asarray(log[f"{b}_{k}"],float) for k in c]).T
log=read_log("Projects/KO_TRO_2024_RHPS1_SLIPPAGE_1/output_data/kinetics_eval/logReplay_full.bin")
c="LeftFootCenter"
S=np.array([str(s).strip().lower().startswith("set") for s in log[P+f"debug_contactState_isSet_{c}"]])
for lab,key in (("prediction restPos_W",B+f"prediction_contact_{c}_restPos_W"),
                ("Valinor currentWorldContactKine",V+f"contacts_{c}_currentWorldContactKine_position"),
                ("Valinor refPose",V+f"contacts_{c}_refPose_position"),
                ("state contact position",B+f"estimatedState_contact_{c}_position")):
    if f"{key}_x" not in log: print(f"  {lab:34s} ABSENT"); continue
    a=vec(log,key); fin=np.isfinite(a).all(1)
    print(f"  {lab:34s} finite {fin.sum():7d}/{len(a)}   finite&set {(fin&S).sum():7d}"
          f"   median xy {np.median(np.linalg.norm(a[fin&S][:,:2],axis=1)) if (fin&S).sum() else float('nan'):8.3f}")

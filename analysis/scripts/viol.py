import sys, numpy as np, os; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
P="Observers_MainObserverPipeline_MCKineticsObserver_MEKF_prediction_contact_"
E="Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_contact_"
projects=[f"HRP5_MultiContact_{i}" for i in (1,2,3,4)]+["HRP5P_LongWalk"]+\
         [f"KO_TRO2024_RHPS1_{i}" for i in (1,3,5)]
print(f"{'dataset':24s} {'contact':22s} {'p50':>8} {'p90':>8}", flush=True)
for name in projects:
    p=f"Projects/{name}/output_data/kinetics_eval/logReplay_full.bin"
    if not os.path.exists(p):
        print(f"{name:24s} (no full log)", flush=True); continue
    try: log=read_log(p)
    except Exception as e:
        print(f"{name:24s} {e}", flush=True); continue
    contacts=sorted({k.split("_contact_")[1].split("_forces_")[0]
                     for k in log.keys() if "_contact_" in k and k.endswith("_forces_x")})
    for c in contacts:
        try:
            pf=np.array([np.asarray(log[f"{P}{c}_forces_{a}"],float) for a in "xyz"])
            ef=np.array([np.asarray(log[f"{E}{c}_forces_{a}"],float) for a in "xyz"])
        except KeyError: continue
        d=np.linalg.norm(ef-pf,axis=0)
        m=np.isfinite(d)&(np.linalg.norm(ef,axis=0)>20)
        if m.sum()<100: continue
        q=np.percentile(d[m],[50,90])
        print(f"{name:24s} {c:22s} {q[0]:8.1f} {q[1]:8.1f}", flush=True)
    del log
print("DONE", flush=True)

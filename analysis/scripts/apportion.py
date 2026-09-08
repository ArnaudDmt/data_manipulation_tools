import sys, numpy as np, os; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
I="Observers_MainObserverPipeline_MCKineticsObserver_MEKF_innovation_"
runs=[("nom ",f"KO_TRO2024_RHPS1_{i}") for i in (1,3,5)]+\
     [("SLIP",f"KO_TRO_2024_RHPS1_SLIPPAGE_{i}") for i in (1,2,3)]
def v3(log,name):
    return np.array([np.asarray(log[f"{I}{name}_{a}"],float) for a in "xyz"])
print("per-step state correction, horizontal magnitude, median over loaded samples (mm)")
print(f"{'run':28s} {'base pos':>10} {'restpose':>10} {'ratio base/rest':>16}  {'unmodF (N)':>11}")
for lab,name in runs:
    p=f"Projects/{name}/output_data/kinetics_eval/logReplay_full.bin"
    if not os.path.exists(p): continue
    log=read_log(p)
    base=v3(log,"positionW_")
    uf=v3(log,"unmodeledForce_")
    rest=[]
    for foot in ("LeftFootCenter","RightFootCenter"):
        try: rest.append(v3(log,f"contacts_{foot}_position"))
        except KeyError: pass
    bxy=np.hypot(base[0],base[1])*1000.0
    rxy=np.max([np.hypot(r[0],r[1]) for r in rest],axis=0)*1000.0 if rest else np.zeros_like(bxy)
    ufn=np.linalg.norm(uf,axis=0)
    m=np.isfinite(bxy)&np.isfinite(rxy)&(bxy+rxy>0)
    b50=np.median(bxy[m]); r50=np.median(rxy[m])
    print(f"{lab} {name:23s} {b50:10.4f} {r50:10.4f} {b50/max(r50,1e-9):16.1f}  {np.median(ufn[m]):11.2f}")
    del log
print("DONE")

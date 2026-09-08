"""Zero offset or load-proportional leakage? Compare unloaded (swing) vs loaded tangential force."""
import sys, numpy as np; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
P="Observers_MainObserverPipeline_MCKineticsObserver_"; B=P+"MEKF_"
def vec(log,b,c="xyz"): return np.array([np.asarray(log[f"{b}_{k}"],float) for k in c]).T
for name in ("KO_TRO2024_RHPS1_1","KO_TRO_2024_RHPS1_SLIPPAGE_1","HRP5_MultiContact_1"):
    try: log=read_log(f"Projects/{name}/output_data/kinetics_eval/logReplay_full.bin")
    except Exception as e: print(f"{name}: {e}"); continue
    print(f"\n=== {name}")
    for c in ("LeftFootCenter","RightFootCenter"):
        k=B+f"measurements_contacts_force_{c}_measured"
        if f"{k}_x" not in log: print(f"  {c}: absent"); continue
        f=vec(log,k)
        setf=np.array([str(s).strip().lower().startswith("set") for s in log[P+f"debug_contactState_isSet_{c}"]])
        # the 'measured' channel is only meaningful while the contact is set; use raw sensor if present
        fz=f[:,2]; loaded=setf&(fz>150); light=setf&(fz<60)
        print(f"  {c}:  set {setf.sum():6d}   loaded(fz>150N) {loaded.sum():6d}   lightly-loaded(fz<60N) {light.sum():6d}")
        for lab,msk in (("loaded  ",loaded),("light   ",light)):
            if msk.sum()<100: print(f"    {lab} too few"); continue
            a=f[msk]; tan=np.linalg.norm(a[:,:2],axis=1)
            print(f"    {lab} fx {np.median(a[:,0]):7.1f}  fy {np.median(a[:,1]):7.1f}  fz {np.median(a[:,2]):7.1f}"
                  f"   |tan| {np.median(tan):6.1f}   |tan|/fz {np.median(tan/np.maximum(a[:,2],1e-6)):6.3f}")
    del log

"""Does the anchor keep up? Tangential spring deflection |rest - actual contact| in world.
K_tangential = 3e4 N/m, so a 285 N normal load at ratio 0.8 needs 228 N -> 7.6 mm of stretch."""
import sys, numpy as np; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
P="Observers_MainObserverPipeline_MCKineticsObserver_"; B=P+"MEKF_"; V=P+"MCKineticsObserverBackupValinor_"
def vec(log,b,c="xyz"): return np.array([np.asarray(log[f"{b}_{k}"],float) for k in c]).T
for name in ("KO_TRO2024_RHPS1_1","KO_TRO_2024_RHPS1_SLIPPAGE_1","KO_TRO_2024_RHPS1_SLIPPAGE_2"):
    log=read_log(f"Projects/{name}/output_data/kinetics_eval/logReplay_full.bin")
    print(f"\n=== {name}")
    for c in ("LeftFootCenter","RightFootCenter"):
        S=np.array([str(s).strip().lower().startswith("set") for s in log[P+f"debug_contactState_isSet_{c}"]])
        fp=vec(log,B+f"measurements_contacts_force_{c}_predicted")
        rest=vec(log,B+f"prediction_contact_{c}_restPos_W")
        cur=vec(log,V+f"contacts_{c}_currentWorldContactKine_position")
        fin=np.isfinite(rest).all(1)&np.isfinite(cur).all(1)&np.isfinite(fp).all(1)
        ok=S&fin&(fp[:,2]>50)
        ratio=np.full(len(fp),np.nan); ratio[ok]=np.linalg.norm(fp[ok][:,:2],axis=1)/fp[ok][:,2]
        defl=np.linalg.norm((rest-cur)[:,:2],axis=1)*1000            # mm
        inside=ok&(ratio<=0.4); viol=ok&(ratio>0.8)
        if viol.sum()<200: print(f"  {c}: too few"); continue
        print(f"  {c:16s} inside-cone  deflection p50 {np.median(defl[inside]):6.2f} mm"
              f"  p90 {np.percentile(defl[inside],90):6.2f}  p99 {np.percentile(defl[inside],99):7.2f}")
        print(f"  {'':16s} OVER cone    deflection p50 {np.median(defl[viol]):6.2f} mm"
              f"  p90 {np.percentile(defl[viol],90):6.2f}  p99 {np.percentile(defl[viol],99):7.2f}"
              f"   max {defl[viol].max():7.1f}")
    del log

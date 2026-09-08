"""When the model demands an impossible tangential force, who absorbs the innovation --
the contact rest pose (slip) or the unmodeled wrench (a phantom external force)?
Also: does the anchor actually keep up, i.e. does the spring deflection stay bounded?"""
import sys, numpy as np; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
P="Observers_MainObserverPipeline_MCKineticsObserver_"; B=P+"MEKF_"; V=P+"MCKineticsObserverBackupValinor_"
def vec(log,b,c="xyz"): return np.array([np.asarray(log[f"{b}_{k}"],float) for k in c]).T
for name in ("KO_TRO2024_RHPS1_1","KO_TRO_2024_RHPS1_SLIPPAGE_1","KO_TRO_2024_RHPS1_SLIPPAGE_2"):
    log=read_log(f"Projects/{name}/output_data/kinetics_eval/logReplay_full.bin")
    uf=np.linalg.norm(vec(log,B+"innovation_unmodeledForce_"),axis=1)
    print(f"\n=== {name}")
    for c in ("LeftFootCenter","RightFootCenter"):
        S=np.array([str(s).strip().lower().startswith("set") for s in log[P+f"debug_contactState_isSet_{c}"]])
        fp=vec(log,B+f"measurements_contacts_force_{c}_predicted")
        ok=S&(fp[:,2]>50)
        ratio=np.full(len(fp),np.nan); ratio[ok]=np.linalg.norm(fp[ok][:,:2],axis=1)/fp[ok][:,2]
        anch=np.linalg.norm(vec(log,B+f"innovation_contacts_{c}_position"),axis=1)
        rest=vec(log,B+f"prediction_contact_{c}_restPos_W"); cur=vec(log,V+f"contacts_{c}_currentWorldContactKine_position")
        defl=np.linalg.norm((cur-rest)[:,:2],axis=1)
        inside=ok&(ratio<=0.4); viol=ok&(ratio>0.8)
        if viol.sum()<200: print(f"  {c}: too few violating samples"); continue
        print(f"  {c}   inside-cone n={inside.sum():6d}   over-cone n={viol.sum():6d}")
        for lab,m in (("inside cone (<=0.4)",inside),("OVER cone  (>0.8)",viol)):
            print(f"    {lab:22s} anchor corr {np.median(anch[m])*1e6:8.2f} um"
                  f"   unmodeled-force corr {np.median(uf[m]):8.3f} N"
                  f"   tangential deflection {np.median(defl[m])*1000:7.2f} mm")
        r=lambda a: np.median(a[viol])/max(np.median(a[inside]),1e-12)
        print(f"    growth over-cone / inside:   anchor x{r(anch):5.2f}    unmodeled force x{r(uf):5.2f}"
              f"    deflection x{r(defl):5.2f}")
    del log

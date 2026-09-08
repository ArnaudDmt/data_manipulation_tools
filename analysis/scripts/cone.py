"""Does the visco-elastic model predict tangential forces friction cannot supply?
Measured force is calibration-corrected first, so the 18 deg sensor artifact is not mistaken
for a large friction ratio."""
import sys, numpy as np; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
P="Observers_MainObserverPipeline_MCKineticsObserver_"; B=P+"MEKF_"
CAL={"LeftFootCenter":np.array([0.047962,0.324858,0.015621]),
     "RightFootCenter":np.array([-0.282761,0.142506,0.001466])}
def rod(rv):
    a=np.linalg.norm(rv)
    if a<1e-12: return np.eye(3)
    k=rv/a; K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)+np.sin(a)*K+(1-np.cos(a))*K@K
def vec(log,b,c="xyz"): return np.array([np.asarray(log[f"{b}_{k}"],float) for k in c]).T
def ratios(f):
    fn=f[:,2]; ok=fn>50
    return np.linalg.norm(f[ok][:,:2],axis=1)/fn[ok]
for name in ("KO_TRO2024_RHPS1_1","KO_TRO_2024_RHPS1_SLIPPAGE_1","KO_TRO_2024_RHPS1_SLIPPAGE_2"):
    log=read_log(f"Projects/{name}/output_data/kinetics_eval/logReplay_full.bin")
    kind="walk" if "SLIPPAGE" not in name.upper() else "SLIPPAGE"
    print(f"\n=== {name}  ({kind})   tangential/normal force ratio")
    print(f"{'contact':18s}{'source':26s}{'p50':>8s}{'p90':>8s}{'p99':>8s}{'max':>8s}{'  frac>0.8':>10s}")
    for c in ("LeftFootCenter","RightFootCenter"):
        R=rod(CAL[c])
        for lab,key,corr in (("measured (calibrated)","measured",True),
                             ("visco-elastic predicted","predicted",False)):
            k=f"{B}measurements_contacts_force_{c}_{key}"
            if f"{k}_x" not in log: print(f"  {c:16s}{lab:26s}  absent"); continue
            f=vec(log,k)
            if corr: f=np.einsum('jk,nk->nj',R,f)
            r=ratios(f)
            if len(r)<200: print(f"  {c:16s}{lab:26s}  too few"); continue
            print(f"  {c:16s}{lab:26s}{np.percentile(r,50):8.3f}{np.percentile(r,90):8.3f}"
                  f"{np.percentile(r,99):8.3f}{r.max():8.2f}{np.mean(r>0.8)*100:9.1f}%")
    del log

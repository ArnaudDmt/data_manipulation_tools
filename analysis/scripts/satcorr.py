"""Do measured-force saturation episodes coincide with where the KO loses ground to the RI-EKF?
Windowed relative-translation error, after removing a single global yaw per estimator."""
import sys, csv, numpy as np; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
P="Observers_MainObserverPipeline_MCKineticsObserver_"; B=P+"MEKF_"
CAL={"LeftFootCenter":np.array([0.047962,0.324858,0.015621]),
     "RightFootCenter":np.array([-0.282761,0.142506,0.001466])}
def rod(rv):
    a=np.linalg.norm(rv); k=rv/a
    K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)+np.sin(a)*K+(1-np.cos(a))*K@K
def vec(log,b,c="xyz"): return np.array([np.asarray(log[f"{b}_{k}"],float) for k in c]).T
def cols(rows,idx,base):
    c=[f"{base}_{a}" for a in "xyz"]
    return np.array([[float(r[idx[k]]) if r[idx[k]] not in ("","nan") else np.nan for k in c] for r in rows])
W=500   # 1 s at 500 Hz
for name in ("KO_TRO_2024_RHPS1_SLIPPAGE_1","KO_TRO_2024_RHPS1_SLIPPAGE_2","KO_TRO_2024_RHPS1_SLIPPAGE_3"):
    with open(f"Projects/{name}/output_data/synchronizedObserversMocapData.csv") as fh:
        rd=csv.reader(fh,delimiter=';'); h=next(rd); idx={k:j for j,k in enumerate(h)}; rows=list(rd)
    gt=cols(rows,idx,"MocapAligner_worldBodyKine_position")
    ko=cols(rows,idx,"KO_position"); ri=cols(rows,idx,"HartleyIEKF_imuFbKine_position")
    log=read_log(f"Projects/{name}/output_data/kinetics_eval/logReplay_full.bin")
    n=min(len(gt),len(log["t"]))
    sat=np.zeros(n)
    for c in ("LeftFootCenter","RightFootCenter"):
        S=np.array([str(s).strip().lower().startswith("set") for s in log[P+f"debug_contactState_isSet_{c}"]])[:n]
        f=np.einsum('jk,nk->nj',rod(CAL[c]),vec(log,B+f"measurements_contacts_force_{c}_measured")[:n])
        good=S&np.isfinite(f).all(1)&(f[:,2]>50)
        r=np.zeros(n); r[good]=np.linalg.norm(f[good][:,:2],axis=1)/f[good][:,2]
        sat=np.maximum(sat,r)
    gt,ko,ri=gt[:n],ko[:n],ri[:n]
    starts=np.arange(0,n-W,W)
    ok=np.array([np.isfinite(gt[s:s+W]).all() and np.isfinite(ko[s:s+W]).all()
                 and np.isfinite(ri[s:s+W]).all() for s in starts])
    starts=starts[ok]
    dg=np.array([gt[s+W-1,:2]-gt[s,:2] for s in starts])
    out={}
    for lab,est in (("KO",ko),("RI",ri)):
        de=np.array([est[s+W-1,:2]-est[s,:2] for s in starts])
        # single global yaw aligning mocap increments to this estimator's
        num=np.sum(de[:,1]*dg[:,0]-de[:,0]*dg[:,1]); den=np.sum(de[:,0]*dg[:,0]+de[:,1]*dg[:,1])
        th=np.arctan2(num,den); R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
        out[lab]=np.linalg.norm(de-dg@R.T,axis=1)*1000     # mm per 1 s window
    s=np.array([sat[st:st+W].mean() for st in starts])
    hi=s>np.percentile(s,75); lo=s<np.percentile(s,25)
    print(f"\n=== {name}   {len(starts)} windows of 1 s")
    print(f"  measured tan/normal ratio: p50 {np.median(s):.3f}  p75 {np.percentile(s,75):.3f}"
          f"  p95 {np.percentile(s,95):.3f}  max {s.max():.3f}")
    for lab in ("KO","RI"):
        print(f"  {lab} rel-trans err  low-sat {np.median(out[lab][lo]):7.2f} mm"
              f"   high-sat {np.median(out[lab][hi]):7.2f} mm"
              f"   (x{np.median(out[lab][hi])/max(np.median(out[lab][lo]),1e-9):4.2f})")
    rl=np.median(out["KO"][lo])/max(np.median(out["RI"][lo]),1e-9)
    rh=np.median(out["KO"][hi])/max(np.median(out["RI"][hi]),1e-9)
    print(f"  KO/RI ratio:   low-sat {rl:5.3f}   high-sat {rh:5.3f}   "
          f"{'KO loses ground when saturated' if rh>rl else 'KO does NOT lose ground when saturated'}")
    print(f"  corr(sat, KO-RI err) = {np.corrcoef(s, out['KO']-out['RI'])[0,1]:+.3f}")
    del log

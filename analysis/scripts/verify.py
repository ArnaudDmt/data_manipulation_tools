"""Two independent checks of the wrench calibration claim.
 (1) Re-estimate the tilt using the MOCAP body orientation, not the estimator's -- removes the
     circularity of using the filter's own attitude to judge its input.
 (2) Centre of pressure, before and after the rotation: it must lie inside the sole. This tests
     the TORQUE, which the force-only fit never constrained."""
import sys, numpy as np, csv; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
P="Observers_MainObserverPipeline_MCKineticsObserver_"; B=P+"MEKF_"; MASS=58.32; G=9.81
def quat2R(q):
    q=q/np.linalg.norm(q,axis=1,keepdims=True); w,x,y,z=q.T
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]).transpose(2,0,1)
def vec(log,b,c="xyz"): return np.array([np.asarray(log[f"{b}_{k}"],float) for k in c]).T
def min_rot(u,v):
    u=u/np.linalg.norm(u); v=v/np.linalg.norm(v); ax=np.cross(u,v); s=np.linalg.norm(ax)
    return np.zeros(3) if s<1e-12 else ax/s*np.arctan2(s,float(np.dot(u,v)))
def rodrigues(rv):
    a=np.linalg.norm(rv)
    if a<1e-12: return np.eye(3)
    k=rv/a; K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)+np.sin(a)*K+(1-np.cos(a))*K@K
FEET=("LeftFootCenter","RightFootCenter")
for name in ("KO_TRO2024_RHPS1_1","HRP5_MultiContact_1"):
    log=read_log(f"Projects/{name}/output_data/kinetics_eval/logReplay_full.bin")
    n=len(log["t"]); Rest=quat2R(vec(log,B+"estimatedState_ori","wxyz"))
    with open(f"Projects/{name}/output_data/synchronizedObserversMocapData.csv") as fh:
        rd=csv.reader(fh,delimiter=';'); h=next(rd); i={k:j for j,k in enumerate(h)}
        cols=[f"MocapAligner_worldBodyKine_ori_{a}" for a in "wxyz"]
        q=np.array([[float(r[i[c]]) if r[i[c]] not in ("","nan") else np.nan for c in cols] for r in rd])
    m=min(n,len(q)); ok=np.isfinite(q[:m]).all(1)&(np.abs(np.linalg.norm(q[:m],axis=1)-1)<0.1)
    Rmoc=np.tile(np.eye(3),(m,1,1)); Rmoc[ok]=quat2R(q[:m][ok])
    print(f"\n=== {name}   mocap attitude valid on {ok.sum()}/{m} samples")
    for k,c in enumerate(FEET):
        Rcc=quat2R(vec(log,P+f"debug_contactKine_{c}_inputCentroidContactKine_orientation","wxyz"))[:m]
        f=vec(log,B+f"measurements_contacts_force_{c}_measured")[:m]
        t=vec(log,B+f"measurements_contacts_torque_{c}_measured")[:m] \
            if f"{B}measurements_contacts_torque_{c}_measured_x" in log else None
        S=lambda cc: np.array([str(s).strip().lower().startswith("set") for s in log[P+f"debug_contactState_isSet_{cc}"]])[:m]
        ss=np.flatnonzero(S(c)&(~S(FEET[1-k]))&(f[:,2]>0.7*MASS*G)&ok)
        if len(ss)<200: print(f"  {c}: too few"); continue
        u=(f[ss]/np.linalg.norm(f[ss],axis=1,keepdims=True)).sum(0)
        up=np.array([0,0,1.0])
        d_est=np.einsum('nji,j->ni',(Rest[:m]@Rcc)[ss],up).sum(0)
        d_moc=np.einsum('nji,j->ni',(Rmoc@Rcc)[ss],up).sum(0)
        a_est=np.degrees(np.linalg.norm(min_rot(u,d_est))); a_moc=np.degrees(np.linalg.norm(min_rot(u,d_moc)))
        print(f"  {c:18s} tilt from ESTIMATOR attitude {a_est:6.2f} deg    from MOCAP attitude {a_moc:6.2f} deg")
        if t is None: print("      (no measured torque channel)"); continue
        R=rodrigues(min_rot(u,d_moc))
        for lab,(ff,tt) in (("before",(f[ss],t[ss])),
                            ("after ",(np.einsum('jk,nk->nj',R,f[ss]),np.einsum('jk,nk->nj',R,t[ss])))):
            cx=-tt[:,1]/np.maximum(ff[:,2],1e-6); cy=tt[:,0]/np.maximum(ff[:,2],1e-6)
            print(f"      CoP {lab}: x median {np.median(cx)*1000:+7.1f} mm  [p5 {np.percentile(cx,5)*1000:+7.1f},"
                  f" p95 {np.percentile(cx,95)*1000:+7.1f}]   y median {np.median(cy)*1000:+7.1f} mm"
                  f"  [p5 {np.percentile(cy,5)*1000:+7.1f}, p95 {np.percentile(cy,95)*1000:+7.1f}]")
    del log

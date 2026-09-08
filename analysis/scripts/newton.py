"""Is the 137 N of 'external force' real? Newton says sum(contact forces) = m*(a + g).
Mocap gives a independently, so the true residual can be computed without the filter."""
import sys, numpy as np, csv; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
from scipy.spatial.transform import Rotation as R
B="Observers_MainObserverPipeline_MCKineticsObserver_MEKF_"
for name,mass in (("KO_TRO2024_RHPS1_1",58.32),("KO_TRO_2024_RHPS1_SLIPPAGE_1",58.32)):
    log=read_log(f"Projects/{name}/output_data/kinetics_eval/logReplay_full.bin")
    n=len(log["t"])
    # measured contact forces, rotated to world using the contact orientation from FK
    tot=np.zeros((3,n)); contact_frames=[]
    for foot in ("RightFoot","LeftFoot"):
        f=np.array([np.asarray(log[f"{foot}ForceSensor_{a}"],float) for a in ("fx","fy","fz")])
        # world<-contact from mocap body orientation composed with the FK body->contact transform,
        # so nothing here comes from the filter
        q=np.array([np.asarray(log[f"Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{foot}Center_fbContactKine_ori_{a}"],float) for a in ("x","y","z","w")]).T
        ok=np.isfinite(q).all(1)&(np.linalg.norm(q,axis=1)>1e-6)
        qq=np.where(ok[:,None],q,[0,0,0,1.0]); qq/=np.linalg.norm(qq,axis=1,keepdims=True)
        contact_frames.append((qq, f))
    # compose with the mocap body orientation
    with open(f"Projects/{name}/output_data/synchronizedObserversMocapData.csv") as fh0:
        rd0=csv.reader(fh0,delimiter=';'); h0=next(rd0); i0={k:i for i,k in enumerate(h0)}
        bq=[]
        for r in rd0:
            try: bq.append([float(r[i0[f"MocapAligner_worldBodyKine_ori_{a}"]]) for a in ("x","y","z","w")])
            except (ValueError,KeyError): bq.append([0,0,0,1.0])
    bq=np.array(bq); nb=min(len(bq),n)
    bqn=bq[:nb]; norms=np.linalg.norm(bqn,axis=1,keepdims=True)
    bqn=np.where(norms>1e-6, bqn/np.maximum(norms,1e-12), np.array([0,0,0,1.0]))
    Rb=R.from_quat(bqn)
    tot=np.zeros((nb,3))
    for qq,f in contact_frames:
        tot += (Rb * R.from_quat(qq[:nb])).apply(f.T[:nb])
    tot=tot.T; n=nb
    # filter's own unmodeled force
    uf=np.array([np.asarray(log[f"{B}estimatedState_extForceCentr_{a}"],float) for a in "xyz"])
    # mocap acceleration of the body
    with open(f"Projects/{name}/output_data/synchronizedObserversMocapData.csv") as fh:
        rd=csv.reader(fh,delimiter=';'); head=next(rd); idx={k:i for i,k in enumerate(head)}
        cols=[f"MocapAligner_worldBodyKine_linAcc_{a}" for a in "xyz"]
        acc=np.array([[float(r[idx[c]]) if r[idx[c]] not in ("","nan") else np.nan for c in cols] for r in rd])
    m=min(n,len(acc))
    a=acc[:m]; F=tot[:,:m].T; U=uf[:,:m].T
    resid = F - mass*(a + np.array([0,0,-9.81]))     # true external force implied by Newton
    ok=np.isfinite(resid).all(1)&np.isfinite(a).all(1)
    print(f"{name}")
    print(f"   sum of measured contact forces, xy:   median {np.median(np.linalg.norm(F[ok][:,:2],axis=1)):7.1f} N")
    print(f"   Newton residual (true external), xy:  median {np.median(np.linalg.norm(resid[ok][:,:2],axis=1)):7.1f} N")
    print(f"   filter's unmodeled force estimate xy: median {np.median(np.linalg.norm(U[ok][:,:2],axis=1)):7.1f} N")
    del log

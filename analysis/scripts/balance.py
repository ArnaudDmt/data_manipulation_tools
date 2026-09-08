import sys, numpy as np, csv; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
from scipy.signal import butter, filtfilt
P="Observers_MainObserverPipeline_MCKineticsObserver_"
B=P+"MEKF_"
MASS=58.32

def quat2R(q):                      # q = (w,x,y,z), shape (N,4) -> (N,3,3)
    w,x,y,z=q.T
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
                     [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]]).transpose(2,0,1)

def vec(log,base,comps="xyz"):
    return np.array([np.asarray(log[f"{base}_{c}"],float) for c in comps]).T

for name in ("KO_TRO2024_RHPS1_1","KO_TRO_2024_RHPS1_SLIPPAGE_1"):
    log=read_log(f"Projects/{name}/output_data/kinetics_eval/logReplay_full.bin")
    t=np.asarray(log["t"],float); n=len(t); dt=float(np.median(np.diff(t)))
    Rwc=quat2R(vec(log,B+"estimatedState_ori","wxyz"))
    fext=vec(log,B+"estimatedState_extForceCentr")
    fext_w=np.einsum('nij,nj->ni',Rwc,fext)

    fc_est_w=np.zeros((n,3)); fc_meas_w=np.zeros((n,3))
    for c in ("LeftFootCenter","RightFootCenter"):
        setf=np.array([str(s).strip().lower().startswith("set")
                       for s in log[P+f"debug_contactState_isSet_{c}"]])
        Rwci=quat2R(vec(log,B+f"estimatedState_contact_{c}_orientation","wxyz"))
        fe=np.einsum('nij,nj->ni',Rwci,vec(log,B+f"estimatedState_contact_{c}_forces"))
        fm=np.einsum('nij,nj->ni',Rwci,vec(log,B+f"measurements_contacts_force_{c}_measured"))
        fc_est_w+=np.where(setf[:,None],fe,0.0); fc_meas_w+=np.where(setf[:,None],fm,0.0)

    tot_w=fc_est_w+fext_w

    # standstill mask from differentiated mocap position
    with open(f"Projects/{name}/output_data/synchronizedObserversMocapData.csv") as fh:
        rd=csv.reader(fh,delimiter=';'); h=next(rd); i={k:j for j,k in enumerate(h)}
        cc=[f"MocapAligner_worldBodyKine_position_{a}" for a in "xyz"]
        v=np.array([[float(r[i[c]]) if r[i[c]] not in ("","nan") else np.nan for c in cc] for r in rd])
    m=min(n,len(v)); g=np.isfinite(v[:m]).all(1)
    pos=np.vstack([np.interp(np.arange(m),np.flatnonzero(g),v[:m][g,k]) for k in range(3)]).T
    b,a_=butter(2,5.0/(0.5/dt)); sp=np.r_[0,np.linalg.norm(np.diff(filtfilt(b,a_,pos,0),axis=0)[:,:2],axis=1)/dt]
    still=sp<0.02

    print(f"\n=== {name}   ({still.sum()} still / {m} samples)   world frame, medians [N]")
    print(f"{'':28s}{'|xy|':>9s}{'z':>9s}")
    for lab,arr in (("measured contact force",fc_meas_w),("estimated contact force",fc_est_w),
                    ("unmodeled ext force",fext_w),("TOTAL (contacts+unmod)",tot_w)):
        a=arr[:m][still]
        print(f"  {lab:26s}{np.median(np.linalg.norm(a[:,:2],axis=1)):9.1f}{np.median(a[:,2]):9.1f}")
    print(f"  {'required at standstill':26s}{0.0:9.1f}{MASS*9.81:9.1f}")
    del log

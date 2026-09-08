"""Reproduce EXACTLY the filter's centroid force balance (addUnmodeledAndContactWrench_),
using inputCentroidContactKine.orientation, and check it at standstill."""
import sys, numpy as np, csv; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
from scipy.signal import butter, filtfilt
P="Observers_MainObserverPipeline_MCKineticsObserver_"; B=P+"MEKF_"; MASS=58.32
def quat2R(q):
    q=q/np.linalg.norm(q,axis=1,keepdims=True); w,x,y,z=q.T
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]).transpose(2,0,1)
def vec(log,b,c="xyz"): return np.array([np.asarray(log[f"{b}_{k}"],float) for k in c]).T

for name in ("KO_TRO2024_RHPS1_1","KO_TRO_2024_RHPS1_SLIPPAGE_1"):
    log=read_log(f"Projects/{name}/output_data/kinetics_eval/logReplay_full.bin")
    t=np.asarray(log["t"],float); n=len(t); dt=float(np.median(np.diff(t)))
    Rwc=quat2R(vec(log,B+"estimatedState_ori","wxyz"))          # world <- centroid
    fext_c=vec(log,B+"estimatedState_extForceCentr")            # already centroid frame
    tot_c=fext_c.copy(); est_c=np.zeros((n,3)); meas_c=np.zeros((n,3)); per={}
    for c in ("LeftFootCenter","RightFootCenter"):
        setf=np.array([str(s).strip().lower().startswith("set") for s in log[P+f"debug_contactState_isSet_{c}"]])
        Rcc=quat2R(vec(log,P+f"debug_contactKine_{c}_inputCentroidContactKine_orientation","wxyz"))  # centroid <- contact
        fe_l=vec(log,B+f"estimatedState_contact_{c}_forces")
        fm_l=vec(log,B+f"measurements_contacts_force_{c}_measured")
        fe=np.where(setf[:,None],np.einsum('nij,nj->ni',Rcc,fe_l),0.0)
        fm=np.where(setf[:,None],np.einsum('nij,nj->ni',Rcc,fm_l),0.0)
        est_c+=fe; meas_c+=fm; per[c]=(fm_l,fm,setf)
    tot_c=fext_c+est_c
    tow=lambda a: np.einsum('nij,nj->ni',Rwc,a)                 # centroid -> world

    with open(f"Projects/{name}/output_data/synchronizedObserversMocapData.csv") as fh:
        rd=csv.reader(fh,delimiter=';'); h=next(rd); i={k:j for j,k in enumerate(h)}
        cc=[f"MocapAligner_worldBodyKine_position_{a}" for a in "xyz"]
        v=np.array([[float(r[i[k]]) if r[i[k]] not in ("","nan") else np.nan for k in cc] for r in rd])
    m=min(n,len(v)); g=np.isfinite(v[:m]).all(1)
    pos=np.vstack([np.interp(np.arange(m),np.flatnonzero(g),v[:m][g,k]) for k in range(3)]).T
    b,a_=butter(2,5.0/(0.5/dt)); sp=np.r_[0,np.linalg.norm(np.diff(filtfilt(b,a_,pos,0),axis=0)[:,:2],axis=1)/dt]
    still=sp<0.02; both=np.array([per[c][2][:m] for c in per]).all(0); sel=still&both

    print(f"\n=== {name}   {sel.sum()} still+double-support samples   WORLD frame medians [N]")
    print(f"{'':34s}{'|xy|':>9s}{'z':>9s}")
    for lab,arr in (("measured contact force (sum)",tow(meas_c)),("estimated contact force (sum)",tow(est_c)),
                    ("unmodeled ext force",tow(fext_c)),("TOTAL = contacts + unmodeled",tow(tot_c))):
        a=arr[:m][sel]; print(f"  {lab:32s}{np.median(np.linalg.norm(a[:,:2],axis=1)):9.1f}{np.median(a[:,2]):9.1f}")
    print(f"  {'physics requires at standstill':32s}{0.0:9.1f}{MASS*9.81:9.1f}")
    print("  per-foot measured force, CONTACT frame (fx, fy, fz):")
    for c,(fm_l,fm_c,_) in per.items():
        a=fm_l[:m][sel]; print(f"    {c:22s}{np.median(a[:,0]):8.1f}{np.median(a[:,1]):8.1f}{np.median(a[:,2]):8.1f}"
                               f"   |tangential| {np.median(np.linalg.norm(a[:,:2],axis=1)):6.1f}")
    del log

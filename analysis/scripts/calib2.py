"""Identify each foot's wrench rotation independently from SINGLE-SUPPORT samples,
where the whole ground reaction passes through that one foot and must be vertical."""
import sys, numpy as np, csv; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
from scipy.signal import butter, filtfilt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
P="Observers_MainObserverPipeline_MCKineticsObserver_"; B=P+"MEKF_"; MASS=58.32; G=9.81
def quat2R(q):
    q=q/np.linalg.norm(q,axis=1,keepdims=True); w,x,y,z=q.T
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]).transpose(2,0,1)
def vec(log,b,c="xyz"): return np.array([np.asarray(log[f"{b}_{k}"],float) for k in c]).T
FEET=("LeftFootCenter","RightFootCenter")
DS=["KO_TRO2024_RHPS1_1","KO_TRO2024_RHPS1_3","KO_TRO2024_RHPS1_5",
    "KO_TRO_2024_RHPS1_SLIPPAGE_1","HRP5_MultiContact_1"]
print(f"{'dataset':30s}{'foot':6s}{'n_SS':>7s}{'angle':>8s}{'|tan|/fz before':>17s}{'after':>8s}")
for name in DS:
    try: log=read_log(f"Projects/{name}/output_data/kinetics_eval/logReplay_full.bin")
    except Exception as e: print(f"{name}: {e}"); continue
    n=len(log["t"]); Rwc=quat2R(vec(log,B+"estimatedState_ori","wxyz"))
    Rw={};F={};S={}
    for c in FEET:
        Rw[c]=Rwc@quat2R(vec(log,P+f"debug_contactKine_{c}_inputCentroidContactKine_orientation","wxyz"))
        F[c]=vec(log,B+f"measurements_contacts_force_{c}_measured")
        S[c]=np.array([str(s).strip().lower().startswith("set") for s in log[P+f"debug_contactState_isSet_{c}"]])
    for k,c in enumerate(FEET):
        o=FEET[1-k]
        ss=np.flatnonzero(S[c]&(~S[o])&(F[c][:,2]>0.7*MASS*G))     # sole support, carrying ~full weight
        if len(ss)<200:
            print(f"{name:30s}{c[:5]:6s}{len(ss):7d}   (too few single-support samples)"); continue
        ss=ss[::max(1,len(ss)//3000)]
        Rk=Rw[c][ss]; fk=F[c][ss]
        def resid(p):
            Ri=Rot.from_rotvec(p).as_matrix()
            w=np.einsum('nij,jk,nk->ni',Rk,Ri,fk)
            return np.concatenate([w[:,:2].ravel()/np.sqrt(len(ss)), 0.3*p])   # horizontal -> 0, min-norm
        sol=least_squares(resid,np.zeros(3),method='lm')
        Ri=Rot.from_rotvec(sol.x).as_matrix()
        b4=np.median(np.linalg.norm(fk[:,:2],axis=1)/fk[:,2])
        fa=np.einsum('jk,nk->nj',Ri,fk)
        af=np.median(np.linalg.norm(fa[:,:2],axis=1)/np.maximum(fa[:,2],1e-6))
        print(f"{name:30s}{c[:5]:6s}{len(ss):7d}{np.degrees(np.linalg.norm(sol.x)):8.2f}"
              f"{b4:17.3f}{af:8.3f}   rotvec_deg [{np.degrees(sol.x)[0]:+6.2f}{np.degrees(sol.x)[1]:+7.2f}{np.degrees(sol.x)[2]:+7.2f}]")
    del log

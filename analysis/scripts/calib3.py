"""Per-foot wrench rotation via Wahba/Kabsch on single-support samples:
find R minimising  sum || R*(f/|f|) - d ||^2,  d = world-up expressed in the contact frame."""
import sys, numpy as np; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
from scipy.spatial.transform import Rotation as Rot
P="Observers_MainObserverPipeline_MCKineticsObserver_"; B=P+"MEKF_"; MASS=58.32; G=9.81
def quat2R(q):
    q=q/np.linalg.norm(q,axis=1,keepdims=True); w,x,y,z=q.T
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]).transpose(2,0,1)
def vec(log,b,c="xyz"): return np.array([np.asarray(log[f"{b}_{k}"],float) for k in c]).T
def wahba(A,Bv):
    M=Bv.T@A; U,_,Vt=np.linalg.svd(M); d=np.sign(np.linalg.det(U@Vt))
    return U@np.diag([1,1,d])@Vt
FEET=("LeftFootCenter","RightFootCenter")
DS=["KO_TRO2024_RHPS1_1","KO_TRO2024_RHPS1_2","KO_TRO2024_RHPS1_3","KO_TRO2024_RHPS1_4",
    "KO_TRO2024_RHPS1_5","KO_TRO_2024_RHPS1_SLIPPAGE_1","HRP5_MultiContact_1","HRP5P_LongWalk"]
print(f"{'dataset':30s}{'foot':7s}{'n':>6s}{'angle':>7s}{'  rotvec (deg, contact frame)':32s}"
      f"{'|tan|/fz':>10s}{'after':>8s}")
for name in DS:
    try: log=read_log(f"Projects/{name}/output_data/kinetics_eval/logReplay_full.bin")
    except Exception as e: print(f"{name:30s} -- {type(e).__name__}"); continue
    Rwc=quat2R(vec(log,B+"estimatedState_ori","wxyz")); Rw={};F={};S={}
    for c in FEET:
        Rw[c]=Rwc@quat2R(vec(log,P+f"debug_contactKine_{c}_inputCentroidContactKine_orientation","wxyz"))
        F[c]=vec(log,B+f"measurements_contacts_force_{c}_measured")
        S[c]=np.array([str(s).strip().lower().startswith("set") for s in log[P+f"debug_contactState_isSet_{c}"]])
    for k,c in enumerate(FEET):
        ss=np.flatnonzero(S[c]&(~S[FEET[1-k]])&(F[c][:,2]>0.7*MASS*G))
        if len(ss)<200: print(f"{name:30s}{c[:5]:7s}{len(ss):6d}   too few single-support"); continue
        ss=ss[::max(1,len(ss)//4000)]
        f=F[c][ss]; u=f/np.linalg.norm(f,axis=1,keepdims=True)
        d=np.einsum('nji,j->ni',Rw[c][ss],np.array([0,0,1.0]))     # world up in contact frame
        R=wahba(u,d); rv=np.degrees(Rot.from_matrix(R).as_rotvec())
        fa=np.einsum('jk,nk->nj',R,f)
        b4=np.median(np.linalg.norm(f[:,:2],axis=1)/f[:,2])
        af=np.median(np.linalg.norm(fa[:,:2],axis=1)/np.maximum(fa[:,2],1e-6))
        print(f"{name:30s}{c[:5]:7s}{len(ss):6d}{np.linalg.norm(rv):7.2f}"
              f"  [{rv[0]:+6.2f} {rv[1]:+6.2f} {rv[2]:+6.2f}]{'':10s}{b4:10.3f}{af:8.3f}")
    del log

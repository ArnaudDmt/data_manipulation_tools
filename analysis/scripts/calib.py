"""Fit a constant per-foot rotation of the measured wrench so that quasi-static
double support satisfies  sum_i R_w_ci * R_i * f_i = (0,0,m g)."""
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

for name in ("KO_TRO2024_RHPS1_1","KO_TRO_2024_RHPS1_SLIPPAGE_1"):
    log=read_log(f"Projects/{name}/output_data/kinetics_eval/logReplay_full.bin")
    t=np.asarray(log["t"],float); n=len(t); dt=float(np.median(np.diff(t)))
    Rwc=quat2R(vec(log,B+"estimatedState_ori","wxyz"))
    Rw={}; F={}; S={}
    for c in FEET:
        Rcc=quat2R(vec(log,P+f"debug_contactKine_{c}_inputCentroidContactKine_orientation","wxyz"))
        Rw[c]=Rwc@Rcc                                              # world <- contact
        F[c]=vec(log,B+f"measurements_contacts_force_{c}_measured") # contact frame
        S[c]=np.array([str(s).strip().lower().startswith("set") for s in log[P+f"debug_contactState_isSet_{c}"]])
    with open(f"Projects/{name}/output_data/synchronizedObserversMocapData.csv") as fh:
        rd=csv.reader(fh,delimiter=';'); h=next(rd); i={k:j for j,k in enumerate(h)}
        cc=[f"MocapAligner_worldBodyKine_position_{a}" for a in "xyz"]
        v=np.array([[float(r[i[k]]) if r[i[k]] not in ("","nan") else np.nan for k in cc] for r in rd])
    m=min(n,len(v)); g=np.isfinite(v[:m]).all(1)
    pos=np.vstack([np.interp(np.arange(m),np.flatnonzero(g),v[:m][g,k]) for k in range(3)]).T
    b,a_=butter(2,5.0/(0.5/dt)); sp=np.r_[0,np.linalg.norm(np.diff(filtfilt(b,a_,pos,0),axis=0)[:,:2],axis=1)/dt]
    sel=np.flatnonzero((sp<0.02)&S[FEET[0]][:m]&S[FEET[1]][:m]&(F[FEET[0]][:m,2]>150)&(F[FEET[1]][:m,2]>150))
    sel=sel[::max(1,len(sel)//4000)]
    target=np.array([0,0,MASS*G])
    def resid(p):
        tot=np.zeros((len(sel),3))
        for k,c in enumerate(FEET):
            Ri=Rot.from_rotvec(p[3*k:3*k+3]).as_matrix()
            tot+=np.einsum('nij,jk,nk->ni',Rw[c][sel],Ri,F[c][sel])
        return (tot-target).ravel()
    r0=resid(np.zeros(6)).reshape(-1,3)
    sol=least_squares(resid,np.zeros(6),method='lm')
    r1=sol.x.reshape(2,3); rf=resid(sol.x).reshape(-1,3)
    print(f"\n=== {name}   {len(sel)} quasi-static double-support samples")
    print(f"  net force error BEFORE:  |xy| {np.median(np.linalg.norm(r0[:,:2],axis=1)):7.1f} N"
          f"   z {np.median(r0[:,2]):+7.1f} N")
    print(f"  net force error AFTER :  |xy| {np.median(np.linalg.norm(rf[:,:2],axis=1)):7.1f} N"
          f"   z {np.median(rf[:,2]):+7.1f} N")
    for k,c in enumerate(FEET):
        ang=np.degrees(np.linalg.norm(r1[k])); ax=r1[k]/max(np.linalg.norm(r1[k]),1e-12)
        print(f"  {c:18s} correction {ang:5.2f} deg about [{ax[0]:+.3f} {ax[1]:+.3f} {ax[2]:+.3f}]"
              f"   rpy {np.degrees(Rot.from_rotvec(r1[k]).as_euler('xyz'))}")
    del log

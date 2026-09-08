"""Did the correction reach the observer? Unmodeled force and net contact force at standstill."""
import sys, numpy as np, csv; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
from scipy.signal import butter, filtfilt
P="Observers_MainObserverPipeline_MCKineticsObserver_"; B=P+"MEKF_"
def quat2R(q):
    q=q/np.linalg.norm(q,axis=1,keepdims=True); w,x,y,z=q.T
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]).transpose(2,0,1)
def vec(log,b,c="xyz"): return np.array([np.asarray(log[f"{b}_{k}"],float) for k in c]).T
log=read_log("Projects/KO_TRO2024_RHPS1_1/output_data/kinetics_eval/logReplay_full.bin")
t=np.asarray(log["t"],float); n=len(t); dt=float(np.median(np.diff(t)))
Rwc=quat2R(vec(log,B+"estimatedState_ori","wxyz")); fext=vec(log,B+"estimatedState_extForceCentr")
fw=np.einsum('nij,nj->ni',Rwc,fext)
meas=np.zeros((n,3))
for c in ("LeftFootCenter","RightFootCenter"):
    S=np.array([str(s).strip().lower().startswith("set") for s in log[P+f"debug_contactState_isSet_{c}"]])
    R=Rwc@quat2R(vec(log,P+f"debug_contactKine_{c}_inputCentroidContactKine_orientation","wxyz"))
    meas+=np.where(S[:,None],np.einsum('nij,nj->ni',R,vec(log,B+f"measurements_contacts_force_{c}_measured")),0.0)
with open("Projects/KO_TRO2024_RHPS1_1/output_data/synchronizedObserversMocapData.csv") as fh:
    rd=csv.reader(fh,delimiter=';'); h=next(rd); i={k:j for j,k in enumerate(h)}
    cc=[f"MocapAligner_worldBodyKine_position_{a}" for a in "xyz"]
    v=np.array([[float(r[i[k]]) if r[i[k]] not in ("","nan") else np.nan for k in cc] for r in rd])
m=min(n,len(v)); g=np.isfinite(v[:m]).all(1)
pos=np.vstack([np.interp(np.arange(m),np.flatnonzero(g),v[:m][g,k]) for k in range(3)]).T
b,a_=butter(2,5.0/(0.5/dt)); sp=np.r_[0,np.linalg.norm(np.diff(filtfilt(b,a_,pos,0),axis=0)[:,:2],axis=1)/dt]
st=sp<0.02
print(f"standstill samples {st.sum()}   (world frame, medians)")
print(f"  measured contact force  |xy| {np.median(np.linalg.norm(meas[:m][st][:,:2],axis=1)):7.1f} N   "
      f"z {np.median(meas[:m][st][:,2]):7.1f} N     [was 146.7 / 568.9]")
print(f"  unmodeled ext force     |xy| {np.median(np.linalg.norm(fw[:m][st][:,:2],axis=1)):7.1f} N   "
      f"z {np.median(fw[:m][st][:,2]):7.1f} N     [was 135.1 /  -3.1]")

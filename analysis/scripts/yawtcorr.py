"""Does the yaw-torque residual predict the KO's yaw error, sub-trajectory by sub-trajectory?
Compared against the RI-EKF on the same windows (which cannot see torque at all)."""
import sys, glob, numpy as np; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
from scipy.spatial.transform import Rotation as Rot
P="Observers_MainObserverPipeline_MCKineticsObserver_"; B=P+"MEKF_"; KA=727.0
def yaw_of(R):
    m=R.as_matrix(); return np.arctan2(m[1,0],m[0,0])
def vec(log,b,c="xyz"): return np.array([np.asarray(log[f"{b}_{k}"],float) for k in c]).T
PROJ=["KO_TRO2024_RHPS1_1","KO_TRO2024_RHPS1_2","KO_TRO2024_RHPS1_3","KO_TRO2024_RHPS1_4","KO_TRO2024_RHPS1_5"]
EK=[];ER=[];TQ=[]
for p in PROJ:
    base=f"Projects/{p}/output_data/kinetics_eval"
    gt=np.loadtxt(f"{base}/reference/mocap.txt",comments="#",ndmin=2)
    ri=np.loadtxt(f"{base}/reference/riekf.txt",comments="#",ndmin=2)
    ko=np.loadtxt(glob.glob(f"results/PW_cal-*/{p}/kinetics.txt")[0],comments="#",ndmin=2)
    t=gt[:,0]; it=lambda a: np.column_stack([np.interp(t,a[:,0],a[:,c]) for c in range(1,8)])
    G=gt[:,1:8]; R_=it(ri); K_=it(ko)
    log=read_log(f"{base}/logReplay_full.bin"); lt=np.asarray(log["t"],float); n=len(lt)
    tz=np.zeros(n)
    for c in ("LeftFootCenter","RightFootCenter"):
        S=np.array([str(s).strip().lower().startswith("set") for s in log[P+f"debug_contactState_isSet_{c}"]])
        d=vec(log,f"{B}measurements_contacts_torque_{c}_measured")-vec(log,f"{B}measurements_contacts_torque_{c}_predicted")
        f=vec(log,f"{B}measurements_contacts_force_{c}_measured")
        m=S&np.isfinite(d).all(1)&np.isfinite(f).all(1)&(f[:,2]>50)
        tz[m]+=np.abs(d[m,2])
    tzt=np.interp(t,lt,tz)
    step=np.r_[0,np.cumsum(np.linalg.norm(np.diff(G[:,:3],axis=0),axis=1))]
    ends=np.searchsorted(step,step+1.0)
    for i0 in range(0,len(t)-1,10):
        j=ends[i0]
        if j>=len(t): break
        Rg=Rot.from_quat(G[i0,3:7]).inv()*Rot.from_quat(G[j,3:7])
        EK.append(abs(np.degrees(yaw_of(Rg.inv()*(Rot.from_quat(K_[i0,3:7]).inv()*Rot.from_quat(K_[j,3:7]))))))
        ER.append(abs(np.degrees(yaw_of(Rg.inv()*(Rot.from_quat(R_[i0,3:7]).inv()*Rot.from_quat(R_[j,3:7]))))))
        TQ.append(np.degrees(np.mean(tzt[i0:j])/KA))
    del log
EK,ER,TQ=map(np.array,(EK,ER,TQ))
rms=lambda a: float(np.sqrt(np.mean(a**2)))
print(f"  n = {len(EK)} sub-trajectories of 1 m")
print(f"  corr(yaw-torque residual, KO yaw err) = {np.corrcoef(TQ,EK)[0,1]:+.3f}   [bias gave +0.113]")
print(f"  corr(yaw-torque residual, RI yaw err) = {np.corrcoef(TQ,ER)[0,1]:+.3f}   <- control: RI never sees torque")
print(f"  corr(yaw-torque residual, KO-RI diff) = {np.corrcoef(TQ,EK-ER)[0,1]:+.3f}")
q=np.percentile(TQ,[0,25,50,75,100])
print(f"\n  {'residual quartile':22s}{'KO yaw':>10s}{'RI yaw':>10s}{'ratio':>9s}")
for i in range(4):
    m=(TQ>=q[i])&(TQ<=q[i+1])
    print(f"  Q{i+1} ({q[i]:.3f}-{q[i+1]:.3f} deg){'':2s}{rms(EK[m]):10.4f}{rms(ER[m]):10.4f}{rms(EK[m])/rms(ER[m]):9.3f}")

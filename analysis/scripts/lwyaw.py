"""Does the yaw-torque friction problem affect LongWalk too, or is it masked by 10 m scoring?"""
import sys, glob, numpy as np; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
from scipy.spatial.transform import Rotation as Rot
P="Observers_MainObserverPipeline_MCKineticsObserver_"; B=P+"MEKF_"; KA=1000.0   # hrp5_p
def yaw_of(R):
    m=R.as_matrix(); return np.arctan2(m[1,0],m[0,0])
def vec(log,b,c="xyz"): return np.array([np.asarray(log[f"{b}_{k}"],float) for k in c]).T
p="HRP5P_LongWalk"; base=f"Projects/{p}/output_data/kinetics_eval"
gt=np.loadtxt(f"{base}/reference/mocap.txt",comments="#",ndmin=2)
ri=np.loadtxt(f"{base}/reference/riekf.txt",comments="#",ndmin=2)
kf=(glob.glob(f"results/W_LW-*/{p}/kinetics.txt")+glob.glob(f"results/V_tuned_a0.0-*/{p}/kinetics.txt"))[0]
ko=np.loadtxt(kf,comments="#",ndmin=2)
t=gt[:,0]; it=lambda a: np.column_stack([np.interp(t,a[:,0],a[:,c]) for c in range(1,8)])
G=gt[:,1:8]; R_=it(ri); K_=it(ko)
log=read_log(f"{base}/logReplay_full.bin"); lt=np.asarray(log["t"],float); n=len(lt)
tz=np.zeros(n); parts=[]
for c in ("LeftFootCenter","RightFootCenter"):
    S=np.array([str(s).strip().lower().startswith("set") for s in log[P+f"debug_contactState_isSet_{c}"]])
    d=vec(log,f"{B}measurements_contacts_torque_{c}_measured")-vec(log,f"{B}measurements_contacts_torque_{c}_predicted")
    f=vec(log,f"{B}measurements_contacts_force_{c}_measured")
    m=S&np.isfinite(d).all(1)&np.isfinite(f).all(1)&(f[:,2]>50)
    dz=np.abs(d[m,2]); dxy=np.linalg.norm(d[m,:2],axis=1)
    parts.append((c,np.median(dz),np.percentile(dz,90),np.median(dxy)))
    tz[m]+=np.abs(d[m,2])
del log
print(f"=== HRP5P_LongWalk   angStiffness {KA:.0f} N.m/rad")
print(f"{'contact':18s}{'|dTz| p50':>11s}{'p90':>9s}{'-> yaw p90 [deg]':>18s}{'|dTxy| p50':>12s}")
for c,a,b,x in parts:
    print(f"{c:18s}{a:11.3f}{b:9.3f}{np.degrees(b/KA):18.4f}{x:12.3f}")
print("  [RHPS1 for comparison: p50 0.4-1.7, p90 8.5-11.9, yaw p90 0.67-0.94 deg]")
print("  [HRP5P MultiContact:   p50 0.7-1.1, p90 3.1-4.4,  yaw p90 0.18-0.25 deg]")
tzt=np.interp(t,lt,tz)
step=np.r_[0,np.cumsum(np.linalg.norm(np.diff(G[:,:3],axis=0),axis=1))]
rms=lambda a: float(np.sqrt(np.mean(np.asarray(a)**2)))
for D in (1.0,10.0):
    ends=np.searchsorted(step,step+D); EK=[];ER=[];TQ=[]
    for i0 in range(0,len(t)-1,25):
        j=ends[i0]
        if j>=len(t): break
        Rg=Rot.from_quat(G[i0,3:7]).inv()*Rot.from_quat(G[j,3:7])
        EK.append(abs(np.degrees(yaw_of(Rg.inv()*(Rot.from_quat(K_[i0,3:7]).inv()*Rot.from_quat(K_[j,3:7]))))))
        ER.append(abs(np.degrees(yaw_of(Rg.inv()*(Rot.from_quat(R_[i0,3:7]).inv()*Rot.from_quat(R_[j,3:7]))))))
        TQ.append(np.degrees(np.mean(tzt[i0:j])/KA))
    EK,ER,TQ=map(np.array,(EK,ER,TQ)); q=np.percentile(TQ,[0,25,50,75,100])
    print(f"\n  --- {D:.0f} m sub-trajectories (n={len(EK)})   overall ratio {rms(EK)/rms(ER):.3f}")
    print(f"  {'residual quartile':26s}{'KO yaw':>10s}{'RI yaw':>10s}{'ratio':>9s}")
    for i in range(4):
        m=(TQ>=q[i])&(TQ<=q[i+1])
        print(f"  Q{i+1} ({q[i]:.3f}-{q[i+1]:.3f}){'':6s}{rms(EK[m]):10.4f}{rms(ER[m]):10.4f}{rms(EK[m])/rms(ER[m]):9.3f}")

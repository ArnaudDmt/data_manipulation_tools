"""Is the KO's short-horizon yaw deficit caused by its noisier gyro-bias estimate?
Yaw injected over a sub-trajectory ~ (mean bias_z in window - long-run bias_z) * duration."""
import sys, csv, glob, numpy as np; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
from scipy.spatial.transform import Rotation as Rot
KB="Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_gyroBias_Accelerometer"
def yaw_of(R):
    m=R.as_matrix(); return np.arctan2(m[1,0],m[0,0])
PROJ=["KO_TRO2024_RHPS1_1","KO_TRO2024_RHPS1_2","KO_TRO2024_RHPS1_3","KO_TRO2024_RHPS1_4","KO_TRO2024_RHPS1_5"]
D=1.0
agg={"KO":[[],[]],"RI":[[],[]]}
for p in PROJ:
    base=f"Projects/{p}/output_data/kinetics_eval"
    gt=np.loadtxt(f"{base}/reference/mocap.txt",comments="#",ndmin=2)
    ri=np.loadtxt(f"{base}/reference/riekf.txt",comments="#",ndmin=2)
    ko=np.loadtxt(glob.glob(f"results/PW_cal-*/{p}/kinetics.txt")[0],comments="#",ndmin=2)
    t=gt[:,0]; it=lambda a: np.column_stack([np.interp(t,a[:,0],a[:,c]) for c in range(1,8)])
    G=gt[:,1:8]; R_=it(ri); K_=it(ko)
    with open(f"Projects/{p}/output_data/synchronizedObserversMocapData.csv") as fh:
        rd=csv.reader(fh,delimiter=';'); h=next(rd); i={k:j for j,k in enumerate(h)}
        c=[f"Hartley_IMU_gyroBias_{a}" for a in "xyz"]
        hb=np.array([[float(r[i[k]]) if r[i[k]] not in ("","nan") else np.nan for k in c] for r in rd])
    log=read_log(f"{base}/logReplay_full.bin"); lt=np.asarray(log["t"],float)
    kb=np.array([np.asarray(log[f"{KB}_{a}"],float) for a in "xyz"]).T
    m=min(len(lt),len(hb)); lt,kb,hb=lt[:m],kb[:m],hb[:m]
    kz=np.interp(t,lt,kb[:,2]); hz=np.interp(t,lt,np.nan_to_num(hb[:,2],nan=0.0))
    step=np.r_[0,np.cumsum(np.linalg.norm(np.diff(G[:,:3],axis=0),axis=1))]
    ends=np.searchsorted(step,step+D)
    kzg,hzg=np.mean(kz),np.mean(hz)
    for lab,E,bz,bg in (("KO",K_,kz,kzg),("RI",R_,hz,hzg)):
        for i0 in range(0,len(t)-1,10):
            j=ends[i0]
            if j>=len(t): break
            Rg=Rot.from_quat(G[i0,3:7]).inv()*Rot.from_quat(G[j,3:7])
            Re=Rot.from_quat(E[i0,3:7]).inv()*Rot.from_quat(E[j,3:7])
            agg[lab][0].append(abs(np.degrees(yaw_of(Rg.inv()*Re))))
            agg[lab][1].append(abs(np.degrees((bz[i0:j].mean()-bg)*(t[j]-t[i0]))))
    del log
print(f"{'':6s}{'yaw RPE rms':>14s}{'bias-injected yaw':>20s}{'corr':>8s}{'share':>9s}")
for lab in ("KO","RI"):
    e=np.array(agg[lab][0]); b=np.array(agg[lab][1])
    rms=lambda a: float(np.sqrt(np.mean(a**2)))
    print(f"{lab:6s}{rms(e):14.4f}{rms(b):20.4f}{np.corrcoef(e,b)[0,1]:8.3f}{rms(b)/rms(e)*100:8.1f}%")
print("\n  yaw RPE gap KO-RI  = %.4f deg" % (np.sqrt(np.mean(np.array(agg['KO'][0])**2))
                                            -np.sqrt(np.mean(np.array(agg['RI'][0])**2))))
print("  bias-injected gap  = %.4f deg" % (np.sqrt(np.mean(np.array(agg['KO'][1])**2))
                                           -np.sqrt(np.mean(np.array(agg['RI'][1])**2))))

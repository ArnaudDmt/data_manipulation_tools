"""Saturation vs RPE error, on the benchmark's own sub-trajectory definition:
1 m of ground-truth path, SE(3)-aligned at the sub-trajectory start, horizontal translation error.
First reproduce the published numbers, then stratify."""
import sys, glob, numpy as np; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
from scipy.spatial.transform import Rotation as Rot
P="Observers_MainObserverPipeline_MCKineticsObserver_"; B=P+"MEKF_"
CAL={"LeftFootCenter":np.array([0.047962,0.324858,0.015621]),
     "RightFootCenter":np.array([-0.282761,0.142506,0.001466])}
def rod(rv):
    a=np.linalg.norm(rv); k=rv/a
    K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)+np.sin(a)*K+(1-np.cos(a))*K@K
def vec(log,b,c="xyz"): return np.array([np.asarray(log[f"{b}_{k}"],float) for k in c]).T
def rel(Tp,Tq,i,j):           # translation of inv(T_i) * T_j
    Ri=Rot.from_quat(Tq[i]); return Ri.inv().apply(Tp[j]-Tp[i]), Ri.inv()*Rot.from_quat(Tq[j])
DIST=1.0
allko=[];allri=[]
for name in ("KO_TRO_2024_RHPS1_SLIPPAGE_1","KO_TRO_2024_RHPS1_SLIPPAGE_2","KO_TRO_2024_RHPS1_SLIPPAGE_3"):
    base=f"Projects/{name}/output_data/kinetics_eval"
    gt=np.loadtxt(f"{base}/reference/mocap.txt",comments="#",ndmin=2)
    ri=np.loadtxt(f"{base}/reference/riekf.txt",comments="#",ndmin=2)
    ko=np.loadtxt(glob.glob(f"results/PW_cal-*/{name}/kinetics.txt")[0],comments="#",ndmin=2)
    t=gt[:,0]
    it=lambda a: np.column_stack([np.interp(t,a[:,0],a[:,c]) for c in range(1,8)])
    G=gt[:,1:8]; R_=it(ri); K_=it(ko)
    gp,gq=G[:,:3],G[:,3:7]
    step=np.r_[0,np.cumsum(np.linalg.norm(np.diff(gp,axis=0),axis=1))]
    log=read_log(f"{base}/logReplay_full.bin"); lt=np.asarray(log["t"],float); n=len(lt)
    sat=np.zeros(n)
    for c in ("LeftFootCenter","RightFootCenter"):
        S=np.array([str(s).strip().lower().startswith("set") for s in log[P+f"debug_contactState_isSet_{c}"]])
        f=np.einsum('jk,nk->nj',rod(CAL[c]),vec(log,B+f"measurements_contacts_force_{c}_measured"))
        good=S&np.isfinite(f).all(1)&(f[:,2]>50)
        r=np.zeros(n); r[good]=np.linalg.norm(f[good][:,:2],axis=1)/f[good][:,2]
        sat=np.maximum(sat,r)
    st=np.interp(t,lt,sat)
    ends=np.searchsorted(step,step+DIST)
    starts=np.array([i for i in range(0,len(t)-1,10) if ends[i]<len(t)])
    e={}
    for lab,E in (("KO",K_),("RI",R_)):
        v=[]
        for i in starts:
            j=ends[i]
            dg,_=rel(gp,gq,i,j); de,_=rel(E[:,:3],E[:,3:7],i,j)
            v.append(np.linalg.norm((de-dg)[:2])*1000)
        e[lab]=np.array(v)
    frac=np.array([np.mean(st[i:ends[i]]>0.40) for i in starts])
    rms=lambda a: float(np.sqrt(np.mean(a**2)))
    allko.append(e["KO"]); allri.append(e["RI"])
    hi=frac>np.median(frac); lo=~hi
    print(f"\n=== {name}   {len(starts)} sub-trajectories of {DIST} m")
    print(f"  RPE trans_xy RMSE   KO {rms(e['KO']):7.2f} mm   RI {rms(e['RI']):7.2f} mm"
          f"   ratio {rms(e['KO'])/rms(e['RI']):5.3f}   <-- benchmark KO 18.64 / RI 31.26 = 0.596")
    print(f"  saturated fraction per sub-traj: p50 {np.median(frac):.3f}  p90 {np.percentile(frac,90):.3f}")
    print(f"  low-saturation  KO {rms(e['KO'][lo]):7.2f}  RI {rms(e['RI'][lo]):7.2f}  ratio {rms(e['KO'][lo])/rms(e['RI'][lo]):5.3f}")
    print(f"  high-saturation KO {rms(e['KO'][hi]):7.2f}  RI {rms(e['RI'][hi]):7.2f}  ratio {rms(e['KO'][hi])/rms(e['RI'][hi]):5.3f}")
    del log
k=np.concatenate(allko); r=np.concatenate(allri)
print(f"\nPOOLED slippage: KO {np.sqrt(np.mean(k**2)):.2f} mm  RI {np.sqrt(np.mean(r**2)):.2f} mm"
      f"  ratio {np.sqrt(np.mean(k**2))/np.sqrt(np.mean(r**2)):.3f}")

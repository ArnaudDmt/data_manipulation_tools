"""Is the contact frame the filter uses actually near-vertical in world during stance?"""
import sys, numpy as np; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
P="Observers_MainObserverPipeline_MCKineticsObserver_"; B=P+"MEKF_"
def quat2R(q):
    q=q/np.linalg.norm(q,axis=1,keepdims=True); w,x,y,z=q.T
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]).transpose(2,0,1)
def vec(log,b,c="xyz"): return np.array([np.asarray(log[f"{b}_{k}"],float) for k in c]).T
for name in ("KO_TRO2024_RHPS1_1","HRP5_MultiContact_1"):
    log=read_log(f"Projects/{name}/output_data/kinetics_eval/logReplay_full.bin")
    Rwc=quat2R(vec(log,B+"estimatedState_ori","wxyz"))
    print(f"\n=== {name}")
    for c in ("LeftFootCenter","RightFootCenter"):
        setf=np.array([str(s).strip().lower().startswith("set") for s in log[P+f"debug_contactState_isSet_{c}"]])
        f=vec(log,B+f"measurements_contacts_force_{c}_measured"); msk=setf&(f[:,2]>150)
        Rcc=quat2R(vec(log,P+f"debug_contactKine_{c}_inputCentroidContactKine_orientation","wxyz"))
        zc=np.einsum('nij,nj->ni',Rwc@Rcc,np.tile([0,0,1.0],(len(Rwc),1)))     # contact z in world
        tiltc=np.degrees(np.arccos(np.clip(zc[:,2],-1,1)))
        # direction of the tangential force in the contact frame, and its stability
        ang=np.degrees(np.arctan2(f[msk][:,1],f[msk][:,0]))
        print(f"  {c}: contact-frame z tilt from vertical  median {np.median(tiltc[msk]):5.2f} deg"
              f"  (p10 {np.percentile(tiltc[msk],10):.2f}, p90 {np.percentile(tiltc[msk],90):.2f})")
        print(f"    tangential force direction in contact frame: median {np.median(ang):7.1f} deg"
              f"   IQR {np.percentile(ang,75)-np.percentile(ang,25):5.1f} deg")
    del log

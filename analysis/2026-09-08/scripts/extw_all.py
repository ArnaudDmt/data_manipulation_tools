"""Estimated perturbation wrench vs its known truth of zero, across every dataset.
Nothing external touches the robot in any of these recordings, so the true unmodeled wrench is
zero throughout, and exactly zero (no inertial excuse) while the robot stands still."""
import sys, os, csv, ctypes, glob, collections, numpy as np
from pathlib import Path
for prefix in os.environ.get("AMENT_PREFIX_PATH","").split(os.pathsep):
    if not prefix: continue
    d=Path(prefix)/"lib"
    for lib in sorted(list(d.glob("libkinetics_observer_ros2*.so"))+list(d.glob("python*/site-packages/kinetics_observer_ros2/*.so"))):
        try: ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
        except OSError: pass
import rosbag2_py
from rclpy.serialization import deserialize_message
from kinetics_observer_ros2.msg import KineticsState
from scipy.signal import butter, filtfilt
PROJ=["HRP5_MultiContact_1","HRP5_MultiContact_2","HRP5_MultiContact_3","HRP5_MultiContact_4",
 "KO_TRO2024_RHPS1_1","KO_TRO2024_RHPS1_2","KO_TRO2024_RHPS1_3","KO_TRO2024_RHPS1_4","KO_TRO2024_RHPS1_5",
 "KO_TRO_2024_RHPS1_SLIPPAGE_1","KO_TRO_2024_RHPS1_SLIPPAGE_2","KO_TRO_2024_RHPS1_SLIPPAGE_3"]
def cat(p):
    if "MultiContact" in p: return "MultiContact (HRP5P, uncorrected control)"
    if "SLIPPAGE" in p.upper(): return "RHPS1 slippage"
    return "RHPS1 walk"
def read(bag):
    sid="mcap" if any(Path(bag).glob("*.mcap")) else "sqlite3"
    r=rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(bag),storage_id=sid),rosbag2_py.ConverterOptions("",""))
    t=[];f=[];q=[]
    while r.has_next():
        topic,ser,_=r.read_next()
        if topic!="/kinetics_observer/estimated_state": continue
        s=deserialize_message(ser,KineticsState); w=s.unmodeled_wrench
        t.append(s.header.stamp.sec+s.header.stamp.nanosec*1e-9)
        f.append([w.force.x,w.force.y,w.force.z]); q.append([w.torque.x,w.torque.y,w.torque.z])
    return np.array(t),np.array(f),np.array(q)
def still_mask(proj,n,dt):
    p=f"Projects/{proj}/output_data/synchronizedObserversMocapData.csv"
    if not os.path.isfile(p): return None
    with open(p) as fh:
        rd=csv.reader(fh,delimiter=';'); h=next(rd); i={k:j for j,k in enumerate(h)}
        cc=[f"MocapAligner_worldBodyKine_position_{a}" for a in "xyz"]
        if any(c not in i for c in cc): return None
        v=np.array([[float(r[i[k]]) if r[i[k]] not in ("","nan") else np.nan for k in cc] for r in rd])
    g=np.isfinite(v).all(1)
    if g.sum()<100: return None
    pos=np.vstack([np.interp(np.arange(len(v)),np.flatnonzero(g),v[g,k]) for k in range(3)]).T
    b,a_=butter(2,min(0.9,5.0/(0.5/dt)))
    sp=np.r_[0,np.linalg.norm(np.diff(filtfilt(b,a_,pos,0),axis=0)[:,:2],axis=1)/dt]
    m=min(n,len(sp)); out=np.zeros(n,bool); out[:m]=sp[:m]<0.02; return out
runs={}
for lab,pref in (("baseline","PW_base"),("calibrated","PW_cal")):
    d=glob.glob(f"results/{pref}-*/")
    if not d: print(f"!! no results for {pref}"); continue
    runs[lab]=d[0]
rows=collections.defaultdict(dict)
for proj in PROJ:
    for lab,d in runs.items():
        bag=os.path.join(d,proj,"standalone_replay")
        if not os.path.isdir(bag): continue
        t,f,q=read(bag)
        if len(t)<100: continue
        dt=float(np.median(np.diff(t))) or 0.002
        st=still_mask(proj,len(t),dt)
        nf=np.linalg.norm(f,axis=1); nq=np.linalg.norm(q,axis=1)
        rows[proj][lab]=(np.median(nf),np.median(nq),
                         np.median(nf[st]) if st is not None and st.sum()>200 else np.nan,
                         np.median(nq[st]) if st is not None and st.sum()>200 else np.nan)
print(f"{'dataset':30s}{'|force| [N]':>26s}{'|torque| [N.m]':>26s}")
print(f"{'':30s}{'base':>12s}{'calib':>12s}{'base':>13s}{'calib':>12s}   (standstill)")
agg=collections.defaultdict(lambda: collections.defaultdict(list))
for proj in PROJ:
    r=rows.get(proj,{})
    if "baseline" not in r or "calibrated" not in r: print(f"{proj:30s}  (missing)"); continue
    b,c=r["baseline"],r["calibrated"]
    print(f"{proj:30s}{b[2]:12.1f}{c[2]:12.1f}{b[3]:13.2f}{c[3]:12.2f}")
    for k,(bi,ci) in (("f",(b[2],c[2])),("t",(b[3],c[3]))):
        if np.isfinite(bi) and np.isfinite(ci): agg[cat(proj)][k].append((bi,ci))
print()
for c,v in agg.items():
    f=np.array(v["f"]); q=np.array(v["t"])
    print(f"{c:44s} force {f[:,0].mean():7.1f} -> {f[:,1].mean():6.1f} N"
          f"   ({f[:,0].mean()/max(f[:,1].mean(),1e-9):4.1f}x)"
          f"    torque {q[:,0].mean():7.2f} -> {q[:,1].mean():6.2f} N.m"
          f"   ({q[:,0].mean()/max(q[:,1].mean(),1e-9):4.1f}x)")
print("\ntruth = 0 N / 0 N.m everywhere; standstill removes any inertial contribution")

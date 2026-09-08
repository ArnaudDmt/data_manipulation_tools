"""Estimated perturbation (unmodeled) wrench from the replay bags. On RHPS1 nothing touches the
robot, so the true external wrench is zero everywhere -- and exactly zero while it stands still."""
import sys, numpy as np, csv, ctypes, os
from pathlib import Path
sys.path.insert(0,'scripts')
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

def read(bag):
    sid="mcap" if any(Path(bag).glob("*.mcap")) else "sqlite3"
    r=rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(bag),storage_id=sid),rosbag2_py.ConverterOptions("",""))
    t=[];f=[];tq=[]
    while r.has_next():
        topic,ser,_=r.read_next()
        if topic!="/kinetics_observer/estimated_state": continue
        s=deserialize_message(ser,KineticsState)
        t.append(s.header.stamp.sec+s.header.stamp.nanosec*1e-9)
        w=s.unmodeled_wrench
        f.append([w.force.x,w.force.y,w.force.z]); tq.append([w.torque.x,w.torque.y,w.torque.z])
    return np.array(t),np.array(f),np.array(tq)

# standstill mask from mocap
with open("Projects/KO_TRO2024_RHPS1_1/output_data/synchronizedObserversMocapData.csv") as fh:
    rd=csv.reader(fh,delimiter=';'); h=next(rd); i={k:j for j,k in enumerate(h)}
    cc=[f"MocapAligner_worldBodyKine_position_{a}" for a in "xyz"]
    v=np.array([[float(r[i[k]]) if r[i[k]] not in ("","nan") else np.nan for k in cc] for r in rd])
g=np.isfinite(v).all(1)
pos=np.vstack([np.interp(np.arange(len(v)),np.flatnonzero(g),v[g,k]) for k in range(3)]).T
b,a_=butter(2,5.0/(0.5/0.002)); sp=np.r_[0,np.linalg.norm(np.diff(filtfilt(b,a_,pos,0),axis=0)[:,:2],axis=1)/0.002]

print(f"{'run':12s}{'|force| all':>14s}{'|force| still':>15s}{'|torque| all':>15s}{'|torque| still':>15s}")
print(f"{'':12s}{'[N]':>14s}{'[N]':>15s}{'[N.m]':>15s}{'[N.m]':>15s}")
for lab,bag in (("baseline","results/WB_base-532f02dcf0/KO_TRO2024_RHPS1_1/standalone_replay"),
                ("calibrated","results/WB_cal-d2deb1c710/KO_TRO2024_RHPS1_1/standalone_replay")):
    t,f,tq=read(bag)
    n=min(len(t),len(sp)); still=sp[:n]<0.02
    nf=np.linalg.norm(f[:n],axis=1); nt=np.linalg.norm(tq[:n],axis=1)
    print(f"{lab:12s}{np.median(nf):14.1f}{np.median(nf[still]):15.1f}"
          f"{np.median(nt):15.2f}{np.median(nt[still]):15.2f}   (n={n}, still={still.sum()})")
print("\ntruth: 0 N / 0 N.m -- nothing touches the robot in these datasets")

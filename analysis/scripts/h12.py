import sys, numpy as np, csv, os
sys.path.insert(0,'scripts')
from scipy.spatial.transform import Rotation as R

def yaw_of(q):  # q as x,y,z,w
    return R.from_quat(q).as_euler('zyx')[:,0]

def load(project):
    p=f"Projects/{project}/output_data/synchronizedObserversMocapData.csv"
    with open(p) as f:
        rd=csv.reader(f,delimiter=';'); head=next(rd); idx={n:i for i,n in enumerate(head)}
        need = ([f"MocapAligner_worldBodyKine_position_{a}" for a in "xyz"]
              + [f"MocapAligner_worldBodyKine_ori_{a}" for a in "wxyz"]
              + [f"KO_position_{a}" for a in "xyz"] + [f"KO_orientation_{a}" for a in "wxyz"]
              + [f"Hartley_IMU_position_{a}" for a in "xyz"]
              + [f"Hartley_IMU_orientation_{a}" for a in "xyzw"] + ["t"])
        miss=[n for n in need if n not in idx]
        if miss: print(f"{project}: missing {miss[:3]}"); return None
        rows=[]
        for r in rd:
            try: rows.append([float(r[idx[n]]) for n in need])
            except ValueError: rows.append([np.nan]*len(need))
    d=np.array(rows)
    return {"t":d[:,-1],
            "mocap_p":d[:,0:3], "mocap_q":d[:,[4,5,6,3]],
            "ko_p":d[:,7:10],   "ko_q":d[:,[11,12,13,10]],
            "ri_p":d[:,14:17],  "ri_q":d[:,17:21]}

for proj in ("KO_TRO2024_RHPS1_3","KO_TRO2024_RHPS1_1","HRP5_MultiContact_1"):
    if not os.path.exists(f"Projects/{proj}/output_data/synchronizedObserversMocapData.csv"): continue
    d=load(proj)
    if d is None: continue
    ok=np.isfinite(d["mocap_q"]).all(1)&(np.linalg.norm(d["mocap_q"],axis=1)>1e-6)
    ok&=np.isfinite(d["ko_q"]).all(1)&(np.linalg.norm(d["ko_q"],axis=1)>1e-6)
    ok&=np.isfinite(d["ri_q"]).all(1)&(np.linalg.norm(d["ri_q"],axis=1)>1e-6)
    if ok.sum()<1000: print(f"{proj}: too few valid rows ({ok.sum()})"); continue
    n=lambda q: q/np.linalg.norm(q,axis=1,keepdims=True)
    ym=yaw_of(n(d["mocap_q"][ok])); yk=yaw_of(n(d["ko_q"][ok])); yr=yaw_of(n(d["ri_q"][ok]))
    t=d["t"][ok]
    unwrap=lambda a: np.unwrap(a)
    ek=unwrap(yk)-unwrap(ym); er=unwrap(yr)-unwrap(ym)
    ek-=ek[0]; er-=er[0]
    dur=t[-1]-t[0]
    # linear fit: how much of the drift is a constant rate?
    A=np.vstack([t-t[0], np.ones_like(t)]).T
    ck,*_=np.linalg.lstsq(A,ek,rcond=None); cr,*_=np.linalg.lstsq(A,er,rcond=None)
    r2k=1-np.var(ek-A@ck)/max(np.var(ek),1e-15); r2r=1-np.var(er-A@cr)/max(np.var(er),1e-15)
    print(f"{proj}  ({dur:.0f} s, n={ok.sum()})")
    print(f"   KO    yaw drift {np.degrees(ek[-1]):+7.2f} deg total | linear rate {np.degrees(ck[0])*60:+6.3f} deg/min | R2(linear) {r2k:5.2f}")
    print(f"   RIEKF yaw drift {np.degrees(er[-1]):+7.2f} deg total | linear rate {np.degrees(cr[0])*60:+6.3f} deg/min | R2(linear) {r2r:5.2f}")

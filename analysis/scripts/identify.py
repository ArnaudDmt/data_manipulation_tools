"""Identify the contact stiffness and damping from mocap, without using the filter's rest pose.

Within one contact phase the physical foot is fixed in the world, so the rest pose is constant and
differencing removes it:  dF = -K.dx - C.dv, where dx is the change in the forward-kinematic contact
position built from the *mocap* base pose and the encoders. Nothing here comes from the observer,
so the identification is not circular.
"""
import sys, numpy as np, csv, os
sys.path.insert(0,'scripts')
from mc_log_ui import read_log
from scipy.spatial.transform import Rotation as R

def run(project, mass_label):
    log = read_log(f"Projects/{project}/output_data/kinetics_eval/logReplay_full.bin")
    B = "Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_"
    S = "Observers_MainObserverPipeline_MCKineticsObserver_debug_contactState_isSet_"
    # mocap base pose, synchronised row-for-row with the log
    csvp = f"Projects/{project}/output_data/synchronizedObserversMocapData.csv"
    cols = {}
    with open(csvp) as f:
        rd = csv.reader(f, delimiter=';')
        head = next(rd)
        idx = {n: i for i, n in enumerate(head)}
        want = [f"MocapAligner_worldBodyKine_position_{a}" for a in "xyz"] + \
               [f"MocapAligner_worldBodyKine_ori_{a}" for a in "wxyz"]
        if any(w not in idx for w in want):
            print(f"{project}: mocap columns missing"); return
        data = np.array([[float(r[idx[w]]) if r[idx[w]] not in ("","nan") else np.nan for w in want]
                         for r in rd])
    n = min(len(data), len(log["t"]))
    p_base = data[:n, 0:3]
    q_base = data[:n, [4,5,6,3]]            # x,y,z,w for scipy
    out = {}
    for foot in ("RightFoot", "LeftFoot"):
        name = f"{foot}Center"
        try:
            p_fb = np.array([np.asarray(log[f"{B}{name}_fbContactKine_position_{a}"], float)[:n] for a in "xyz"]).T
            q_fb = np.array([np.asarray(log[f"{B}{name}_fbContactKine_ori_{a}"], float)[:n] for a in ("x","y","z","w")]).T
            raw = np.asarray(log[f"{S}{name}"])[:n]
            # the flag is logged as a string state ("Set"/"notSet") rather than a number
            isset = np.array([str(v).strip().lower().startswith("set") for v in raw])
            f_s = np.array([np.asarray(log[f"{foot}ForceSensor_{a}"], float)[:n] for a in ("fx","fy","fz")]).T
        except KeyError as e:
            print(f"{project} {name}: missing {e}"); continue
        # Mocap drops out on some rows, leaving all-zero quaternions. Exclude those rather than
        # zero-filling them, and substitute identity so the rotation objects can still be built.
        nb = np.linalg.norm(q_base, axis=1); nf = np.linalg.norm(q_fb, axis=1)
        ok = (np.isfinite(p_base).all(1) & np.isfinite(p_fb).all(1)
              & np.isfinite(q_base).all(1) & np.isfinite(q_fb).all(1)
              & (nb > 1e-6) & (nf > 1e-6))
        qb = np.where(ok[:, None], q_base, np.array([0.0, 0.0, 0.0, 1.0]))
        qf = np.where(ok[:, None], q_fb, np.array([0.0, 0.0, 0.0, 1.0]))
        qb = qb / np.linalg.norm(qb, axis=1, keepdims=True)
        qf = qf / np.linalg.norm(qf, axis=1, keepdims=True)
        Rb = R.from_quat(qb); Rf = R.from_quat(qf)
        p_w = p_base + Rb.apply(p_fb)                       # world contact position, mocap-based
        Rc = Rb * Rf                                        # world<-contact orientation
        f_w = Rc.apply(f_s)                                 # measured force in world
        # contact phases
        edges = np.flatnonzero(np.diff(isset.astype(int)))
        starts = edges[::2] + 1 if isset[0] == False else np.r_[0, edges[1::2] + 1]
        X, Y = [], []
        for s in starts:
            e = s
            while e < n and isset[e]: e += 1
            if e - s < 200: continue
            sl = slice(s + 50, e - 50)                      # drop touch-down/lift-off transients
            if sl.stop <= sl.start: continue
            m = ok[sl] & (f_w[sl, 2] > 100.0)
            if m.sum() < 100: continue
            dx = p_w[sl][m] - p_w[sl][m].mean(0)
            df = f_w[sl][m] - f_w[sl][m].mean(0)
            X.append(dx); Y.append(df)
        if not X: print(f"{project} {name}: no usable phase"); continue
        X = np.vstack(X); Y = np.vstack(Y)
        res = {}
        for i, a in enumerate("xyz"):
            k, *_ = np.linalg.lstsq(X[:, [i]], Y[:, i], rcond=None)
            pred = X[:, i] * k[0]
            r2 = 1.0 - np.var(Y[:, i] - pred) / max(np.var(Y[:, i]), 1e-12)
            res[a] = (-k[0], r2, np.std(Y[:, i] - pred))
        out[name] = (res, len(X))
        print(f"{mass_label} {name}: n={len(X)}")
        print(f"    displacement spread within a stance phase: "
              f"x {np.std(X[:,0])*1000:.1f} mm, y {np.std(X[:,1])*1000:.1f} mm, z {np.std(X[:,2])*1000:.1f} mm")
        print(f"    deflection implied by the configured K=3e4 and the measured force spread: "
              f"{np.std(Y[:,0])/3e4*1000:.2f} mm")
        for a in "xyz":
            K, r2, resid = res[a]
            print(f"    axis {a}: K = {K:12.1f} N/m    R2 = {r2:6.3f}    residual = {resid:6.2f} N")
        del X, Y
    del log
    return out

for proj, lab in (("KO_TRO2024_RHPS1_1", "RHPS1"), ("HRP5_MultiContact_1", "HRP5P")):
    if os.path.exists(f"Projects/{proj}/output_data/synchronizedObserversMocapData.csv"):
        run(proj, lab)
    else:
        print(f"{proj}: no synchronised mocap file")
print("DONE")

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

CSV_PATH = "/tmp/gyro_bias.csv"
DO_SAVE = True
SAVE_DIR = "/tmp/figures"
SAVE_FORMATS = ("pdf", )
pio.renderers.default = "browser"

def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def style_layout(fig):
    fig.update_layout(
        template="simple_white",
        width=900,
        height=500,
        font=dict(size=22),
        margin=dict(l=80, r=20, t=20, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=22)),
        title=None,
        showlegend=True,
        legend_title_text=None
    )
    fig.update_xaxes(
        title_text="Time (s)",
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(0,0,0,0.1)",
        zeroline=False,
        ticks="outside",
        tickfont=dict(size=20)
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(0,0,0,0.1)",
        zeroline=False,
        ticks="outside",
        tickfont=dict(size=20)
    )

def save_fig(fig, name_base):
    if not DO_SAVE:
        return
    ensure_dir(SAVE_DIR)
    for fmt in SAVE_FORMATS:
        out = os.path.join(SAVE_DIR, f"{name_base}.{fmt}")
        try:
            pio.write_image(fig, out)
        except Exception as e:
            print(f"[warn] Could not save {out}: {e}")

def wrap_pi(a):
    a = (a + np.pi) % (2*np.pi)
    return a - np.pi

def continuous_euler(angles):
    angles = np.asarray(angles)
    out = np.empty_like(angles)
    out[0] = angles[0]
    for i in range(1, len(angles)):
        diff = angles[i] - angles[i-1]
        for j in range(len(diff)):
            if diff[j] > np.pi:
                diff[j] -= 2*np.pi
            elif diff[j] < -np.pi:
                diff[j] += 2*np.pi
        out[i] = out[i-1] + diff
    return out

def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n = np.where(n == 0.0, 1.0, n)
    return q / n

def quat_conjugate(q):
    out = q.copy()
    out[:, 1:] *= -1.0
    return out

def quat_multiply(a, b):
    a0, a1, a2, a3 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    b0, b1, b2, b3 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    c0 = a0*b0 - a1*b1 - a2*b2 - a3*b3
    c1 = a0*b1 + a1*b0 + a2*b3 - a3*b2
    c2 = a0*b2 - a1*b3 + a2*b0 + a3*b1
    c3 = a0*b3 + a1*b2 - a2*b1 + a3*b0
    return np.column_stack((c0, c1, c2, c3))

def quat_angle_error(q_est, q_true):
    qe = quat_normalize(np.asarray(q_est))
    qt = quat_normalize(np.asarray(q_true))
    qrel = quat_multiply(quat_conjugate(qt), qe)
    vnorm = np.linalg.norm(qrel[:, 1:], axis=1)
    w = np.clip(np.abs(qrel[:, 0]), 0.0, 1.0)
    return 2.0 * np.arctan2(vnorm, w)

def quat_hemisphere(q):
    q = np.asarray(q, dtype=float).copy()
    q[0] = q[0] / (np.linalg.norm(q[0]) + 1e-12)
    for i in range(1, len(q)):
        if np.dot(q[i-1], q[i]) < 0.0:
            q[i] = -q[i]
        n = np.linalg.norm(q[i])
        if n == 0.0:
            q[i] = q[i-1]
        else:
            q[i] = q[i] / n
    return q

def quat_to_rpy_zyx(q):
    q = np.asarray(q, dtype=float)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    t0 = 2.0*(w*x + y*z)
    t1 = 1.0 - 2.0*(x*x + y*y)
    roll = np.arctan2(t0, t1)
    t2 = 2.0*(w*y - z*x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    t3 = 2.0*(w*z + x*y)
    t4 = 1.0 - 2.0*(y*y + z*z)
    yaw = np.arctan2(t3, t4)
    return np.column_stack((roll, pitch, yaw))

def relative_quat(q_est, q_true):
    qrel = quat_multiply(quat_conjugate(q_true), q_est)
    mask = qrel[:, 0] < 0.0
    qrel[mask] = -qrel[mask]
    return qrel

def compute_errors(df):
    x1_est = df[["est_x1_x", "est_x1_y", "est_x1_z"]].to_numpy()
    x1_true = df[["true_x1_x", "true_x1_y", "true_x1_z"]].to_numpy()
    x2_est = df[["est_x2_x", "est_x2_y", "est_x2_z"]].to_numpy()
    x2_true = df[["true_x2_x", "true_x2_y", "true_x2_z"]].to_numpy()
    p_est = df[["est_px", "est_py", "est_pz"]].to_numpy()
    p_true = df[["true_px", "true_py", "true_pz"]].to_numpy()
    b_est = df[["est_bx", "est_by", "est_bz"]].to_numpy()
    b_true = df[["true_bx", "true_by", "true_bz"]].to_numpy()

    err_x1 = np.linalg.norm(x1_est - x1_true, axis=1)

    l_hat = x2_est / np.where(np.linalg.norm(x2_est, axis=1, keepdims=True) == 0.0, 1.0, np.linalg.norm(x2_est, axis=1, keepdims=True))
    l_gt  = x2_true / np.where(np.linalg.norm(x2_true, axis=1, keepdims=True) == 0.0, 1.0, np.linalg.norm(x2_true, axis=1, keepdims=True))
    dot = np.sum(l_gt * l_hat, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    err_x2 = np.arccos(dot)

    err_p = np.linalg.norm(p_est - p_true, axis=1)
    err_b = np.linalg.norm(b_est - b_true, axis=1)

    have_quats = {"est_qw","est_qx","est_qy","est_qz","true_qw","true_qx","true_qy","true_qz"}.issubset(df.columns)
    if have_quats:
        q_est = df[["est_qw","est_qx","est_qy","est_qz"]].to_numpy()
        q_true = df[["true_qw","true_qx","true_qy","true_qz"]].to_numpy()
        err_att = quat_angle_error(q_est, q_true)
    else:
        est_rpy = df[["est_roll","est_pitch","est_yaw"]].to_numpy()
        true_rpy = df[["true_roll","true_pitch","true_yaw"]].to_numpy()
        err_att = np.linalg.norm(wrap_pi(est_rpy - true_rpy), axis=1)

    return {"err_x1": err_x1, "err_x2": err_x2, "err_p": err_p, "err_b": err_b, "err_att": err_att}

def add_xyz(fig, t, est_cols, true_cols, df, idx_for_x, use_p=False):
    comps = ("x", "y", "z")
    for comp, ec, tc in zip(comps, est_cols, true_cols):
        if use_p:
            est_name = f"$\\Large \\hat{{p}}_{{l,{comp}}}$"
            tru_name = f"$\\Large p_{{l,{comp}}}$"
        else:
            est_name = f"$\\Large \\hat{{x}}_{{{idx_for_x},{comp}}}$"
            tru_name = f"$\\Large x_{{{idx_for_x},{comp}}}$"
        fig.add_trace(go.Scatter(x=t, y=df[ec], mode="lines", name=est_name, line=dict(width=2.5)))
        fig.add_trace(go.Scatter(x=t, y=df[tc], mode="lines", name=tru_name, line=dict(dash="dash", width=2.5)))

def plot_run(df, run_title):
    t = df["t"].to_numpy()

    fig_x1 = go.Figure()
    add_xyz(fig_x1, t, ("est_x1_x","est_x1_y","est_x1_z"), ("true_x1_x","true_x1_y","true_x1_z"), df, idx_for_x="1", use_p=False)
    fig_x1.update_yaxes(title_text="Local linear velocity [m/s]")
    style_layout(fig_x1); fig_x1.show(); save_fig(fig_x1, "x1_components")

    fig_x2 = go.Figure()
    add_xyz(fig_x2, t, ("est_x2_x","est_x2_y","est_x2_z"), ("true_x2_x","true_x2_y","true_x2_z"), df, idx_for_x="2", use_p=False)
    fig_x2.update_yaxes(title_text="Tilt [rad]")
    style_layout(fig_x2); fig_x2.show(); save_fig(fig_x2, "x2_components")

    fig_p = go.Figure()
    add_xyz(fig_p, t, ("est_px","est_py","est_pz"), ("true_px","true_py","true_pz"), df, idx_for_x="", use_p=True)
    fig_p.update_yaxes(title_text="Position [m]")
    style_layout(fig_p); fig_p.show(); save_fig(fig_p, "pl_components")

    # New: gyrometer bias components (estimate vs. true)
    fig_bcomps = go.Figure()
    comps = ("x", "y", "z")
    for comp, ec, tc in zip(comps, ("est_bx","est_by","est_bz"), ("true_bx","true_by","true_bz")):
        est_name = f"$\\Large \\hat{{b}}_{{{comp}}}$"
        tru_name = f"$\\Large b_{{{comp}}}$"
        fig_bcomps.add_trace(go.Scatter(x=t, y=df[ec], mode="lines", name=est_name, line=dict(width=2.5)))
        fig_bcomps.add_trace(go.Scatter(x=t, y=df[tc], mode="lines", name=tru_name, line=dict(dash="dash", width=2.5)))
    fig_bcomps.update_yaxes(title_text="Gyrometer Bias [rad/s]")
    style_layout(fig_bcomps); fig_bcomps.show(); save_fig(fig_bcomps, "b_components")

    have_quats = {"est_qw","est_qx","est_qy","est_qz","true_qw","true_qx","true_qy","true_qz"}.issubset(df.columns)
    if have_quats:
        q_est = quat_hemisphere(df[["est_qw","est_qx","est_qy","est_qz"]].to_numpy())
        q_true = quat_hemisphere(df[["true_qw","true_qx","true_qy","true_qz"]].to_numpy())
        est_rpy = quat_to_rpy_zyx(q_est)
        true_rpy = quat_to_rpy_zyx(q_true)
    else:
        est_rpy = df[["est_roll","est_pitch","est_yaw"]].to_numpy()
        true_rpy = df[["true_roll","true_pitch","true_yaw"]].to_numpy()

    est_rpy_cont = continuous_euler(est_rpy)
    true_rpy_cont = continuous_euler(true_rpy)

    errs = compute_errors(df)

    fig_all_errs = go.Figure()
    fig_all_errs.add_trace(go.Scatter(x=t, y=errs["err_x1"], mode="lines", name=r"$\Large \|\tilde{x}_1\|\,[\mathrm{m}.\mathrm{s}^{-1}]$", line=dict(width=2.5)))
    fig_all_errs.add_trace(go.Scatter(x=t, y=errs["err_x2"], mode="lines", name=r"$\Large \|\tilde{x}_2\|\,[\mathrm{rad}]$", line=dict(width=2.5)))
    fig_all_errs.add_trace(go.Scatter(x=t, y=errs["err_p"],  mode="lines", name=r"$\Large \|\tilde{\mathbf{p}}_{\ell}\|\,[\mathrm{m}]$", line=dict(width=2.5)))

    have_quats = {"est_qw","est_qx","est_qy","est_qz","true_qw","true_qx","true_qy","true_qz"}.issubset(df.columns)
    if have_quats:
        q_est = quat_hemisphere(df[["est_qw","est_qx","est_qy","est_qz"]].to_numpy())
        q_true = quat_hemisphere(df[["true_qw","true_qx","true_qy","true_qz"]].to_numpy())
        q_rel = relative_quat(q_est, q_true)
        q_vec_norm = np.linalg.norm(q_rel[:, 1:], axis=1)
        fig_all_errs.add_trace(go.Scatter(x=t, y=q_vec_norm, mode="lines", name=r"$\Large \|\tilde{\mathbf{q}}_{v}\|$", line=dict(width=2.5)))

    fig_all_errs.add_trace(go.Scatter(x=t, y=errs["err_b"], mode="lines", name=r"$\Large \|\tilde{\mathbf{b}}\|\,[\mathrm{rad}.\mathrm{s}^{-1}]$", line=dict(width=2.5)))

    fig_all_errs.update_yaxes(title_text="Errors (mixed units)")
    style_layout(fig_all_errs); fig_all_errs.show(); save_fig(fig_all_errs, "all_errors_x1_x2_pl_b_qnorm")
 
 

def main(path=CSV_PATH):
    if not os.path.exists(path):
        raise SystemExit(
            f"No CSV found at: {path}\n"
            f"- Run your C++ test to generate {path}\n"
            f"- Or call main('<your_csv_path>') to point to another file."
        )
    df = pd.read_csv(path)
    run_title = os.path.basename(path)
    plot_run(df, run_title)

if __name__ == "__main__":
    main()

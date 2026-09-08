import sys, glob, yaml; sys.path.insert(0,'.')
from pathlib import Path
import kinetics_eval as ke
RUN=Path(sorted(glob.glob("../results/W_tuned_a1_posx30-*"))[0])
name="KO_TRO2024_RHPS1_1"
proj,cache = ke.project_paths(name) if hasattr(ke,"project_paths") else (None,None)
import kinetics_tune as kt
proj,cache = kt.project_paths(name)
out = RUN/name/"eval_multi"
ke.analyze(RUN/name/"kinetics.txt", cache/"reference/mocap.txt", out, [1.0,2.0,3.0])
ri = proj/"output_data/evals/Hartley/saved_results/traj_est"
print(f"\n{name}: yaw RPE [deg] vs sub-trajectory length\n")
print(f"{'length':>8}{'KO':>10}{'RI-EKF':>10}{'ratio':>9}   {'KO trans_xy':>12}{'RI trans_xy':>12}{'ratio':>9}")
for L in (1.0,2.0,3.0):
    tag=f"{L:.1f}".replace('.','_')
    ko=yaml.safe_load((out/"saved_results/traj_est"/f"relative_error_statistics_{tag}.yaml").read_text())
    rr=yaml.safe_load((ri/f"relative_error_statistics_{tag}.yaml").read_text())
    ky,ry=ko["yaw"]["mean"], rr["yaw"]["mean"]
    kt_,rt=ko["trans_x_y_norm"]["mean"], rr["trans_x_y_norm"]["mean"]
    print(f"{L:8.1f}{ky:10.4f}{ry:10.4f}{ky/ry:9.3f}   {kt_*1000:12.2f}{rt*1000:12.2f}{kt_/rt:9.3f}")

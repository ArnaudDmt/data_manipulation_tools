#!/usr/bin/env python3
"""Evaluate a wrench-process-noise variant from a saved covariance overlay."""

import argparse
import csv
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import kinetics_tune as kt


GROUPS = {
    "MultiContact": [f"HRP5_MultiContact_{i}" for i in range(1, 5)],
    "LongWalk": ["HRP5P_LongWalk"],
    "RHPS1 walk": [f"KO_TRO2024_RHPS1_{i}" for i in range(1, 6)],
    "RHPS1 slip": [f"KO_TRO_2024_RHPS1_SLIPPAGE_{i}" for i in range(1, 4)],
}
POSE_METRICS = ("trans_xy", "trans_z", "yaw", "tilt")


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-overlay", type=Path,
                        default=ROOT / "results/best-accepted-covariance-overlay.backup.yaml")
    parser.add_argument("--force-std", type=float, default=150.0,
                        help="force process-noise standard deviation in N")
    parser.add_argument("--torque-std", type=float, default=100.0,
                        help="torque process-noise standard deviation in N.m")
    parser.add_argument("--label", default=f"wrench-{time.strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--workspace", default="/home/arnaud/devel/src/catkin_ws")
    args = parser.parse_args()
    if not all(math.isfinite(x) and x >= 0 for x in (args.force_std, args.torque_std)):
        parser.error("standard deviations must be finite and non-negative")

    overlay = yaml.safe_load(args.base_overlay.read_text(encoding="utf-8"))
    if len(overlay["covariances"]["unmodeled_wrench_process"]) != 6:
        raise ValueError("expected 3 force and 3 torque covariance entries")
    overlay["covariances"]["unmodeled_wrench_process"] = (
        [args.force_std ** 2] * 3 + [args.torque_std ** 2] * 3)
    projects = [project for group in GROUPS.values() for project in group]

    with TemporaryDirectory() as temporary:
        overlay_path = Path(temporary) / "covariance_overlay.yaml"
        overlay_path.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
        subprocess.run([
            sys.executable, str(ROOT / "scripts/kinetics_eval.py"),
            "--workspace", args.workspace, "--projects", ",".join(projects),
            "--covariance-overlay", str(overlay_path), "run", "--label", args.label,
            "--no-plots", "--no-latest", "--no-open",
        ], cwd=ROOT, check=True)

    outputs = sorted(ROOT.glob(f"results/{args.label}-*"), key=lambda p: p.stat().st_mtime)
    if not outputs or not (outputs[-1] / "summary.csv").exists():
        raise FileNotFoundError(f"no evaluation output for {args.label}")
    output = outputs[-1]
    rows = list(csv.DictReader((output / "summary.csv").open(newline="", encoding="utf-8")))
    print(f"output: {output}")
    print(f"process std: force={args.force_std:g} N, torque={args.torque_std:g} N.m")
    for group, names in GROUPS.items():
        values = defaultdict(list)
        for row in rows:
            if row["project"] in names and row["statistic"] == "mean":
                scale = 1000 if row["metric"] in ("trans_xy", "trans_z") else 1
                values[(row["estimator"], row["metric"])].append(float(row["value"]) * scale)
        velocity = defaultdict(list)
        for name in names:
            grid, mocap, riekf, settings = kt.reference_velocities([name])[name]
            pose = np.loadtxt(output / name / "kinetics.txt", comments="#", ndmin=2)
            estimated = np.loadtxt(output / name / "kinetics_velocity.txt", comments="#", ndmin=2)
            velocity["Kinetics"].append(kt.velocity_errors(
                kt.estimated_local_velocity(pose, estimated, grid, settings), mocap))
            velocity["RI-EKF"].append(riekf)
        print(group)
        for metric in POSE_METRICS:
            ko = mean(values[("Kinetics", metric)])
            ri = mean(values[("RI-EKF", metric)])
            print(f"  {metric:8s} KO={ko:.3f} RI={ri:.3f}")
        print(f"  {'vel_xy':8s} KO={mean(x[0] for x in velocity['Kinetics']):.5f} "
              f"RI={mean(x[0] for x in velocity['RI-EKF']):.5f}")
        print(f"  {'vel_z':8s} KO={mean(x[1] for x in velocity['Kinetics']):.5f} "
              f"RI={mean(x[1] for x in velocity['RI-EKF']):.5f}")


if __name__ == "__main__":
    main()

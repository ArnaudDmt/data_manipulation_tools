#!/usr/bin/env python3
"""Sanity-check a tuned Kinetics result against RI-EKF beyond the relative pose metrics.

Relative pose errors reward an estimator that tracks the local shape of the trajectory, and an
over-damped tuning can win on them while lagging badly in velocity or drifting in Z. This
compares those two quantities between a tuned run and a baseline run, so a winning tuning can
be rejected if it bought its pose numbers with sluggishness.

Usage: kinetics_check.py <tuned-results-dir> <baseline-results-dir>
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from kinetics_eval import PROJECTS, project_paths


def load(path):
    return np.loadtxt(path, comments="#", ndmin=2)


def yaw_align(estimate, reference):
    """The planar rotation the plots use, so Z and speed are compared in the reference frame."""
    relative = (Rotation.from_quat(estimate[0, 4:8]).as_matrix() @
                Rotation.from_quat(reference[0, 4:8]).as_matrix().T)
    theta = np.pi / 2 - np.arctan2(relative[0, 0] + relative[1, 1], relative[0, 1] - relative[1, 0])
    return Rotation.from_euler("z", theta)


def resample(times, values, targets):
    return np.column_stack([np.interp(targets, times, values[:, axis]) for axis in range(values.shape[1])])


def metrics(trajectory, velocity_path, reference):
    rotation = yaw_align(trajectory, reference)
    times = reference[:, 0]
    position = resample(trajectory[:, 0], rotation.apply(trajectory[:, 1:4]), times)
    position += reference[0, 1:4] - position[0]
    drift = float(np.sqrt(np.mean((position[:, 2] - reference[:, 3]) ** 2)))
    reference_velocity = np.gradient(reference[:, 1:4], times, axis=0)
    if velocity_path is not None and Path(velocity_path).exists():
        raw = load(velocity_path)
        velocity = resample(raw[:, 0], rotation.apply(raw[:, 1:4]), times)
    else:
        velocity = np.gradient(position, times, axis=0)
    speed = float(np.sqrt(np.mean(np.linalg.norm(velocity - reference_velocity, axis=1) ** 2)))
    return drift, speed


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    tuned, baseline = Path(sys.argv[1]), Path(sys.argv[2])
    print(f"{'dataset':32} {'Z rms drift [m]':>26}   {'velocity rms [m/s]':>26}")
    print(f"{'':32} {'base':>8} {'tuned':>8} {'riekf':>8}   {'base':>8} {'tuned':>8} {'riekf':>8}")
    totals = {"base": [], "tuned": [], "riekf": []}
    for name in PROJECTS:
        project, cache = project_paths(name)
        reference = load(cache / "reference/mocap.txt")
        riekf = load(project / "output_data/evals/Hartley/stamped_traj_estimate.txt")
        row = {}
        for key, directory in (("base", baseline), ("tuned", tuned)):
            path = directory / name / "kinetics.txt"
            if not path.exists():
                raise SystemExit(f"missing {path}")
            row[key] = metrics(load(path), directory / name / "kinetics_velocity.txt", reference)
        row["riekf"] = metrics(riekf, None, reference)
        for key in totals:
            totals[key].append(row[key])
        print(f"{name:32} " + " ".join(f"{row[k][0]:8.4f}" for k in ("base", "tuned", "riekf")) +
              "   " + " ".join(f"{row[k][1]:8.4f}" for k in ("base", "tuned", "riekf")))
    print(f"{'MEAN':32} " + " ".join(f"{np.mean([v[0] for v in totals[k]]):8.4f}" for k in ("base", "tuned", "riekf")) +
          "   " + " ".join(f"{np.mean([v[1] for v in totals[k]]):8.4f}" for k in ("base", "tuned", "riekf")))


if __name__ == "__main__":
    main()

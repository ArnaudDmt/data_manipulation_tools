"""Keep a decimated copy of a run's estimated and true trajectory.

The full `stamped_traj_estimate.txt` is 70 MB a trial and is rewritten by the next variant, so a
question about the *shape* of a run's error -- steady drift or slow wander, at what frequency --
cannot be answered after the fact. The relative-error cache cannot answer it either: it holds
statistics, not a signal.

Decimating to 10 Hz costs 2 MB a trial and keeps everything that matters below the gait frequency.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_HZ = 10.0


def keep(evals, observer, destination):
    source = Path(evals) / observer
    estimate = source / "stamped_traj_estimate.txt"
    truth = source / "stamped_groundtruth.txt"
    if not estimate.exists() or not truth.exists():
        return False
    read = lambda p: pd.read_csv(p, sep=r"\s+", comment="#", header=None).to_numpy()
    e, g = read(estimate), read(truth)
    n = min(len(e), len(g))
    e, g = e[:n], g[:n]
    dt = np.median(np.diff(e[:, 0]))
    step = max(1, int(round(1.0 / (TARGET_HZ * dt))))
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, estimate=e[::step].astype(np.float32),
                        truth=g[::step].astype(np.float32), hz=1.0 / (dt * step))
    return True


def main(argv):
    if len(argv) != 3:
        print("usage: keep_traj.py <evals dir> <destination dir>")
        return 2
    evals, destination = argv[1], argv[2]
    kept = []
    for observer in ("KO", "Hartley"):
        if keep(evals, observer, Path(destination) / f"{observer}_traj10.npz"):
            kept.append(observer)
    print(f"trajectoires decimees : {', '.join(kept) if kept else 'aucune'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

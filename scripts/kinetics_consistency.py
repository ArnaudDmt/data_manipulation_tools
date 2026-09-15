#!/usr/bin/env python3
"""Report the Kinetics estimate's internal self-consistency for one or more result directories.

    scripts/kinetics_consistency.py results/<label>-<hash> [more...]

rms(v - dp/dt) / rms(dp/dt), where both sides come from the estimate itself. Because no
reference is involved, this separates observer behaviour from anything the evaluation harness
does about time alignment or ground truth. A value near 0 means the published velocity is the
derivative of the published position; the baseline sits at 0.18 on HRP5 and 0.35 on RHPS1.

Diagnostic only -- see self_consistency() in kinetics_tune.py for why it must not be optimised.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kinetics_eval import PROJECTS
from kinetics_tune import self_consistency


def main():
    directories = [Path(argument) for argument in sys.argv[1:]]
    if not directories:
        print(__doc__.strip())
        return 2
    import numpy as np
    for directory in directories:
        present = [name for name in PROJECTS if (directory / name / "kinetics_velocity.txt").exists()]
        if not present:
            print(f"{directory.name}: no result files")
            continue
        values = self_consistency(directory, present)
        numbers = [values[f"{name}|consistency"] for name in present]
        print(f"\n{directory.name}  ({len(present)}/{len(PROJECTS)} datasets, "
              f"geomean {float(np.exp(np.mean(np.log(numbers)))):.4f})")
        for name in present:
            print(f"   {name:34s} {values[f'{name}|consistency']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

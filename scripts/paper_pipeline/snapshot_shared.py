"""Save each named observer from the shared replay in the existing per-variant layout."""
import os
import shutil
import sys

import config_home
import manifest as m
from keep_traj import keep


def snapshot(project):
    out = m.ROOT / "Projects" / project / "output_data"
    for variant, observer in m.SHARED_ROUTINE_OBSERVERS.items():
        store = m.WORK / "runs" / variant / project
        store.mkdir(parents=True, exist_ok=True)
        if "KO_CONFIG_HOME" in os.environ:
            config_home.keep_provenance(os.environ["KO_CONFIG_HOME"], store)
        cache = out / "evals" / observer / "saved_results/traj_est/cached/cached_rel_err.pickle"
        if cache.exists():
            shutil.copy(cache, store / "cached_rel_err.pickle")
        elif project in m.ALL:
            raise FileNotFoundError(cache)
        for source, target in ((observer, "KO"), ("Hartley", "Hartley"),
                               ("Tilt", "Tilt"), ("mocap", "mocap")):
            shutil.copy(out / f"{source}_loc_vel.pickle", store / f"{target}_loc_vel.pickle")
        for source, target in ((observer, "KO"), ("Hartley", "Hartley")):
            if not keep(out / "evals", source, store / f"{target}_traj10.npz"):
                raise FileNotFoundError(out / "evals" / source)


if __name__ == "__main__":
    snapshot(sys.argv[1])

"""Check that the Kinetics Observer on this machine is still the one that produced the paper.

    .venv/bin/python scripts/paper_pipeline/verify_paper_ko.py            # hashes only, instant
    .venv/bin/python scripts/paper_pipeline/verify_paper_ko.py --replay   # + replay two datasets

Without options it compares, against paper_ko.lock.json, the md5 of the installed binaries and the
sha256 of every file of config_base/, and says whether ~/.config/mc_rtc still matches config_base/
(the replay reads ~/.config by default, not config_base/).

--replay replays HRP5_MultiContact_1 and KO_TRO2024_RHPS1_1 through kinetics_eval.py in an
isolated cache (output_data/kinetics_eval_verif, built from the existing logReplay_full.bin, never
from the original bag) and compares the rpg means with the reference values of the lock file.
Replay and routine agree to 5-6 digits on these 200 Hz datasets, hence the 1e-4 relative tolerance.
It is the first thing to run before building a new variant; see README.md.
"""
import argparse
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCK = json.loads((HERE / "paper_ko.lock.json").read_text())
BASE = HERE / "config_base"
LIVE = Path.home() / ".config/mc_rtc"
TOLERANCE = 1e-4
METRICS = ("rel_trans_x_y_norm", "rel_trans_z", "rel_yaw", "rel_tilt", "rel_rot", "rel_gravity")


def digest(path, algorithm):
    return hashlib.new(algorithm, Path(path).read_bytes()).hexdigest()


def check_hashes():
    ok = True
    for path, expected in LOCK["binaries"].items():
        if not path.startswith("/"):
            continue
        current = digest(path, "md5") if Path(path).exists() else "absent"
        same = current == expected
        ok &= same
        print(f"{'ok   ' if same else 'DIFF '} {path}")
    # The package's own configuration, installed with the observers: the lowest layer, below
    # config_base/. A reinstall from another revision would change it silently.
    for path, expected in LOCK.get("installed_package_config_sha256", {}).items():
        if not path.startswith("/"):
            continue
        current = digest(path, "sha256") if Path(path).exists() else "absent"
        same = current == expected
        ok &= same
        print(f"{'ok   ' if same else 'DIFF '} {path}")
    for name, expected in LOCK["config_base_sha256"].items():
        current = digest(BASE / name, "sha256") if (BASE / name).exists() else "absent"
        same = current == expected
        ok &= same
        print(f"{'ok   ' if same else 'DIFF '} config_base/{name}")
    # Informational: the replay's default configuration layer is ~/.config/mc_rtc, not config_base/.
    # Compared by CONTENT (comments do not matter). withDebugLogs is overridden by the controller's
    # inline block, and of Passthrough.yaml the replay reads only the unnamed MCKineticsObserver
    # block (kinetics_eval.pipeline_observer_config); VALINOR's block does not reach it.
    import yaml

    def content(path, name):
        # mc_rtc tolerates tabs, PyYAML does not (same workaround as kinetics_eval.read_yaml)
        data = yaml.safe_load(path.read_text().replace("\t", " ")) or {}
        if name.endswith("Passthrough.yaml"):
            pipelines = data.get("ObserverPipelines")
            pipelines = pipelines if isinstance(pipelines, list) else [pipelines]
            return [o.get("config") for p in pipelines for o in p.get("observers", [])
                    if o.get("type") == "MCKineticsObserver" and "name" not in o]
        if isinstance(data, dict):
            data.pop("withDebugLogs", None)
        return data

    for name in LOCK["config_base_sha256"]:
        live = LIVE / name
        if name.endswith("MocapAligner.yaml") or not live.exists():
            continue
        if content(live, name) != content(BASE / name, name):
            print(f"note  ~/.config/mc_rtc/{name} differs from config_base/ in what the replay reads: "
                  f"a replay without --observer-config would not run the paper's tuning")
    return ok


def replay():
    python = str(ROOT / ".venv/bin/python")
    env = {**os.environ, "ROS_DOMAIN_ID": os.environ.get("ROS_DOMAIN_ID", "77")}
    ok = True
    for project, reference in LOCK["reference_single_datasets"].items():
        cache = ROOT / "Projects" / project / "output_data/kinetics_eval_verif"
        cache.mkdir(parents=True, exist_ok=True)
        source = cache / "logReplay_full.bin"
        if not source.exists():
            source.symlink_to("../kinetics_eval/logReplay_full.bin")
        common = [python, "scripts/kinetics_eval.py", "--projects", project, "--cache-suffix", "_verif"]
        subprocess.run([*common, "prepare", "--no-regenerate"], cwd=ROOT, env=env, check=True,
                       stdout=subprocess.DEVNULL)
        output = subprocess.run([*common, "run", "--label", "paper-verif", "--no-plots", "--no-open",
                                 "--no-latest"], cwd=ROOT, env=env, check=True,
                                capture_output=True, text=True).stdout
        results = Path(re.search(r"^Results: (.+)$", output, re.M).group(1))
        pickle_path = results / project / "eval/saved_results/traj_est/cached/cached_rel_err.pickle"
        data = pickle.load(pickle_path.open("rb"))
        block = data.get(reference["sublength_m"], data.get(str(reference["sublength_m"])))
        print(f"{project} ({reference['sublength_m']} m, {results.name})")
        for metric in METRICS:
            values = np.abs(np.asarray(block[metric], float))
            got, want = values.mean(), reference[metric]
            same = abs(got - want) <= TOLERANCE * abs(want) and values.size == reference["count"]
            ok &= same
            print(f"  {'ok   ' if same else 'DIFF '} {metric:20s} {got:.6f}  reference {want:.6f}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replay", action="store_true", help="also replay two datasets (minutes)")
    args = parser.parse_args()
    ok = check_hashes()
    if args.replay:
        ok &= replay()
    print("\nCONFORME au papier" if ok else "\nNON CONFORME: something moved since the paper; "
          "find out what before building on it")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

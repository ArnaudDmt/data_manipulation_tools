"""A private HOME per run, so that no experiment ever edits ~/.config/mc_rtc again.

mc_rtc reads every user configuration from $HOME/.config/mc_rtc (mc_rtc/path.cpp:19, observers in
ObserverPipeline.cpp:45, controllers in mc_global_controller_configuration.cpp:374, plugins in
MCController.cpp:1111) and the offline RI-EKF parser reads $HOME/.config/mc_rtc/plugins/HartleyIEKF.yaml
(kinematics.cpp:71). Pointing HOME at a directory built here is therefore enough to give one run
its own configuration, without touching the one installed for everything else.

The private HOME mirrors the real one entry by entry through symlinks -- ROS, Python, the mc_rtc
module cache under ~/.local keep working -- except .config/mc_rtc, which is a real copy: the
installed tree, overwritten by the versioned base in config_base/ (the files the paper pipeline
depends on), on which a variant is then applied. The digest of those files identifies the run.

Before this, variants were written into ~/.config and restored from a snapshot afterwards. A crash
left a variant installed, and a snapshot taken once and never refreshed silently reverted the
adoption of a new setting (2026-09-16: the RI-EKF went back to gyroBiasInitVariance 1e-8).
"""
import hashlib
import os
import shutil
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent / "config_base"
REAL_HOME = Path(os.environ.get("KO_REAL_HOME", str(Path.home())))
IGNORED = shutil.ignore_patterns(".git", "build", "*.pre-*", "*.bak*", "*.before-*", "*.with-*", "*.avant-*")


def tracked():
    return sorted(p.relative_to(BASE) for p in BASE.rglob("*") if p.is_file())


def mc_rtc(home):
    return Path(home) / ".config/mc_rtc"


def materialize(home):
    """Build a private HOME at `home` holding the versioned base configuration."""
    home = Path(home)
    if home.resolve() == REAL_HOME.resolve():
        raise RuntimeError("refusing to materialize over the real HOME")
    if home.exists():
        shutil.rmtree(home)
    home.mkdir(parents=True)
    for entry in REAL_HOME.iterdir():
        if entry.name != ".config":
            (home / entry.name).symlink_to(entry)
    (home / ".config").mkdir()
    for entry in (REAL_HOME / ".config").iterdir():
        if entry.name != "mc_rtc":
            (home / ".config" / entry.name).symlink_to(entry)
    shutil.copytree(REAL_HOME / ".config/mc_rtc", mc_rtc(home), symlinks=True, ignore=IGNORED)
    for relative in tracked():
        target = mc_rtc(home) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(BASE / relative, target)
    return home


def observer_aliases():
    """Map each paper abbreviation to its controller instance, e.g. KO_ZPC -> KOZPC.

    Derived from observersInfos.yaml instead of hardcoded: its log keys are shaped
    `Observers_MainObserverPipeline_<instance>_...`, and the instance is exactly the `name:` of
    the controller entry (or its `type:` when the entry has no name). Deriving it means a new
    instance needs no edit here.
    """
    infos = yaml.safe_load((BASE.parents[2] / "observersInfos.yaml").read_text())
    aliases = {}
    for observer in infos["observers"]:
        for group in observer.get("kinematics", {}).values():
            keys = [v[0] if isinstance(v, list) else v for v in group.values()]
            for key in keys:
                parts = str(key).split("_")
                if len(parts) > 2 and parts[0] == "Observers":
                    aliases[observer["abbreviation"]] = parts[2]
                    break
            if observer["abbreviation"] in aliases:
                break
    return aliases


def select_observers(home, selection):
    """Keep only `selection` in the materialised Passthrough.yaml of `home`.

    Re-ticking one estimator otherwise recomputes all of them, which costs time and -- the sharper
    problem -- lets a variant's per-robot configuration reach instances it was never meant for.
    Selecting here rather than by commenting the shared file means the choice is explicit, lands in
    the private HOME only, and is covered by digest(): two runs with different selections cannot be
    confused for one another.

    Refuses rather than produces a wrong result:
      * the Encoder is structural and always kept;
      * the last observer with `update: true` is the realRobot, and the mocap cross-correlation is
        aligned on it -- dropping it would silently shift every number in the run;
      * an unknown name is a typo, and silently keeping nothing would look like a clean result.

    The rewrite drops the file's comments, which is why it is applied only when a selection is
    asked for: without one the materialised file stays byte-identical to the versioned base.
    """
    path = mc_rtc(home) / "controllers/Passthrough.yaml"
    document = yaml.safe_load(path.read_text())
    entries = document["ObserverPipelines"]["observers"]

    def identity(entry):
        return entry.get("name") or entry["type"]

    aliases = observer_aliases()
    wanted = {aliases.get(name, name) for name in selection}
    present = {identity(entry) for entry in entries}
    unknown = wanted - present
    if unknown:
        raise RuntimeError(f"unknown observer(s) {sorted(unknown)}; the pipeline has {sorted(present)}")

    updating = [identity(e) for e in entries if e.get("update")]
    if updating and updating[-1] not in wanted:
        raise RuntimeError(f"refusing to drop {updating[-1]}: it is the realRobot the mocap is "
                           f"aligned on, so every result would move. Add it to the selection.")

    kept = [e for e in entries if identity(e) in wanted or e["type"] == "Encoder"]
    document["ObserverPipelines"]["observers"] = kept
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return [identity(e) for e in kept]


def drop_plugins(home, names):
    """Remove `names` from the Plugins list of the materialised mc_rtc.yaml.

    This is how the RI-EKF is excluded: it is not an observer of the pipeline but the HartleyIEKF
    plugin, which writes HartleyInput.txt during the tick.
    """
    path = mc_rtc(home) / "mc_rtc.yaml"
    document = yaml.safe_load(path.read_text())
    document["Plugins"] = [p for p in document.get("Plugins", []) if p not in set(names)]
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return document["Plugins"]


def digest(home):
    """sha256 over the tracked configuration files as they stand in `home`."""
    h = hashlib.sha256()
    for relative in tracked():
        h.update(str(relative).encode() + b"\0" + (mc_rtc(home) / relative).read_bytes() + b"\0")
    return h.hexdigest()


def keep_provenance(home, destination):
    """Copy the tracked files and their digest next to a run's results."""
    destination = Path(destination)
    for relative in tracked():
        target = destination / "config" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(mc_rtc(home) / relative, target)
    (destination / "config.sha256").write_text(digest(home) + "\n")


if __name__ == "__main__":
    import sys
    print(digest(materialize(sys.argv[1])))

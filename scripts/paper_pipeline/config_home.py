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

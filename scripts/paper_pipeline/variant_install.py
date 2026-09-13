"""Write the observer configuration of one paper variant into ~/.config/mc_rtc/observers.

Every variant is the retained configuration plus one deliberate change, so they are all derived
from a pristine copy of it rather than from whatever the previous variant left installed.
"""
import re
import shutil
import sys
from pathlib import Path

CONFIG = Path.home() / ".config/mc_rtc/observers/MCKineticsObserver.yaml"
ROBOTS = Path.home() / ".config/mc_rtc/observers/MCKineticsObserver"
REFERENCE = Path(__file__).resolve().parents[2] / "results/var-clean-ref-f0b7eff088/MCKineticsObserver.yaml"
PRISTINE = Path(__file__).resolve().parents[2] / "results/paper-rebuild/pristine"
FLEX_KEYS = ("linStiffness", "angStiffness", "linDamping", "angDamping")


def snapshot():
    """Keep one untouched copy of the retained tuning, robot files included."""
    PRISTINE.mkdir(parents=True, exist_ok=True)
    if not (PRISTINE / "MCKineticsObserver.yaml").exists():
        shutil.copy(REFERENCE, PRISTINE / "MCKineticsObserver.yaml")
        for robot in ("hrp5_p", "rhps1"):
            shutil.copy(ROBOTS / f"{robot}.yaml", PRISTINE / f"{robot}.yaml")


def restore():
    shutil.copy(PRISTINE / "MCKineticsObserver.yaml", CONFIG)
    for robot in ("hrp5_p", "rhps1"):
        shutil.copy(PRISTINE / f"{robot}.yaml", ROBOTS / f"{robot}.yaml")


def scale_flexibilities(factor):
    """hrp5_p.yaml declares the stiffnesses twice, under `contacts:` and again at the root.

    MCKineticsObserver.cpp:186 reads the indented one, but a rewrite that misses either copy
    leaves the file self-contradictory, so every occurrence is scaled.
    """
    for robot in ("hrp5_p", "rhps1"):
        path = ROBOTS / f"{robot}.yaml"
        text = path.read_text()
        for key in FLEX_KEYS:
            def rescale(match, key=key):
                values = [float(item) * factor for item in match.group(2).split(",")]
                return f"{match.group(1)}[{', '.join(repr(value) for value in values)}]"
            text = re.sub(rf"(\s*{key}:\s*)\[([^\]]*)\]", rescale, text)
        path.write_text(text)


def set_unmodeled(value):
    """Overwrite both unmodeled wrench process variances with one scalar."""
    text = CONFIG.read_text()
    for key in ("unmodeledForceProcessVariance", "unmodeledTorqueProcessVariance"):
        text, n = re.subn(rf"(?m)^(\s*{key}:\s*)\[[^\]]*\]",
                          rf"\g<1>[{value}, {value}, {value}]", text)
        assert n == 1, f"{key} not found once ({n})"
    CONFIG.write_text(text)


def main(variant):
    snapshot()
    restore()
    # "hidehand+uw0.09" = the hidehand figure variant with the disturbance-wrench process moved.
    if "+uw" in variant:
        variant, _, value = variant.partition("+uw")
        set_unmodeled(float(value))
    if variant == "clean":
        pass
    elif variant == "pc":
        CONFIG.write_text("pinContacts: true\n\n" + CONFIG.read_text())
    elif variant == "zpc":
        text = CONFIG.read_text()
        text = re.sub(r"(?m)^(\s*contactPositionProcessVariance:\s*)\[[^\]]*\]",
                      r"\g<1>[0.0, 0.0, 0.0]", text)
        text = re.sub(r"(?m)^(\s*contactOrientationProcessVariance:\s*)\[[^\]]*\]",
                      r"\g<1>[0.0, 0.0, 0.0]", text)
        # A zero contact process covariance leaves nothing for the load weighting to distribute.
        text = re.sub(r"(?m)^contactCovLoadWeightExponent:.*$",
                      "contactCovLoadWeightExponent: 0.0", text)
        CONFIG.write_text(text)
    elif variant == "tightyaw":
        # The contact rest orientation's yaw process covariance is 1e-4, ten thousand times the
        # roll and pitch. Bring it down to theirs and see whether the yaw drift follows.
        text = CONFIG.read_text()
        text = re.sub(r"(?m)^(\s*contactOrientationProcessVariance:\s*)\[[^\]]*\]",
                      r"\g<1>[1e-08, 1e-08, 1e-08]", text)
        CONFIG.write_text(text)
    elif variant == "freezebz":
        # Freeze the z gyro bias the way the RI-EKF effectively does: no absolute yaw reference
        # makes it unobservable there, and it never leaves zero. Zero process and zero initial
        # variance pin it for the Kinetics Observer too.
        text = CONFIG.read_text()
        text = re.sub(r"(?m)^(\s*gyroBiasProcessVariance:\s*)\[[^\]]*\]",
                      r"\g<1>[1e-18, 1e-18, 0.0]", text)
        text = re.sub(r"(?m)^(\s*gyroBiasInitVariance:\s*)\[[^\]]*\]",
                      r"\g<1>[1e-08, 1e-08, 0.0]", text)
        CONFIG.write_text(text)
    elif variant == "hidehand":
        # Disturbance-wrench figure: the estimator is blind to the left hand, so the effort
        # measured there becomes a disturbance it has to recover.
        text = CONFIG.read_text()
        assert re.search(r"(?m)^contacts:$", text), "no contacts block to extend"
        text = re.sub(r"(?m)^contacts:$",
                      "contacts:\n  ignoredSensors: [LeftHandForceSensor]", text, count=1)
        CONFIG.write_text(text)
    elif variant.startswith("orierror"):
        # Paper experiment: every contact created after the start gets its rest orientation
        # rotated by a fixed angle about a random axis, and the estimator has to recover it.
        # The seed is pinned so the figure can be regenerated; the branch this comes from used an
        # unseeded Eigen::Vector3d::Random(), which no rerun could reproduce.
        degrees = float(variant[len("orierror"):] or 30)
        text = CONFIG.read_text()
        assert re.search(r"(?m)^contacts:$", text), "no contacts block to extend"
        text = re.sub(r"(?m)^contacts:$",
                      f"contacts:\n  restOrientationErrorDeg: {degrees}\n  restOrientationErrorSeed: 1",
                      text, count=1)
        CONFIG.write_text(text)
    elif variant in ("flexdiv10", "flexmul10"):
        scale_flexibilities(0.1 if variant == "flexdiv10" else 10.0)
    else:
        raise SystemExit(f"unknown variant: {variant}")
    print(f"installed {variant}")


if __name__ == "__main__":
    main(sys.argv[1])

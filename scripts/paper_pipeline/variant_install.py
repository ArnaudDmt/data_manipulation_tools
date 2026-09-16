"""Write the observer configuration of one paper variant into ~/.config/mc_rtc/observers.

Every variant is the retained configuration plus one deliberate change, so they are all derived
from a pristine copy of it rather than from whatever the previous variant left installed.
"""
import math
import os
import re
import shutil
import sys
from pathlib import Path

# Rebound by bind() when a variant is written into a private HOME (config_home.py) rather than
# into the real ~/.config.
MC = Path.home() / ".config/mc_rtc"
CONFIG = MC / "observers/MCKineticsObserver.yaml"
ROBOTS = MC / "observers/MCKineticsObserver"
# Files outside the observer tree that an experiment may still touch. They were NOT snapshotted
# before, so a variant that enabled the noise plugin or retuned the RI-EKF would have survived a
# crash and contaminated every later run -- the failure mode of the hidden-hand experiment.
#   mc_rtc.yaml            carries the Plugins list (NoisySensors is enabled there)
#   plugins/HartleyIEKF    the RI-EKF's own sensor variances, kept mirror of the KO's
#   plugins/NoisySensors   the synthetic IMU noise model
EXTRA = {
    "mc_rtc.yaml": MC / "mc_rtc.yaml",
    "HartleyIEKF.yaml": MC / "plugins/HartleyIEKF.yaml",
    "NoisySensors.yaml": MC / "plugins/NoisySensors.yaml",
}
# The retained tuning, kept with the pipeline. It must NOT point into results/var-clean-ref-*:
# that is an OUTPUT of the replay stage whose directory hash changes with the configuration.
REFERENCE = Path(__file__).resolve().parents[2] / "results/paper-rebuild/configs/clean/MCKineticsObserver.yaml"
PRISTINE = Path(__file__).resolve().parents[2] / "results/paper-rebuild/pristine"
FLEX_KEYS = ("linStiffness", "angStiffness", "linDamping", "angDamping")


def snapshot():
    """Keep an untouched copy of the retained tuning, robot files included.

    Refreshed whenever the reference is newer. The snapshot used to be taken once and kept
    forever, so changing the retained tuning would have left every variant silently running the
    previous one -- restore() reinstalls this copy before each variant.
    """
    PRISTINE.mkdir(parents=True, exist_ok=True)
    kept = PRISTINE / "MCKineticsObserver.yaml"
    if not kept.exists() or REFERENCE.stat().st_mtime > kept.stat().st_mtime:
        shutil.copy(REFERENCE, kept)
        for robot in ("hrp5_p", "rhps1"):
            shutil.copy(ROBOTS / f"{robot}.yaml", PRISTINE / f"{robot}.yaml")
    # Taken once and never refreshed from a live file afterwards: refreshing them from disk while
    # a variant is installed would freeze the variant as the new reference.
    for name, live in EXTRA.items():
        if live.exists() and not (PRISTINE / name).exists():
            shutil.copy(live, PRISTINE / name)


def restore():
    shutil.copy(PRISTINE / "MCKineticsObserver.yaml", CONFIG)
    for robot in ("hrp5_p", "rhps1"):
        shutil.copy(PRISTINE / f"{robot}.yaml", ROBOTS / f"{robot}.yaml")
    for name, live in EXTRA.items():
        if (PRISTINE / name).exists():
            shutil.copy(PRISTINE / name, live)


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


def main(variant, isolated=False):
    if not isolated:
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
    elif variant == "imunoise":
        # Degrade gyrometer AND accelerometer together, noise only, and widen both estimators'
        # measurement variances by the same amount. Measured at standstill on MultiContact_1, the
        # robot's own noise is gyro [6.3e-4, 7.4e-4, 1.9e-3] rad/s and accelerometer
        # [0.065, 0.092, 0.054] m/s2 -- the accelerometer is ALREADY noisier than a middle-grade
        # industrial MEMS, so degrading it means going to a consumer-grade unit.
        # Targets: gyro 2.057e-3 rad/s (ARW 0.5 deg/sqrt(h)), accelerometer 0.15 m/s2
        # (VRW ~0.64 m/s/sqrt(h)). Injected = sqrt(target^2 - existing^2). No bias: a calibrated
        # IMU has none at the start of a 14 s trial, and the in-run drift has not had time to grow.
        GYRO_INJECT = "[0.001958, 0.001919, 0.000857]"
        ACC_INJECT = "[0.13522, 0.1187, 0.14001]"
        GYRO_VAR = "4.2308e-06"
        ACC_VAR = "2.2500e-02"
        plugins = MC / "mc_rtc.yaml"
        text = plugins.read_text()
        text, n = re.subn(r"(?m)^Plugins:\s*\[([^\]]*)\]",
                          lambda m: "Plugins: [" + m.group(1) + ", NoisySensors]", text, count=1)
        assert n == 1, "no active Plugins entry in mc_rtc.yaml"
        plugins.write_text(text)

        noisefile = MC / "plugins/NoisySensors.yaml"
        text = noisefile.read_text()
        if re.search(r"(?m)^seed:", text):
            text = re.sub(r"(?m)^seed:.*$", "seed: 20260914", text)
        else:
            text = "seed: 20260914\n" + text
        for key, value in (("withNoisyGyro", "true"), ("withAcceleroNoise", "true"),
                           ("gyroNoise_StdDev", GYRO_INJECT), ("gyroOffset", "[0.0, 0.0, 0.0]"),
                           ("acceleroNoise_StdDev", ACC_INJECT), ("AcceleroOffset", "[0.0, 0.0, 0.0]")):
            text, n = re.subn(r"(?m)^" + key + r":.*$", key + ": " + value, text)
            assert n == 1, key + " not found once"
        noisefile.write_text(text)

        robot = ROBOTS / "hrp5_p.yaml"
        text = robot.read_text()
        for key, value in (("gyroSensorVariance", GYRO_VAR), ("acceleroSensorVariance", ACC_VAR)):
            triple = "[" + ", ".join([value] * 3) + "]"
            text, n = re.subn(r"(?m)^(\s*)" + key + r":\s*\[[^\]]*\]",
                              lambda m, k=key, t=triple: m.group(1) + k + ": " + t, text)
            assert n == 1, key + " not found once in hrp5_p.yaml"
        robot.write_text(text)

        hartley = MC / "plugins/HartleyIEKF.yaml"
        text = hartley.read_text()
        for key, old, value in (("gyroscopeVariance", "2.5e-7", GYRO_VAR),
                                ("accelerometerVariance", "2.5e-3", ACC_VAR)):
            text, n = re.subn(r"(?m)^(\s*)" + key + r":\s*" + re.escape(old) + r"(.*)$",
                              lambda m, k=key, v=value: m.group(1) + k + ": " + v + m.group(2), text)
            assert n == 1, key + " not found once in HartleyIEKF.yaml"
        hartley.write_text(text)
    elif variant.startswith("biasinit_"):
        # Same gyro-bias initial variance on BOTH estimators, to see what the pair does when the
        # bias state is given more room. The retained tuning is 1e-8 on each (the RI-EKF's config
        # file carried 1e-12 from 2026-08-28 to 2026-09-14, but its parses were produced with the
        # code default 1e-8, so the published numbers are unaffected).
        value = variant[len("biasinit_"):]
        text = CONFIG.read_text()
        text, n = re.subn(r"(?m)^(\s*gyroBiasInitVariance:\s*)\[[^\]]*\]",
                          rf"\g<1>[{value}, {value}, {value}]", text)
        assert n == 1, f"gyroBiasInitVariance not found once in the KO config ({n})"
        CONFIG.write_text(text)
        hartley = MC / "plugins/HartleyIEKF.yaml"
        text = hartley.read_text()
        text, n = re.subn(r"(?m)^gyroBiasInitVariance:.*$",
                          f"gyroBiasInitVariance: [{value}, {value}, {value}]", text)
        assert n == 1, f"gyroBiasInitVariance not found once in the RI-EKF config ({n})"
        hartley.write_text(text)
    elif variant.startswith("gyrobias_"):
        # Residual bias sweep. IMUs are calibrated before a campaign, so a 0.2 deg/s turn-on bias
        # is not what an experiment actually carries: what is left is the calibration residual and
        # the thermal drift since. HRP-5P's own bias measures 0.0003 deg/s on the yaw axis, so it
        # is effectively calibrated. The variant name carries the residual in deg/s, applied on the
        # three axes with the sign pattern of the turn-on model.
        degpersec = float(variant[len("gyrobias_"):].removesuffix("_asis"))
        magnitude = math.radians(degpersec)
        bias = f"[{magnitude:.4e}, {-magnitude:.4e}, {magnitude:.4e}]"
        plugins = MC / "mc_rtc.yaml"
        text = plugins.read_text()
        text, n = re.subn(r"(?m)^Plugins:\s*\[([^\]]*)\]",
                          lambda m: f"Plugins: [{m.group(1)}, NoisySensors]", text, count=1)
        assert n == 1, "no active Plugins entry in mc_rtc.yaml"
        plugins.write_text(text)
        noisefile = MC / "plugins/NoisySensors.yaml"
        text = noisefile.read_text()
        if re.search(r"(?m)^seed:", text):
            text = re.sub(r"(?m)^seed:.*$", "seed: 20260914", text)
        else:
            text = "seed: 20260914\n" + text
        for key, value in (("withNoisyGyro", "true"), ("withAcceleroNoise", "false"),
                           ("gyroNoise_StdDev", "[0.0, 0.0, 0.0]"), ("gyroOffset", bias)):
            text, n = re.subn(rf"(?m)^{key}:.*$", f"{key}: {value}", text)
            assert n == 1, f"{key} not found once ({n})"
        noisefile.write_text(text)
        # "_asis" keeps the bias covariances the paper publishes -- KO 1e-8 (std 1e-4 rad/s,
        # i.e. 0.006 deg/s) and RI-EKF 1e-12 -- instead of opening them. That is the configuration
        # every published number was produced with, and it is far too tight to estimate a bias of
        # a few hundredths of a degree per second.
        if variant.endswith("_asis"):
            return
        for path, pattern, replacement in (
                (CONFIG, r"(?m)^(\s*gyroBiasInitVariance:\s*)\[[^\]]*\]",
                 r"\g<1>[2.5e-05, 2.5e-05, 2.5e-05]"),
                (MC / "plugins/HartleyIEKF.yaml",
                 r"(?m)^gyroBiasInitVariance:.*$",
                 "gyroBiasInitVariance: [2.5e-5, 2.5e-5, 2.5e-5]")):
            text = path.read_text()
            text, n = re.subn(pattern, replacement, text)
            assert n == 1, f"gyroBiasInitVariance not found once in {path.name} ({n})"
            path.write_text(text)
    elif variant in ("noisygyro", "gyronoise", "gyrobias"):
        # HRP-5P carries a very good gyrometer, which is why the RI-EKF stays accurate through the
        # multicontact slippage: it trusts that gyrometer heavily. These variants degrade it to a
        # middle-grade industrial MEMS unit through the NoisySensors plugin, which rewrites the
        # sensor signal in the tick so BOTH estimators read the same corrupted values.
        #
        #   noisygyro  white noise + turn-on bias      gyronoise  noise only      gyrobias  bias only
        #
        # What is injected is what the signal LACKS, measured on the four multicontact trials with
        # the robot standing still (2 s, 401 samples each), so the TOTAL matches the target:
        #
        #        existing white noise        existing bias
        #   x         5.8e-4 rad/s              2.4e-5 rad/s
        #   y         5.2e-4                    4.5e-4
        #   z         1.4e-3                    5.0e-6      <- already high, and it is the yaw axis
        #
        # Target: angle random walk 0.5 deg/sqrt(h) = 2.057e-3 rad/s discrete at 200 Hz, and a
        # turn-on bias of 0.2 deg/s. Injected = sqrt(target^2 - existing^2) for the noise, and
        # target - existing for the bias. Note the z axis already carries two thirds of the target
        # noise on its own, and that the estimators declare 5e-4 there -- they over-trust it.
        inject_noise = variant in ("noisygyro", "gyronoise")
        inject_bias = variant in ("noisygyro", "gyrobias")
        noise = "[1.973e-3, 1.990e-3, 1.507e-3]" if inject_noise else "[0.0, 0.0, 0.0]"
        bias = "[3.476e-3, -3.450e-3, 3.995e-3]" if inject_bias else "[0.0, 0.0, 0.0]"

        plugins = MC / "mc_rtc.yaml"
        text = plugins.read_text()
        text, n = re.subn(r"(?m)^Plugins:\s*\[([^\]]*)\]",
                          lambda m: f"Plugins: [{m.group(1)}, NoisySensors]", text, count=1)
        assert n == 1, "no active Plugins entry in mc_rtc.yaml"
        plugins.write_text(text)

        noisefile = MC / "plugins/NoisySensors.yaml"
        text = noisefile.read_text()
        if re.search(r"(?m)^seed:", text):
            text = re.sub(r"(?m)^seed:.*$", "seed: 20260914", text)
        else:
            text = "seed: 20260914\n" + text
        for key, value in (("withNoisyGyro", "true"), ("withAcceleroNoise", "false")):
            text, n = re.subn(rf"(?m)^{key}:.*$", f"{key}: {value}", text)
            assert n == 1, f"{key} not found once ({n})"
        for key, value in (("gyroNoise_StdDev", noise), ("gyroOffset", bias)):
            text, n = re.subn(rf"(?m)^{key}:.*$", f"{key}: {value}", text)
            assert n == 1, f"{key} not found once ({n})"
        noisefile.write_text(text)

        # Both estimators are widened by the SAME amount, and only where the signal changed: the
        # measurement variance follows the white noise, the bias init variance follows the bias.
        if inject_noise:
            robot = ROBOTS / "hrp5_p.yaml"
            text = robot.read_text()
            text, n = re.subn(r"(?m)^(\s*gyroSensorVariance:\s*)\[[^\]]*\]",
                              r"\g<1>[4.231e-6,4.231e-6,4.231e-6]", text)
            assert n == 1, f"gyroSensorVariance not found once ({n})"
            robot.write_text(text)

        if inject_bias:
            text = CONFIG.read_text()
            text, n = re.subn(r"(?m)^(\s*gyroBiasInitVariance:\s*)\[[^\]]*\]",
                              r"\g<1>[2.5e-05, 2.5e-05, 2.5e-05]", text)
            assert n == 1, f"gyroBiasInitVariance not found once ({n})"
            CONFIG.write_text(text)

        hartley = MC / "plugins/HartleyIEKF.yaml"
        text = hartley.read_text()
        if inject_noise:
            text, n = re.subn(r"(?m)^(\s*gyroscopeVariance:\s*)2\.5e-7(.*)$", r"\g<1>4.231e-6\g<2>", text)
            assert n == 1, f"hrp5_p gyroscopeVariance not found once ({n})"
        if inject_bias:
            text, n = re.subn(r"(?m)^gyroBiasInitVariance:.*$",
                              "gyroBiasInitVariance: [2.5e-5, 2.5e-5, 2.5e-5]", text)
            assert n == 1, f"RI-EKF gyroBiasInitVariance not found once ({n})"
        hartley.write_text(text)
    elif variant == "pointcontact":
        # The point-contact Kinetics Observer: no moment transmitted at the contacts at all, and
        # the two states pinContacts also drops. It differs from pinContacts on exactly two
        # points -- the yaw viscous damping is removed here (MCKineticsObserver.cpp:192 keeps it,
        # which a point contact cannot exert) and the contact wrench measurements STILL correct
        # the filter. Measured on LongWalk, `noangular` alone scored 0.299 deg against 0.258 for
        # pinContacts although it removes less; this variant says whether the gap comes from that
        # damping term or from the states pinContacts drops.
        main("noangular", isolated)
        text = CONFIG.read_text()
        for key in ("withGyroBias", "withUnmodeledWrench"):
            text, n = re.subn(rf"(?m)^{key}:.*$", f"{key}: false", text)
            assert n == 1, f"{key} not found once ({n})"
        CONFIG.write_text(text)
        return
    elif variant == "noangclean":
        # KO-Lin in the paper. The three changes -- no angular stiffness, no angular damping, no
        # contact torque measurement, no contact torque process -- are now ONE observer option,
        # applied per instance in MCKineticsObserver.cpp right before setObserverCovariances().
        # Writing it into the global observer configuration rather than rewriting the robot files
        # keeps the variant a single deliberate change, the way `pc` and the flexibility variants
        # are, and removes the two regex rewrites that had to be kept in step with the C++.
        CONFIG.write_text("noAngularFlexibility: true\n\n" + CONFIG.read_text())
    elif variant == "noangular":
        # Arnaud's variant, and the cleaner test of "does the contact ORIENTATION carry the yaw":
        # `noangstiff` mirrors pinContacts and therefore keeps the yaw angular damping, which
        # still couples the contact orientation to the base through the relative angular velocity.
        # Here the whole angular channel goes: no angular stiffness, no angular damping at all, so
        # the reaction torque no longer depends on the contact orientation.
        # The contact ORIENTATION covariances are deliberately left alone: an earlier version of
        # this branch also pinned contactOrientationProcessVariance and the contactOriInitVariance
        # keys to zero, which Arnaud never asked for and which made the variant test two things at
        # once. Results produced before 2026-09-16 under the name `noangular` carry that extra
        # change and are not comparable with the ones produced after it.
        for robot in ("hrp5_p", "rhps1"):
            path = ROBOTS / f"{robot}.yaml"
            text = path.read_text()
            for key in ("angStiffness", "angDamping"):
                text = re.sub(rf"(?m)^(\s*{key}:\s*)\[[^\]]*\]", r"\g<1>[0.0, 0.0, 0.0]", text)
            path.write_text(text)
    elif variant in ("noangstiff", "nogyrobias", "nounmodeled"):
        # `pinContacts` changes four things at once, so the KO-PC result cannot say which of them
        # carries the yaw. These three are each ONE of those four, and they need no code change.
        #   noangstiff  -> the angular visco-elastic model, i.e. the contact orientation coupling
        #   nogyrobias  -> the gyrometer bias state (HRP-5P's yaw bias is 0.049 deg/s)
        #   nounmodeled -> the disturbance wrench state
        # What pinContacts does and these do not is disabling the contact wrench CORRECTION; that
        # one is left to elimination.
        if variant == "noangstiff":
            # Mirror MCKineticsObserver.cpp:190 exactly: zero angular stiffness, and the roll and
            # pitch angular damping, keeping the yaw damping. Declared twice per robot file.
            for robot in ("hrp5_p", "rhps1"):
                path = ROBOTS / f"{robot}.yaml"
                text = path.read_text()
                text = re.sub(r"(?m)^(\s*angStiffness:\s*)\[[^\]]*\]",
                              r"\g<1>[0.0, 0.0, 0.0]", text)
                text = re.sub(r"(?m)^(\s*angDamping:\s*)\[\s*([^,\]]*),\s*([^,\]]*),\s*([^,\]]*)\]",
                              r"\g<1>[0.0, 0.0, \g<4>]", text)
                path.write_text(text)
        else:
            key = "withGyroBias" if variant == "nogyrobias" else "withUnmodeledWrench"
            text = CONFIG.read_text()
            text, n = re.subn(rf"(?m)^{key}:.*$", f"{key}: false", text)
            assert n == 1, f"{key} not found once ({n})"
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
    elif variant == "handinput":
        # The left hand stops being a detected contact, but its force sensor is NOT ignored: with
        # no contact claiming it, MCKineticsObserver::inputAdditionalWrench sums its measured
        # wrench into the additional wrench handed to the filter. The hand effort is therefore a
        # KNOWN input rather than a disturbance to estimate -- the opposite of "hidehand".
        # The list is declared twice: in MCKineticsObserver.yaml and again per robot, and the
        # per-robot file wins. Editing only the first one silently changes nothing.
        def drop_hand(path):
            text = path.read_text()
            edited, n = re.subn(r"(?m)^(\s*surfacesForContactDetection:\s*\[)([^\]]*)(\])",
                                lambda m: m.group(1) + ", ".join(
                                    s.strip() for s in m.group(2).split(",")
                                    if "LeftHand" not in s) + m.group(3), text)
            if n:
                path.write_text(edited)
            return n
        touched = drop_hand(CONFIG)
        for robot in ("hrp5_p", "rhps1"):
            touched += drop_hand(ROBOTS / f"{robot}.yaml")
        assert touched, "surfacesForContactDetection found nowhere"
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


def bind(home):
    """Point every path of this module at `home`/.config/mc_rtc."""
    global MC, CONFIG, ROBOTS, EXTRA
    MC = Path(home) / ".config/mc_rtc"
    CONFIG = MC / "observers/MCKineticsObserver.yaml"
    ROBOTS = MC / "observers/MCKineticsObserver"
    EXTRA = {"mc_rtc.yaml": MC / "mc_rtc.yaml", "HartleyIEKF.yaml": MC / "plugins/HartleyIEKF.yaml",
             "NoisySensors.yaml": MC / "plugins/NoisySensors.yaml"}


def install_into(home, variant):
    """Materialize a private HOME from the versioned base and apply `variant` there.

    No snapshot and no restore: the real ~/.config is never written, so nothing can be left
    installed by a crash, and the base is re-read from config_base/ every time.
    """
    import config_home
    config_home.materialize(home)
    bind(home)
    real_main = globals()["main"]
    real_main(variant, isolated=True)
    # Opt-in: re-tick only the estimators asked for. Without KO_OBSERVERS every instance runs, as
    # before, and the materialised controller stays byte-identical to the versioned base.
    selection = os.environ.get("KO_OBSERVERS", "").replace(",", " ").split()
    if selection:
        kept = config_home.select_observers(home, selection)
        print(f"observateurs retenus : {' '.join(kept)}")
    dropped = os.environ.get("KO_DROP_PLUGINS", "").replace(",", " ").split()
    if dropped:
        print(f"plugins restants : {' '.join(config_home.drop_plugins(home, dropped))}")
    (Path(home) / "variant.txt").write_text(variant + "\n")
    print(f"{variant}: {config_home.digest(home)}")


if __name__ == "__main__":
    if sys.argv[1] == "--home":
        install_into(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1])

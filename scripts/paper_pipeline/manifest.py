"""Single source of truth for what the paper is built from.

Every stage reads this: which datasets belong to which result category, which observer variants
have to be run, and which figure each script produces. Adding a dataset or a variant means
editing this file and nothing else.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "results/paper-rebuild"
PAPER = Path("/home/arnaud/Documents/ResearchNotes/Topics/KineticsObserver/Papers/"
             "IJRR/Third_submission/Paper")
EXPORT = PAPER / "figures-export"

# --- datasets -------------------------------------------------------------------------------

MULTICONTACT = [f"HRP5_MultiContact_{i}" for i in (1, 2, 3, 4)]
FLAT = [f"KO_TRO2024_RHPS1_{i}" for i in (1, 2, 3, 4, 5)]
SLIPPING = [f"KO_TRO_2024_RHPS1_SLIPPAGE_{i}" for i in (1, 2, 3)]
LONGWALK = ["HRP5P_LongWalk"]
ALL = MULTICONTACT + FLAT + SLIPPING + LONGWALK

# Left hand removed from the estimation: the effort it measures becomes the disturbance the
# observer has to recover. Separate projects, not a re-tick of the ones above.
NO_LEFT_HAND = [f"HRP5_MultiContact_{i}_WO_LeftHand" for i in (1, 2, 3, 4)]
ORI_ERROR = ["HRP5_MultiContact_ContactInitOriError"]

# Sub-trajectory length the RPE is computed over, per category. Set by the distance each
# scenario actually covers; see the paper's evaluation section.
CATEGORIES = {
    "Multicontact": (MULTICONTACT, 0.3),
    "Flatodometry": (FLAT, 1.0),
    "Slippingodometry": (SLIPPING, 1.0),
    "Longwalk": (LONGWALK, 10.0),
}

# --- observer variants ----------------------------------------------------------------------

# name -> (variant_install argument, datasets, macro estimator name or None)
# The macro name is what metrics.py builds \<Category><Estimator>Relerror... from; None means the
# variant feeds a figure or a table rather than the relative-error macros.
VARIANTS = {
    "clean":      ("clean",      ALL,                    "Kineticsobserver"),
    "zpc":        ("zpc",        ALL,                    "KoZpc"),
    "pc":         ("pc",         ALL,                    "Kowithoutwrenchsensors"),
    "flexdiv10":  ("flexdiv10",  MULTICONTACT + SLIPPING, None),
    "flexmul10":  ("flexmul10",  MULTICONTACT + SLIPPING, None),
    "hidehand":   ("hidehand",   NO_LEFT_HAND,           None),
    "orierror":   ("orierror30", ORI_ERROR,              None),
    # KO-Lin in the paper: the Kinetics Observer with the whole ANGULAR contact channel removed --
    # no angular stiffness, no angular damping, no contact torque measurement, and no process
    # covariance on the contact torque state. The gyro bias, the disturbance wrench and the contact
    # FORCES are kept, so the comparison isolates what the contact orientation brings and nothing
    # else. It is a variant, never a replacement of the KO.
    #
    # It replaces `noangular`, which zeroed the angular visco-elastic law while keeping the torque
    # measurement: the filter was told the contact makes no torque while being shown one, and the
    # corrected torque state still reached the angular dynamics one step later with every Jacobian
    # zeroed. Arnaud called that version erroneous and no longer wants it studied, so it is gone
    # from the paper. The `noangular` branch of variant_install survives only because the
    # `pointcontact` analysis variant builds on it.
    "noangclean": ("noangclean", ALL,                    "Kolinear"),
}

# Analysis variant, not a paper table: the retained tuning with the constrained contact process
# covariance switched off and NOTHING else changed. KO-ZPC conflates that switch with a zero
# contact process covariance, and at zero the switch is inert (covMv = M.0.M' = 0), so the two
# contributions have never been separated. KO -> noconstraint measures the projector M alone;
# noconstraint -> KO-ZPC measures the process covariance alone. Replay only, no figure.
ANALYSIS_REPLAY_LABELS = {"noconstraint": "var-noconstraint"}

# The flexibility ablation writes its own macro category, suffixed, on two categories only.
FLEX_SUFFIX = {"flexdiv10": "b", "flexmul10": "c"}
FLEX_CATEGORIES = ["Multicontact", "Slippingodometry"]

# Variants whose relative errors come from the replay pipeline (kinetics_eval.py), which is what
# results/var-*/ holds. The routine chain is still run for them: it is where the velocity
# pickles, the RI-EKF baseline and every figure's input come from.
REPLAY_LABELS = {"clean": "var-clean-ref", "zpc": "var-zpc", "pc": "var-pc",
                 "flexdiv10": "var-flex-div10", "flexmul10": "var-flex-mul10",
                 # noangclean (KO-Lin) has no replay overlay: its observer option removes the angular
                 # stiffness and damping and neutralises the contact torque channel, none of which is
                 # a covariance the replay can overlay. It is listed here only because metrics.py
                 # gates the relative-error macros on this dict; stage_replay.sh iterates a literal
                 # list, so nothing tries to replay it. Its errors come from the routine, like
                 # every other variant's since the window fix.
                 "noangclean": "var-noangclean"}

# --- controller configuration ------------------------------------------------------------------

# These estimators run together in Passthrough with the retained tuning. Other experiments
# still need separate runs because they change the shared robot or sensor configuration.
# KO-Lin is NOT here: it changes the global observer configuration, which every instance of the
# tick reads, so it cannot share one. The KONOANG instance stays declared in the controller and
# still ticks, but nothing reads its column any more -- and since noAngularFlexibility now also
# neutralises the contact torque channel, that column would be KO-Lin rather than the retired
# `noangular` anyway.
SHARED_ROUTINE_OBSERVERS = {"clean": "KO", "zpc": "KO_ZPC", "pc": "KO_WWS"}
CONTROLLER = (Path(__import__("os").environ.get("KO_CONFIG_HOME", str(Path.home())))
              / ".config/mc_rtc/controllers/Passthrough.yaml")

# --- figures ----------------------------------------------------------------------------------

# produced name -> (script in this package, a glob for the file it writes)
# The trajectory script names its output after the project's group rather than after the figure,
# hence the pattern rather than a fixed name.
FIGURES = {
    # The trajectory figures do not all carry the same curves: KO-ZPC is drawn where the
    # comparison is the point. All KO instances run together regardless of the figure selection.
    "multicontact-odom-traj": ("fig_traj.py HRP5_MultiContact_1 KO,KO_ZPC,Hartley,Control,Mocap",
                               "trajectories_*.pdf"),
    "slipping-odom-traj":     ("fig_traj.py KO_TRO_2024_RHPS1_SLIPPAGE_1 KO,KO_ZPC,Hartley,Control,Mocap",
                               "trajectories_*.pdf"),
    "friends-traj":           ("fig_traj.py KO_TRO2024_RHPS1_1 KO,Hartley,Control,Mocap",
                               "trajectories_*.pdf"),
    "traj_hrp5_long":         ("fig_longwalk.py", "traj_hrp5_long.pdf"),
    "poseAndVel":             ("fig_posevel.py", "poseAndVel.pdf"),
    "extForces":              ("fig_wrench.py", "extForces.pdf"),
    "rightFoot_yaw":          ("fig_yaw.py", "rightFoot_yaw.pdf"),
    # plotContactRestPoses calls exit(0) right after writing its two SVGs, before its own
    # PDF line, so the main panel is what comes out -- mirrored to PDF by also_write_pdf.
    # The published figure is that panel alone; the inset is composed by hand.
    "RightFootRoll":          ("fig_ori.py", "rightFoot_rest_roll_main.pdf"),
}

# Figures the paper ships that no script here produces. Listed so the export stage carries them
# through instead of silently dropping them, and so the gap stays visible.
STATIC_FIGURES = ["compute_time", "framesAndVars", "KineticsObserver", "summary", "viscoFeet",
                  "friends", "tilesOnFloor", "multiContactExpe"]

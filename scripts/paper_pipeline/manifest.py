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
}

# The flexibility ablation writes its own macro category, suffixed, on two categories only.
FLEX_SUFFIX = {"flexdiv10": "b", "flexmul10": "c"}
FLEX_CATEGORIES = ["Multicontact", "Slippingodometry"]

# Variants whose relative errors come from the replay pipeline (kinetics_eval.py), which is what
# results/var-*/ holds. The routine chain is still run for them: it is where the velocity
# pickles, the RI-EKF baseline and every figure's input come from.
REPLAY_LABELS = {"clean": "var-clean-ref", "zpc": "var-zpc", "pc": "var-pc",
                 "flexdiv10": "var-flex-div10", "flexmul10": "var-flex-mul10"}

# --- controller configuration ------------------------------------------------------------------

# The KO-ZPC curve comes from a SECOND MCKineticsObserver instance named KOZPC in the controller's
# observer pipeline, so both estimators come out of a single tick. It is deliberately limited to
# the one short dataset whose figure needs it: two instances re-register their logger keys every
# iteration, which is harmless over 11k iterations and produced a 37.8 GB runaway on LongWalk.
NEEDS_KOZPC = {"HRP5_MultiContact_1"}
CONTROLLER = Path.home() / ".config/mc_rtc/controllers/Passthrough.yaml"

# --- figures ----------------------------------------------------------------------------------

# produced name -> (script in this package, a glob for the file it writes)
# The trajectory script names its output after the project's group rather than after the figure,
# hence the pattern rather than a fixed name.
FIGURES = {
    "multicontact-odom-traj": ("fig_traj.py HRP5_MultiContact_1", "trajectories_*.pdf"),
    "slipping-odom-traj":     ("fig_traj.py KO_TRO_2024_RHPS1_SLIPPAGE_1", "trajectories_*.pdf"),
    "friends-traj":           ("fig_traj.py KO_TRO2024_RHPS1_1", "trajectories_*.pdf"),
    "traj_hrp5_long":         ("fig_longwalk.py", "traj_hrp5_long.pdf"),
    "poseAndVel":             ("fig_posevel.py", "poseAndVel.pdf"),
    "extForces":              ("fig_wrench.py", "extForces.pdf"),
    "rightFoot_yaw":          ("fig_yaw.py", "rightFoot_yaw.pdf"),
    "RightFootRoll":          ("fig_ori.py", "RightFootRoll.pdf"),
}

# Figures the paper ships that no script here produces. Listed so the export stage carries them
# through instead of silently dropping them, and so the gap stays visible.
STATIC_FIGURES = ["compute_time", "framesAndVars", "KineticsObserver", "summary", "viscoFeet",
                  "friends", "tilesOnFloor", "multiContactExpe"]

"""Regenerate each variant's covariance overlay from the retained tuning.

The overlays under results/ were frozen full copies of the tuning they were made against. Change
one covariance in the retained configuration and the variants would keep running the old value --
the tables would then compare, say, a Kinetics Observer at one disturbance-wrench process against
a KO-ZPC at another, and nothing in the output would say so.

Here each variant is expressed as the retained tuning plus an explicit delta, so the two can never
drift apart. Only the delta is variant-specific; everything else is read from the configuration.
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as m

sys.path.insert(0, str(m.ROOT / "scripts"))
from kinetics_tune import MC_RTC_KEYS

RETAINED = m.WORK / "configs/clean/MCKineticsObserver.yaml"

# variant label -> what it changes about the retained tuning.
# `covariances` entries are (field, slice, value) applied after the config is read;
# `settings` and `per_robot` are copied through as-is.
DELTAS = {
    "var-zpc": {
        # Contact rest poses frozen: the position and orientation blocks of contact_process go to
        # zero, the force and torque blocks stay. The adaptive flag is inert once the process is
        # zero (covMv = M.0.M' = 0) and is kept only to match how the variant has always been run.
        "covariances": [("contact_process", slice(0, 6), 0.0)],
        "settings": {"with_adaptative_contact_process_covariance": False},
    },
    "var-noconstraint": {
        # The retained tuning untouched, with only the constrained process covariance switched
        # off: this isolates the projector M, which KO-ZPC cannot because its zero process
        # covariance annihilates both code paths.
        "settings": {"with_adaptative_contact_process_covariance": False},
    },
    "var-flex-div10": {"per_robot_from": "results/var-flex-div10-overlay.yaml"},
    "var-flex-mul10": {"per_robot_from": "results/var-flex-mul10-overlay.yaml"},
}


def retained_covariances():
    """Invert MC_RTC_KEYS: read the mc_rtc YAML back into overlay-shaped vectors."""
    # The observer YAML carries tabs inside some values, which the YAML scanner rejects.
    config = yaml.safe_load(RETAINED.read_text().replace("\t", " "))
    process = config["ekfStateProcessVariances"]
    flat = dict(process)
    for key, value in config.items():
        if not isinstance(value, (dict, list)):
            flat.setdefault(key, value)
    for section in ("ekfStateInitCovariances", "ekfStateProcessVariances"):
        flat.update(config.get(section, {}) or {})

    covariances = {}
    for field, axes, key in MC_RTC_KEYS:
        if key not in flat:
            raise SystemExit(f"{key} absent de {RETAINED}")
        values = [float(v) for v in flat[key]]
        vector = covariances.setdefault(field, [0.0] * (axes.stop if axes.stop > 3 else 3))
        while len(vector) < axes.stop:
            vector.append(0.0)
        vector[axes] = values
    return covariances


def build(label, delta, covariances):
    overlay = {"covariances": {k: list(v) for k, v in sorted(covariances.items())}}
    for field, axes, value in delta.get("covariances", []):
        span = range(*axes.indices(len(overlay["covariances"][field])))
        for index in span:
            overlay["covariances"][field][index] = value
    if "settings" in delta:
        overlay["settings"] = delta["settings"]
    source = delta.get("per_robot_from")
    if source:
        previous = yaml.safe_load((m.ROOT / source).read_text())
        if "per_robot" in previous:
            overlay["per_robot"] = previous["per_robot"]
        else:
            raise SystemExit(f"{source} n'a pas de bloc per_robot a reprendre")
    return overlay


# The pinContacts variant cannot be expressed as a covariance overlay -- it changes the state
# dimension, so it runs from a standalone observer configuration. That file used to be a frozen
# copy, which is how it kept the previous disturbance-wrench process while every overlay followed
# the new one. Derived here for the same reason, and by the same rule.
STANDALONE = {"pc": "pinContacts: true\n\n"}


def write_standalone():
    for name, prelude in STANDALONE.items():
        target = m.WORK / f"configs/{name}/MCKineticsObserver.yaml"
        if not target.exists():
            print(f"  {name}: {target} absent, ignore")
            continue
        backup = m.WORK / "overlays-previous" / f"{name}-MCKineticsObserver.yaml"
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_text(target.read_text())
        target.write_text(prelude + RETAINED.read_text())
        print(f"ecrit configs/{name}/MCKineticsObserver.yaml (config retenue + {prelude.strip()})")


def main():
    covariances = retained_covariances()
    print("=== covariances lues dans la config retenue ===")
    for field, vector in sorted(covariances.items()):
        print(f"  {field:30s} {vector}")
    for label, delta in DELTAS.items():
        target = m.ROOT / f"results/{label}-overlay.yaml"
        overlay = build(label, delta, covariances)
        # These overlays are not tracked by git, so a regeneration is the only copy there is.
        if target.exists():
            backup = m.WORK / "overlays-previous" / target.name
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_text(target.read_text())
        target.write_text(yaml.safe_dump(overlay, sort_keys=True))
        changed = [f"{f}{a}" for f, a, _ in delta.get("covariances", [])]
        print(f"ecrit {target.name}" + (f"  (delta: {', '.join(changed)})" if changed else ""))
    write_standalone()


if __name__ == "__main__":
    main()

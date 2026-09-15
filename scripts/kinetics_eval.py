#!/usr/bin/env python3

import argparse
import csv
import hashlib
import math
import os
import importlib.util
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import webbrowser
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = Path("/home/arnaud/devel/src/catkin_ws")
DEFAULT_MC_RTC_CONFIG = Path("/home/arnaud/.config/mc_rtc/mc_rtc.yaml")
DEFAULT_MCKINETICS_CONFIG = Path("/home/arnaud/.config/mc_rtc/observers/MCKineticsObserver.yaml")
DEFAULT_PASSTHROUGH_CONFIG = Path("/home/arnaud/.config/mc_rtc/controllers/Passthrough.yaml")
OBSERVER_PREFIX = "Observers_MainObserverPipeline_MCKineticsObserver"
# Channel families the converter needs but that lightenOutputBin.py strips from logReplay.bin.
# A prefix ending in "_" matches any entry below it; the others must be present verbatim.
REQUIRED_LOG_ENTRIES = (f"{OBSERVER_PREFIX}_constants_mass",
                        f"{OBSERVER_PREFIX}_MEKF_inputs_",
                        f"{OBSERVER_PREFIX}_MEKF_measurements_",
                        f"{OBSERVER_PREFIX}_debug_contactKine_")
PROJECTS = [*(f"HRP5_MultiContact_{index}" for index in range(1, 5)),
            "HRP5P_LongWalk",
            *(f"KO_TRO2024_RHPS1_{index}" for index in range(1, 6)),
            *(f"KO_TRO_2024_RHPS1_SLIPPAGE_{index}" for index in range(1, 4))]
CACHE_SUFFIX = ""


def merge_covariance_overlay(configuration, overlay, robot=None):
    """Apply absolute `covariances` and/or multiplicative `covariance_scales` to a replay config.

    Scales exist for covariances that are legitimately per-robot, such as contact_wrench, which
    carries each robot's force/torque sensor calibration. Searching those as absolute values would
    force one robot onto the other's calibration; a shared multiplier keeps the ratio intact.
    """
    # overlay key -> (config section it edits, whether its numbers are multipliers)
    sections = {"covariances": ("covariances", False), "covariance_scales": ("covariances", True),
                "contact_model": ("contact_model", False), "contact_model_scales": ("contact_model", True),
                "contact_model_per_contact": ("contact_model", None),
                "covariances_per_contact": ("covariances", None)}
    settings = overlay.get("settings") if isinstance(overlay, dict) else None
    # `per_robot` selects the layer that applies to this config's robot and folds it in before
    # anything else. The visco-elastic contact model is genuinely per-robot -- RHPS1's feet are
    # an order of magnitude softer than HRP5P's -- so a single shared overlay cannot express a
    # search over one robot's flexibility without dragging the other's along with it.
    per_robot = overlay.get("per_robot") if isinstance(overlay, dict) else None
    if per_robot is not None:
        if not isinstance(per_robot, dict):
            raise ValueError("per_robot must be a mapping of robot name to overlay")
        if robot is None:
            raise ValueError("per_robot needs the robot this configuration belongs to")
        overlay = {key: value for key, value in overlay.items() if key != "per_robot"}
        mine = per_robot.get(robot)
        if mine is not None:
            if not isinstance(mine, dict):
                raise ValueError(f"per_robot.{robot} must be a mapping")
            clashing = sorted(set(mine) & set(overlay))
            if clashing:
                raise ValueError(f"per_robot.{robot} repeats shared keys: {', '.join(clashing)}")
            overlay = {**overlay, **mine}
    overlay = {key: value for key, value in overlay.items() if key != "settings"} if settings is not None else overlay
    if not isinstance(overlay, dict) or not (set(overlay) or settings) or not set(overlay) <= set(sections):
        raise ValueError(f"overlay must contain only {', '.join(sorted(sections))}, per_robot and/or settings")
    output = {**configuration}
    if settings is not None:
        # Top-level switches such as with_adaptative_contact_process_covariance; scalars only, and
        # only ones the replay config already defines, so a typo cannot silently do nothing.
        if not isinstance(settings, dict):
            raise ValueError("settings must be a mapping")
        unknown = sorted(set(settings) - set(configuration))
        if unknown:
            raise ValueError(f"unknown settings: {', '.join(unknown)}")
        for name, value in settings.items():
            if isinstance(configuration[name], (dict, list)) or isinstance(value, (dict, list)):
                raise ValueError(f"settings.{name} must be a scalar")
            output[name] = value
    for section in {sections[key][0] for key in overlay}:
        if not isinstance(configuration.get(section), dict):
            raise ValueError(f"{section} must be a mapping")
        output[section] = {**configuration[section]}
    for section in {sections[key][0] for key in overlay}:
        keys = [key for key in overlay if sections[key][0] == section]
        overlapping = sorted(set.intersection(*(set(overlay[key]) for key in keys))) if len(keys) > 1 else []
        if overlapping:
            raise ValueError(f"{section} fields set both absolutely and by scale: {', '.join(overlapping)}")
    for key in sorted(set(overlay)):
        section, is_scale = sections[key]
        reference_section = configuration[section]
        values = overlay[key]
        if is_scale is None:
            # Per-contact visco-elastic overrides: {contact name: {field: [3 values]}}. Validated
            # against the robot-wide model so a typo cannot silently do nothing.
            if not isinstance(values, dict):
                raise ValueError(f"{key} must be a mapping of contact name to fields")
            # Every field must match a vector field the robot-wide model already defines.
            for contact, fields in values.items():
                if not isinstance(fields, dict):
                    raise ValueError(f"{key}.{contact} must be a mapping")
                for name, candidate in fields.items():
                    if name not in reference_section or not isinstance(reference_section[name], list):
                        raise ValueError(f"{key}.{contact}: unknown {section} field {name}")
                    if not isinstance(candidate, list) or len(candidate) != len(reference_section[name]):
                        raise ValueError(f"{key}.{contact}.{name} must contain "
                                         f"{len(reference_section[name])} values")
            output[section]["per_contact"] = {
                contact: {name: [float(v) for v in numbers] for name, numbers in fields.items()}
                for contact, fields in values.items()}
            continue
        if not isinstance(values, dict):
            raise ValueError(f"{key} must be a mapping")
        numeric = {name for name, value in reference_section.items() if isinstance(value, list)}
        unknown = sorted(set(values) - numeric)
        if unknown:
            raise ValueError(f"unknown {section} fields: {', '.join(unknown)}")
        for name, candidate in values.items():
            reference = reference_section[name]
            if not isinstance(candidate, list) or len(candidate) != len(reference):
                raise ValueError(f"{key}.{name} must contain {len(reference)} values")
            candidate = [float(value) for value in candidate]
            if any(not math.isfinite(value) or value < 0.0 for value in candidate):
                raise ValueError(f"{key}.{name} values must be finite and non-negative")
            if is_scale:
                candidate = [scale * float(value) for scale, value in zip(candidate, reference)]
            output[section][name] = candidate
    return output


def deep_merge(base, overlay):
    """mc_rtc Configuration::load semantics: nested mappings merge key by key, everything else replaces."""
    output = dict(base)
    for key, value in overlay.items():
        if isinstance(output.get(key), dict) and isinstance(value, dict):
            output[key] = deep_merge(output[key], value)
        else:
            output[key] = value
    return output


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8").replace("\t", " ")) or {}


def pipeline_observer_config(path):
    """The inline `config:` block the controller pins on MCKineticsObserver.

    It is the highest-precedence layer in mc_rtc and the converter knows nothing about it,
    so the replay would otherwise run with a different odometry type and contact thresholds
    than the ticker did.
    """
    pipelines = read_yaml(path).get("ObserverPipelines")
    if pipelines is None:
        raise RuntimeError(f"{path}: no ObserverPipelines")
    for pipeline in (pipelines if isinstance(pipelines, list) else [pipelines]):
        for observer in pipeline.get("observers", []):
            if observer.get("type") != "MCKineticsObserver":
                continue
            if observer.get("name", "MCKineticsObserver") != "MCKineticsObserver":
                continue
            return observer.get("config") or {}
    raise RuntimeError(f"{path}: the pipeline has no unnamed MCKineticsObserver")


def resolved_observer_tree(robot, destination, passthrough):
    """Mirror the mc_rtc observer configuration, inline controller layer included.

    The converter reads `<file>.yaml` and `<file>/<robot>.yaml` and merges them itself, so the
    inline layer is folded into *both* files to reproduce mc_rtc's precedence exactly:
    global < robot overlay < controller `config:` block.
    """
    inline = pipeline_observer_config(passthrough)
    source = DEFAULT_MCKINETICS_CONFIG.expanduser()
    overlay_source = source.parent / source.stem / f"{robot}.yaml"
    if not overlay_source.exists():
        raise FileNotFoundError(overlay_source)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / source.stem).mkdir(exist_ok=True)
    target = destination / source.name
    target.write_text(yaml.safe_dump(deep_merge(read_yaml(source), inline), sort_keys=False), encoding="utf-8")
    (destination / source.stem / f"{robot}.yaml").write_text(
        yaml.safe_dump(deep_merge(read_yaml(overlay_source), inline), sort_keys=False), encoding="utf-8")
    return target


def log_entries(path):
    output = subprocess.check_output(["mc_bin_utils", "show", str(path)], text=True)
    return {line[2:].split(" (", 1)[0] for line in output.splitlines() if line.startswith("- ")}


def missing_log_entries(path):
    entries = log_entries(path)
    return [required for required in REQUIRED_LOG_ENTRIES
            if not (any(entry.startswith(required) for entry in entries) if required.endswith("_")
                    else required in entries)]


def ros_environment(workspace):
    setup = workspace / "install/setup.zsh"
    if not setup.exists():
        raise FileNotFoundError(f"ROS workspace is not built: {setup}")
    command = f"source {shlex.quote(str(setup))} && env -0"
    output = subprocess.check_output(["zsh", "-lc", command])
    return dict(item.split("=", 1) for item in output.decode().split("\0") if item)


def run(command, env=None, cwd=ROOT):
    print("+", shlex.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)


def selected_projects(names):
    if not names:
        return list(PROJECTS)
    chosen = [name.strip() for name in names.split(",") if name.strip()]
    unknown = [name for name in chosen
               if name not in PROJECTS and not (ROOT / "Projects" / name).is_dir()]
    if unknown:
        raise ValueError(f"unknown project(s): {', '.join(unknown)}; known: {', '.join(PROJECTS)}")
    return chosen


def project_paths(name):
    project = ROOT / "Projects" / name
    return project, project / "output_data" / ("kinetics_eval" + CACHE_SUFFIX)


def fingerprint(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def shared_library(binary, stem):
    """Resolve a shared library the given binary links against, via ldd.

    The estimator library is what actually decides the observer's numerics, yet it is neither the
    replay binary nor the mc_rtc plugin. Rebuilding it while leaving a previously ticked log in
    place silently compares two different observers, so it has to be fingerprinted too.
    """
    try:
        output = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, OSError):
        return None
    for line in output.splitlines():
        if stem not in line or "=>" not in line:
            continue
        candidate = Path(line.split("=>", 1)[1].split("(", 1)[0].strip())
        if candidate.exists():
            return candidate.resolve()
    return None


def replay_dependencies(workspace, env):
    converter = workspace / "src/state_observation_ros2/test_state_obs_ros2/scripts/bin_to_rosbag_kinetics.py"
    launch = workspace / "src/state_observation_ros2/test_state_obs_ros2/launch/test_kinetics_replay.launch.py"
    wrapper = workspace / "install/lib/test_state_obs_ros2/rosbag_publish_kinetics"
    observer = workspace / "install/lib/kinetics_observer_ros2/kinetics_offline_replay"
    if not observer.exists():
        observer = Path(shutil.which("kinetics_offline_replay", path=env.get("PATH")) or "")
    if not converter.exists() or not launch.exists() or not wrapper.exists() or not observer.exists():
        raise FileNotFoundError("current test_state_obs_ros2/kinetics_observer_ros2 build is unavailable")
    observer_config = DEFAULT_MCKINETICS_CONFIG.expanduser()
    overlays = [observer_config.parent / "MCKineticsObserver" / name for name in ("hrp5_p.yaml", "rhps1.yaml")]
    if not observer_config.exists() or any(not path.exists() for path in overlays):
        raise FileNotFoundError("mc_rtc MCKineticsObserver configuration is unavailable")
    passthrough = DEFAULT_PASSTHROUGH_CONFIG.expanduser()
    if not passthrough.exists():
        raise FileNotFoundError(f"mc_rtc controller configuration is unavailable: {passthrough}")
    dependencies = {"converter": fingerprint(converter), "launch": fingerprint(launch),
                    "wrapper": fingerprint(wrapper), "observer": fingerprint(observer),
                    "observer_config": fingerprint(observer_config),
                    "observer_hrp5_p": fingerprint(overlays[0]), "observer_rhps1": fingerprint(overlays[1]),
                    "controller_config": fingerprint(passthrough)}
    # The estimator library and the mc_rtc plugin decide the observer's numerics: the first is what
    # the standalone replay runs, the second is what ticked the log. A mismatch between them shows up
    # as a small unexplained trajectory difference rather than a failure, so pin both.
    estimator = shared_library(observer, "libstate-observation")
    if estimator is not None:
        dependencies["estimator_library"] = fingerprint(estimator)
    plugin = Path("/home/arnaud/devel/install/lib/mc_observers/MCKineticsObserver.so")
    if plugin.exists():
        dependencies["mc_rtc_plugin"] = fingerprint(plugin)
    return dependencies


def validate_cache(cache, dependencies):
    manifest_path = cache / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"stale cache without manifest: {cache}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("source", "reference", *dependencies):
        expected = manifest.get(key)
        if expected is None:
            raise RuntimeError(f"stale cache for {cache.parent.parent.name}; rerun prepare --force")
        current = fingerprint(Path(expected["path"]))
        if current != expected:
            raise RuntimeError(f"{key} changed for {cache.parent.parent.name}; rerun prepare --force")
    for key, expected in dependencies.items():
        if manifest[key] != expected:
            raise RuntimeError(f"{key} changed for {cache.parent.parent.name}; rerun prepare --force")


def project_robot(project):
    data = yaml.safe_load((project / "projectConfig.yaml").read_text(encoding="utf-8")) or {}
    robot = str(data.get("EnabledRobot", "")).strip()
    if not robot:
        raise RuntimeError(f"{project.name}: projectConfig.yaml has no EnabledRobot")
    return robot


def observer_robot(robot):
    if robot == "HRP5P":
        return "hrp5_p"
    if robot.startswith("RHPS1"):
        return "rhps1"
    raise RuntimeError(f"unsupported replay robot {robot!r}; add its observer overlay name")


def project_timestep(project):
    """The controller period the log was recorded at, read from the log itself.

    mc_rtc_ticker aborts when the controller period disagrees with the replay's, and the datasets
    do not share one: the multi-contact and RHPS1 logs run at 5 ms, HRP5P_LongWalk at 2 ms.
    perf_GlobalRun_log.csv is the cheapest faithful source, being a direct export of the raw log.
    """
    source = project / "output_data/perf_GlobalRun_log.csv"
    if not source.exists():
        raise FileNotFoundError(f"{source}; needed to learn the replay timestep")
    stamps = []
    with source.open(encoding="utf-8") as stream:
        header = stream.readline().rstrip("\n").split(";")
        if "t" not in header:
            raise RuntimeError(f"{source}: no `t` column")
        column = header.index("t")
        for line in stream:
            fields = line.rstrip("\n").split(";")
            if len(fields) > column and fields[column].strip():
                stamps.append(float(fields[column]))
            if len(stamps) >= 64:
                break
    steps = sorted(round(b - a, 9) for a, b in zip(stamps, stamps[1:]) if b > a)
    if not steps:
        raise RuntimeError(f"{source}: could not derive a timestep from the `t` column")
    return steps[len(steps) // 2]


def replay_config(source, robot, timestep, log_directory=None, log_template=None):
    text = source.read_text(encoding="utf-8")
    if not re.search(r"(?m)^MainRobot:\s*", text):
        raise RuntimeError(f"{source}: no active MainRobot entry")
    text = re.sub(r"(?m)^MainRobot:\s*.*$", f"MainRobot: {robot}", text, count=1)
    text = re.sub(r"(?m)^Plugins:\s*\[.*$", "Plugins: [HartleyIEKF]", text, count=1)
    rendered = f"Timestep: {timestep:.9g}"
    if re.search(r"(?m)^Timestep:\s*", text):
        text = re.sub(r"(?m)^Timestep:\s*.*$", rendered, text, count=1)
    else:
        text += f"\n{rendered}\n"
    if log_directory is not None:
        # Pinning the log destination makes the pickup deterministic, so several projects
        # can be re-ticked at once without racing over /tmp/mc-control-*.bin.
        text = re.sub(r"(?m)^LogDirectory:\s*.*$", "", text)
        text = re.sub(r"(?m)^LogTemplate:\s*.*$", "", text)
        text += f"\nLogDirectory: {log_directory}\nLogTemplate: {log_template}\n"
    handle = tempfile.NamedTemporaryFile(prefix="mc_rtc_replay_", suffix=".yaml", mode="w", delete=False)
    with handle:
        handle.write(text)
    return Path(handle.name)


def write_trajectory(rows, prefix, output):
    columns = ["t", *(f"{prefix}_position_{axis}" for axis in "xyz"),
               *(f"{prefix}_orientation_{axis}" for axis in "xyzw")]
    with output.open("w", encoding="utf-8") as stream:
        stream.write("# timestamp tx ty tz qx qy qz qw\n")
        for row in rows:
            if any(not row.get(column, "").strip() for column in columns):
                continue
            stream.write(" ".join(row[column] for column in columns) + "\n")


VELOCITY_FILTER = (2, 15.0 / (0.5 * 200.0))   # plotAndFormatResults.py:933


def columns_of(rows, names):
    import numpy as np
    return np.array([[float(row[name]) for name in names] for row in rows])


def differentiate(position, step):
    """Filtered finite difference, as the pipeline computes mocap and estimator velocities."""
    import numpy as np
    from scipy.signal import butter, filtfilt
    smoothed = filtfilt(*butter(N=VELOCITY_FILTER[0], Wn=VELOCITY_FILTER[1], btype="low"),
                        position, axis=0)
    return np.vstack((np.zeros((1, 3)), np.diff(smoothed, axis=0) / step))


def imu_offset(rows):
    """posFbImu and rImuFb, following plotAndFormatResults.py:143-150."""
    from scipy.spatial.transform import Rotation
    position = columns_of(rows[:1], [f"HartleyIEKF_imuFbKine_position_{axis}" for axis in "xyz"])[0]
    rotation = Rotation.from_quat(columns_of(rows[:1], [f"HartleyIEKF_imuFbKine_ori_{axis}"
                                                        for axis in "xyzw"])[0])
    return -rotation.apply(position, inverse=True), rotation


def reference_velocities(project, cache, frame):
    """Mocap and RI-EKF local linear velocity in the frame projectConfig asks to evaluate.

    Both are written once at prepare time so every tuning trial only has to transform the
    Kinetics estimate. The mocap has no velocity output, so it is differentiated the way the
    pipeline does; RI-EKF's is its own estimated velocity, transported to the floating base
    when that is the evaluation frame.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    with (project / "output_data/finalDataCSV.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter=";"))
    times = columns_of(rows, ["t"])[:, 0]
    step = float(np.median(np.diff(times)))
    pos_fb_imu, r_imu_fb = imu_offset(rows)

    mocap_position = columns_of(rows, [f"Mocap_position_{axis}" for axis in "xyz"])
    mocap_rotation = Rotation.from_quat(columns_of(rows, [f"Mocap_orientation_{axis}" for axis in "xyzw"]))
    if frame == "IMU":
        imu_position = mocap_position + mocap_rotation.apply(pos_fb_imu)
        imu_rotation = mocap_rotation * r_imu_fb.inv()
        mocap_local = imu_rotation.apply(differentiate(imu_position, step), inverse=True)
    else:
        mocap_local = mocap_rotation.apply(differentiate(mocap_position, step), inverse=True)

    riekf_world = columns_of(rows, [f"Hartley_IMU_linVel_{axis}" for axis in "xyz"])
    riekf_imu_rotation = Rotation.from_quat(columns_of(rows, [f"Hartley_IMU_orientation_{axis}"
                                                              for axis in "xyzw"]))
    riekf_local_imu = riekf_imu_rotation.apply(riekf_world, inverse=True)
    if frame == "IMU":
        riekf_local = riekf_local_imu
    else:
        # Transport RI-EKF's IMU velocity back to the floating base with its own angular rate.
        gyro = columns_of(rows, [f"Accelerometer_angularVelocity_{axis}" for axis in "xyz"])
        bias = columns_of(rows, [f"Hartley_IMU_gyroBias_{axis}" for axis in "xyz"])
        angular_fb = r_imu_fb.apply(gyro - bias, inverse=True)
        riekf_local = r_imu_fb.apply(riekf_local_imu, inverse=True) - np.cross(angular_fb, pos_fb_imu)

    for name, values in (("mocap_velocity.txt", mocap_local), ("riekf_velocity.txt", riekf_local)):
        with (cache / "reference" / name).open("w", encoding="utf-8") as stream:
            stream.write("# timestamp vx vy vz\n")
            for moment, value in zip(times, values):
                stream.write(" ".join(f"{item:.17g}" for item in (moment, *value)) + "\n")
    return {"frame": frame, "pos_fb_imu": list(map(float, pos_fb_imu)),
            "r_imu_fb": list(map(float, r_imu_fb.as_quat()))}


def sampling_step(times):
    """The median period of a timestamp column, robust to a dropped or repeated sample."""
    import numpy as np
    steps = np.diff(times)
    steps = steps[steps > 0]
    if not len(steps):
        raise RuntimeError("cannot derive a sampling step from a non-increasing time column")
    return float(np.median(steps))


def resample_uniform(times, values, step):
    """`values` on a uniform grid of the given step, so two series become index-comparable."""
    import numpy as np
    grid = np.arange(times[0], times[-1] + 0.5 * step, step)
    resampled = np.column_stack([np.interp(grid, times, values[:, axis])
                                 for axis in range(values.shape[1])])
    return grid, resampled


def accelerometer_signal(rows):
    columns = [f"Accelerometer_linearAcceleration_{axis}" for axis in "xyz"]
    import numpy as np
    return np.array([[float(row[column]) for column in columns] for row in rows])


# Bounds for judging the accelerometer alignment. A shift is accepted when its residual is
# clearly the lowest in the sweep; the guard band keeps the neighbouring shifts of the same
# minimum from being mistaken for independent rivals.
GUARD_SECONDS = 0.05
PROBE_COUNT = 40
# 0.9, not tighter: on KO_TRO2024_RHPS1_5 the correct shift -- 207.370 s, the start this dataset
# is independently known to have -- beats the best rival by only 16%, because the probes are
# sparse over a 41k-shift sweep and the reference has been resampled. A gross misalignment scores
# no better than its rivals at all, so it lands near 1.0 and is still caught.
MATCH_MARGIN = 0.9


def reference_time_offset(project, cache):
    """Seconds to subtract from log timestamps to land on the finalDataCSV time base.

    The routine re-zeroes time on the mocap window, which for some datasets starts well
    inside the log (KO_TRO2024_RHPS1_5 begins at 207 s). Replaying with raw log timestamps
    then compares the estimate against an unrelated stretch of ground truth. The offset is
    recovered by cross-correlating the accelerometer, which both files carry verbatim.
    """
    import numpy as np
    from scipy.signal import correlate
    log_csv = project / "output_data/logReplay.csv"
    final_csv = project / "output_data/finalDataCSV.csv"
    if not log_csv.exists():
        print(f"{project.name}: no logReplay.csv, assuming the log and the reference share a time base")
        return 0.0
    with log_csv.open(newline="", encoding="utf-8") as stream:
        log_rows = list(csv.DictReader(stream, delimiter=";"))
    with final_csv.open(newline="", encoding="utf-8") as stream:
        reference_rows = list(csv.DictReader(stream, delimiter=";"))
    log_time = np.array([float(row["t"]) for row in log_rows])
    reference_time = np.array([float(row["t"]) for row in reference_rows])
    log = accelerometer_signal(log_rows)
    reference = accelerometer_signal(reference_rows)
    # The two files need not share a sampling rate: the routine downsamples some datasets on the
    # way to finalDataCSV (HRP5P_LongWalk is logged at 2 ms and referenced at 4 ms). Correlating
    # by sample index would then compare series whose steps mean different amounts of time, so
    # both are put on the same uniform grid, at the coarser of the two steps, before matching.
    step = max(sampling_step(log_time), sampling_step(reference_time))
    log_grid, log = resample_uniform(log_time, log, step)
    _, reference = resample_uniform(reference_time, reference, step)
    # A reference that runs a little past the log is expected. repair_mc_rtc_skipped_iters.py adds
    # back the controller iterations the real-time loop skipped -- detected through perf_GlobalRun,
    # not through `t`, which is a uniform counter -- and finalDataCSV inherits them while the raw
    # .bin this replay reads does not. The overhang is exactly the skip count: 6 iterations on
    # KO_TRO2024_RHPS1_1, 10 on KO_TRO2024_RHPS1_5, at most 12 anywhere. That is under 0.013% of a
    # run, so it shifts the clock by ~0.015 samples inside an RPE window and is not worth
    # correcting; an overhang far larger than that means the two files cover different recordings.
    overhang = len(reference) - len(log)
    if overhang > 0:
        tolerance = max(int(0.01 * len(log)), int(round(2.0 / step)))
        if overhang > tolerance:
            raise RuntimeError(f"{project.name}: the reference runs {overhang * step:.1f} s past "
                               f"the log; they do not cover the same recording")
        reference = reference[:len(log)]
    log = log - log.mean(0)
    reference = reference - reference.mean(0)
    scores = sum(correlate(log[:, axis], reference[:, axis], mode="valid") for axis in range(3))
    shift = int(np.argmax(scores))

    def residual_at(candidate):
        return float(np.abs(log[candidate:candidate + len(reference)] - reference).mean())

    # Whether the offset is right is a question about the shape of the match, not its depth.
    # The two accelerometer series are not copies of one another: the routine resamples the
    # reference on the way to finalDataCSV (HRP5P_LongWalk is logged at 2 ms and referenced at
    # 4 ms) and adds back the skipped controller iterations. Decimating a noisy 200 Hz signal
    # changes its sample values a lot while leaving its timing untouched, so a residual that
    # looks large next to the signal amplitude says nothing at all about the alignment -- the
    # earlier "residual > 5% of signal" test fired on every resampled dataset for that reason.
    # What does carry information is whether moving away from the chosen shift makes the match
    # worse: a correctly located minimum is clearly deeper than anywhere else in the sweep.
    residual = residual_at(shift)
    if len(scores) == 1:
        # The reference spans the whole log, so shift 0 is the only alignment there is and
        # there is no offset to get wrong. Nothing to check.
        pass
    else:
        guard = max(1, int(round(GUARD_SECONDS / step)))
        pool = np.concatenate((np.arange(0, max(0, shift - guard)),
                               np.arange(min(len(scores), shift + guard + 1), len(scores))))
        if pool.size:
            probes = pool[np.linspace(0, pool.size - 1, min(PROBE_COUNT, pool.size)).astype(int)]
            rival = min(residual_at(int(candidate)) for candidate in probes)
            if residual > MATCH_MARGIN * rival:
                print(f"{project.name}: WARNING accelerometer match is poor (residual "
                      f"{residual:.4g} at the chosen shift vs {rival:.4g} {guard * step:.3g} s "
                      f"away); the time offset may be wrong")
    offset = float(log_grid[shift] - reference_time[0])
    if shift:
        print(f"{project.name}: reference starts {offset:.3f} s into the log "
              f"(sample {shift} of a {step * 1e3:.4g} ms grid)")
    return offset


def references(project, cache):
    reference = cache / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    for prefix, filename in (("Mocap", "mocap.txt"), ("Hartley", "riekf.txt")):
        with (project / "output_data/finalDataCSV.csv").open(newline="", encoding="utf-8") as stream:
            write_trajectory(csv.DictReader(stream, delimiter=";"), prefix, reference / filename)


def newest_replay(started):
    candidates = [path for path in Path("/tmp").rglob("mc-control*Passthrough*.bin")
                  if "latest" not in path.name and path.stat().st_mtime >= started - 1]
    if not candidates:
        raise RuntimeError("mc_rtc_ticker produced no Passthrough log under /tmp")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def regenerate_full_log(project, cache, mc_rtc_config, robot, env):
    """Re-tick raw_data/controllerLog.bin to obtain the debug log the converter needs.

    mainRoutine.sh lightens output_data/logReplay.bin in place, which drops every
    MEKF input/measurement channel, so the routine's log can no longer feed the replay.
    Ticking here reuses the very same controller configuration, so the observer that
    writes this log is the one the routine runs.
    """
    controller_log = project / "raw_data/controllerLog.bin"
    if not controller_log.exists():
        raise FileNotFoundError(controller_log)
    destination = cache / "logReplay_full.bin"
    cache.mkdir(parents=True, exist_ok=True)
    tick_directory = cache / "tick"
    shutil.rmtree(tick_directory, ignore_errors=True)
    tick_directory.mkdir()
    template = "replay"
    started = time.time()
    timestep = project_timestep(project)
    replay_file = replay_config(mc_rtc_config, robot, timestep, tick_directory, template)
    try:
        print(f"{project.name}: re-ticking {controller_log.name} to rebuild the full observer log", flush=True)
        run(["mc_rtc_ticker", "-f", replay_file, "--no-sync", "--replay-outputs", "-e", "-l", controller_log], env)
        produced = [path for path in tick_directory.glob(f"{template}*.bin") if "latest" not in path.name]
        if not produced:
            produced = [newest_replay(started)]
        shutil.move(str(max(produced, key=lambda path: path.stat().st_mtime)), destination)
    finally:
        replay_file.unlink(missing_ok=True)
        shutil.rmtree(tick_directory, ignore_errors=True)
    missing = missing_log_entries(destination)
    if missing:
        raise RuntimeError(f"{project.name}: the re-ticked log still lacks {', '.join(missing)}; "
                           f"check that {DEFAULT_PASSTHROUGH_CONFIG} keeps withDebugLogs: true "
                           "on MCKineticsObserver")
    return destination


SAFE_CONVERSION_BYTES = 8 * 1024 ** 3
CONVERSION_CHUNK_BYTES = 2 * 1024 ** 3


def compact_conversion_source(source, directory):
    """Keep the converter below its RAM cliff for very large mc_rtc logs."""
    output = subprocess.check_output(["mc_bin_utils", "show", str(source)], text=True)
    available = {
        line[2:].split(" (", 1)[0]
        for line in output.splitlines()
        if line.startswith("- ")
    }
    exact = {
        "t",
        f"{OBSERVER_PREFIX}_constants_mass",
        f"{OBSERVER_PREFIX}_MEKF_initialState",
        *(f"{OBSERVER_PREFIX}_MEKF_estimatedState_{name}"
          for name in ("position", "ori", "linVel", "angVel", "extForceCentr", "extTorqueCentr")),
    }
    prefixes = (
        # The estimated gyro bias initialises the replay, and its channels are named after the IMU
        # ("..._gyroBias_Accelerometer_x"), so they cannot be listed exactly. Compaction only runs
        # for logs above SAFE_CONVERSION_BYTES, which is why this went unnoticed until a 24 GB
        # LongWalk log took that path and the replay refused it for missing initialisation state.
        f"{OBSERVER_PREFIX}_MEKF_estimatedState_gyroBias_",
        f"{OBSERVER_PREFIX}_MEKF_inputs_",
        f"{OBSERVER_PREFIX}_MEKF_measurements_",
        f"{OBSERVER_PREFIX}_debug_contactKine_",
        f"{OBSERVER_PREFIX}_debug_contactState_isSet_",
    )
    keys = sorted(key for key in available if key in exact or key.startswith(prefixes))
    if not keys:
        raise RuntimeError(f"{source}: no KineticsObserver channels found")

    parts = max(2, math.ceil(source.stat().st_size / CONVERSION_CHUNK_BYTES))
    split_prefix = directory / "part"
    run(["mc_bin_utils", "split", "--in", source, "--out", split_prefix, "--parts", parts])
    chunks = sorted(directory.glob("part_*.bin"))
    if len(chunks) != parts:
        raise RuntimeError(f"{source}: expected {parts} split logs, got {len(chunks)}")
    compact = []
    for index, chunk in enumerate(chunks):
        target = directory / f"compact_{index:02d}.bin"
        run(["mc_bin_utils", "extract", "--in", chunk, "--out", target, "--keys", *keys])
        # Free each split as soon as its channels are out, the way lightenBin.sh does. Keeping all
        # of them alive costs a second full copy of the source -- 24 GB on HRP5P_LongWalk -- and
        # that is what fills the disk, since the caller has already paid for the log itself.
        chunk.unlink(missing_ok=True)
        compact.append(target)
    merged = directory / "compact_merged.bin"
    run([sys.executable, ROOT / "scripts/routine_scripts/mergeBinLogs.py", merged, *compact])
    return merged


def source_log(project, cache, mc_rtc_config, robot, env, regenerate=True):
    """Pick the log to convert: the cached full one, the routine's if still complete, else a fresh tick."""
    cached = cache / "logReplay_full.bin"
    controller_log = project / "raw_data/controllerLog.bin"
    if cached.exists() and controller_log.exists() and cached.stat().st_mtime >= controller_log.stat().st_mtime:
        if not missing_log_entries(cached):
            return cached
    routine_log = project / "output_data/logReplay.bin"
    if routine_log.exists():
        missing = missing_log_entries(routine_log)
        if not missing:
            return routine_log
        print(f"{project.name}: {routine_log.name} was lightened (no {missing[0]})", flush=True)
    if not regenerate:
        raise RuntimeError(f"{project.name}: no complete observer log; drop --no-regenerate to rebuild it")
    return regenerate_full_log(project, cache, mc_rtc_config, robot, env)


def convert(source, cache, resolved_config, env, robot, config=None, passthrough=DEFAULT_PASSTHROUGH_CONFIG):
    with tempfile.TemporaryDirectory(prefix="mc_rtc_convert_") as directory:
        directory = Path(directory)
        conversion_source = (compact_conversion_source(source, directory)
                             if source.stat().st_size > SAFE_CONVERSION_BYTES else source)
        command = ["ros2", "run", "test_state_obs_ros2", "bin_to_rosbag_kinetics.py",
                   "--bin", conversion_source, "--out", cache / "input_bag", "--force",
                   "--robot", robot, "--write-config", resolved_config, "--require-exact",
                   "--observer-prefix", OBSERVER_PREFIX]
        if config is not None:
            run([*command, "--config", config], env)
            return
        mirrored = resolved_observer_tree(robot, directory / "observers", passthrough.expanduser())
        run([*command, "--mc-rtc-config", mirrored], env)


def prepare(args):
    env = ros_environment(args.workspace)
    dependencies = replay_dependencies(args.workspace, env)
    config = args.config.resolve() if args.config else None
    mc_rtc_config = args.mc_rtc_config.expanduser().resolve()
    if not mc_rtc_config.exists():
        raise FileNotFoundError(mc_rtc_config)
    for name in args.projects:
        project, cache = project_paths(name)
        robot = project_robot(project)
        observer = observer_robot(robot)
        if not (project / "output_data/logReplay.bin").exists() and not (project / "raw_data/controllerLog.bin").exists():
            raise FileNotFoundError(project / "raw_data/controllerLog.bin")
        bag = cache / "input_bag"
        ready = bag.exists() and (cache / "resolved_config.yaml").exists() and (cache / "reference/mocap.txt").exists() and (cache / "reference/riekf.txt").exists() and (cache / "manifest.json").exists()
        if ready and not args.force:
            try:
                validate_cache(cache, dependencies)
            except RuntimeError as error:
                print(f"{name}: {error}")
            else:
                print(f"{name}: cache exists")
                continue
        cache.mkdir(parents=True, exist_ok=True)
        resolved_config = cache / "resolved_config.yaml"
        cache_source = source_log(project, cache, mc_rtc_config, robot, env, not args.no_regenerate)
        try:
            convert(cache_source, cache, resolved_config, env, observer, config, args.passthrough_config)
        except subprocess.CalledProcessError:
            if args.no_regenerate or cache_source == cache / "logReplay_full.bin":
                raise
            # The log passed the entry check but the converter still rejected it: re-tick once.
            cache_source = regenerate_full_log(project, cache, mc_rtc_config, robot, env)
            convert(cache_source, cache, resolved_config, env, observer, config, args.passthrough_config)
        references(project, cache)
        configuration = yaml.safe_load((project / "projectConfig.yaml").read_text(encoding="utf-8")) or {}
        frame = str(configuration.get("Body_vel_eval", "FloatingBase")).strip()
        if frame not in ("FloatingBase", "IMU"):
            raise RuntimeError(f"{name}: unsupported Body_vel_eval {frame!r}")
        velocity = reference_velocities(project, cache, frame)
        (cache / "time_offset.json").write_text(
            json.dumps({"offset": reference_time_offset(project, cache), **velocity}) + "\n",
            encoding="utf-8")
        manifest = {"source": fingerprint(cache_source), "reference": fingerprint(project / "output_data/finalDataCSV.csv"), **dependencies}
        (cache / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: prepared {bag}")
    print(f"Editable tuning: {config}")


def sublengths(project):
    data = yaml.safe_load((project / "projectConfig.yaml").read_text(encoding="utf-8"))
    return [float(value) for value in data.get("predefined_sublengths", [1.0])]


def clip_to_span(trajectory, groundtruth, output):
    """Drop estimate samples outside the ground-truth time span.

    Some datasets (KO_TRO2024_RHPS1_5) have a mocap recording much shorter than the log.
    rpg aligns over every frame, so an estimate that runs past the ground truth wrecks the
    alignment and inflates the relative errors by an order of magnitude. plotAndFormatResults.py
    trims the same way for the routine's own evaluation.
    """
    import numpy as np
    estimate = np.loadtxt(trajectory, comments="#", ndmin=2)
    reference = np.loadtxt(groundtruth, comments="#", ndmin=2)
    first, last = reference[0, 0], reference[-1, 0]
    kept = estimate[(estimate[:, 0] >= first) & (estimate[:, 0] <= last)]
    if len(kept) < 2:
        raise RuntimeError(f"{trajectory}: no samples inside the ground-truth span [{first}, {last}]")
    with output.open("w", encoding="utf-8") as stream:
        stream.write("# timestamp tx ty tz qx qy qz qw\n")
        for row in kept:
            stream.write(" ".join(f"{value:.17g}" for value in row) + "\n")
    return len(estimate) - len(kept)


def analyze(trajectory, groundtruth, directory, lengths):
    directory.mkdir(parents=True)
    dropped = clip_to_span(trajectory, groundtruth, directory / "stamped_traj_estimate.txt")
    if dropped:
        print(f"clipped {dropped} estimate samples outside the ground-truth span", flush=True)
    shutil.copy2(groundtruth, directory / "stamped_groundtruth.txt")
    # Must match scripts/routine_scripts/computeMetrics.sh, or the numbers are not comparable.
    (directory / "eval_cfg.yaml").write_text("align_type: posyaw\nalign_num_frames: -1\n", encoding="utf-8")
    command = [sys.executable, ROOT / "rpg_trajectory_evaluation/scripts/analyze_trajectory_single.py",
               directory, "--recalculate_errors", "--no_plot", "--estimator_name", "Kinetics",
               "--predefined_sublengths", *lengths]
    run(command)


def stats_files(directory):
    return sorted((directory / "saved_results/traj_est").glob("relative_error_statistics_*.yaml"))


def read_stats(path, project, estimator):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    distance = float(path.stem.removeprefix("relative_error_statistics_").replace("_", "."))
    rows = []
    # trans_xy and trans_z split the relative translation error the way the tuning objective
    # scores it: horizontal drift and vertical drift answer to different parts of the model.
    for metric, source in (("trans", "trans"), ("trans_perc", "trans_perc"),
                           ("trans_xy", "trans_x_y_norm"), ("trans_z", "trans_z"),
                           ("yaw", "yaw"), ("tilt", "gravity"), ("rot", "rot")):
        if source not in data:
            continue
        for statistic, value in data[source].items():
            if statistic != "num_samples":
                rows.append({"project": project, "estimator": estimator, "distance": distance,
                             "metric": metric, "statistic": statistic, "value": value})
    return rows


def existing_riekf_stats(project, lengths):
    directory = project / "output_data/evals/Hartley/saved_results/traj_est"
    output = []
    for distance in lengths:
        path = directory / f"relative_error_statistics_{distance:.1f}".replace(".", "_")
        path = path.with_suffix(".yaml")
        if not path.exists():
            raise FileNotFoundError(f"RI-EKF baseline is missing: {path}")
        output.extend(read_stats(path, project.name, "RI-EKF"))
    return output


def load_trajectory(path):
    import numpy as np
    data = np.loadtxt(path, comments="#", ndmin=2)
    return data[:, 0], data[:, 1:4], data[:, 4:8]


def rpy_degrees(quaternion):
    import numpy as np
    x, y, z, w = quaternion.T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.rad2deg(np.column_stack((roll, pitch, yaw)))


def plot_trajectory_states(project, kinetics, mocap, riekf, destination):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    trajectories = {"Mocap": load_trajectory(mocap), "RI-EKF": load_trajectory(riekf), "Kinetics": load_trajectory(kinetics)}
    colors = {"Mocap": "black", "RI-EKF": "#ff7f0e", "Kinetics": "#1f77b4"}
    titles = ("XY trajectory", "Position (x/y/z)", "Roll", "Pitch", "Yaw", "")
    fig = make_subplots(rows=2, cols=3, subplot_titles=titles)
    for name, (time, position, quaternion) in trajectories.items():
        color = colors[name]
        fig.add_trace(go.Scatter(x=position[:, 0], y=position[:, 1], mode="lines", name=name, legendgroup=name, line={"color": color, "dash": "dash" if name == "Mocap" else "solid"}), row=1, col=1)
        for index, axis_name in enumerate(("x", "y", "z")):
            fig.add_trace(go.Scatter(x=time, y=position[:, index], mode="lines", name=f"{name} {axis_name}", legendgroup=name, showlegend=False, line={"color": color, "dash": "dash" if name == "Mocap" else "solid"}), row=1, col=2)
        for column, values in enumerate(rpy_degrees(quaternion).T, start=1):
            fig.add_trace(go.Scatter(x=time, y=values, mode="lines", name=name, legendgroup=name, showlegend=False, line={"color": color, "dash": "dash" if name == "Mocap" else "solid"}), row=2, col=column)
    fig.update_xaxes(title_text="x [m]", row=1, col=1)
    fig.update_yaxes(title_text="y [m]", scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(title_text="time [s]", row=1, col=2)
    fig.update_yaxes(title_text="position [m]", row=1, col=2)
    for column in range(1, 4):
        fig.update_xaxes(title_text="time [s]", row=2, col=column)
        fig.update_yaxes(title_text="angle [deg]", row=2, col=column)
    fig.update_layout(title=f"{project}: trajectory and attitude", height=800, hovermode="x unified")
    path = destination / "plots" / f"{project}_trajectory_states.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs=True)
    return path


def skipped_iteration_shift(project, timestep):
    """Cumulative wall time the controller lost to skipped iterations, per log row.

    mc_rtc writes one log row per *executed* iteration, so a row whose perf_GlobalRun exceeded the
    controller period stands for several periods of elapsed time. The routine corrects for this in
    repair_mc_rtc_skipped_iters.py (called from mainRoutine.sh:362) and keeps the corrected stamps;
    the replay stamps from the bag, which carries the log's uncorrected uniform `t`. Without the
    same correction the two pipelines drift apart -- 60 ms by the end of KO_TRO2024_RHPS1_4 -- and
    rpg, which matches estimate to ground truth *by timestamp*, then pairs different samples on
    each side. That biases only the Kinetics column, since the RI-EKF baseline is read from the
    routine's own (corrected) results.
    """
    import numpy as np
    path = project / "output_data/perf_GlobalRun_log.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        delays = [float(row["perf_GlobalRun"]) for row in csv.DictReader(stream, delimiter=";")]
    # The repair compares against the period in milliseconds and skips row 0, as here.
    step_ms = timestep * 1000.0
    skipped = np.zeros(len(delays))
    for index, delay in enumerate(delays):
        if index and delay > step_ms:
            skipped[index] = int(delay / step_ms) - 1
    return np.cumsum(skipped) * timestep


def extract_ros_trajectories(bag, floating_output, centroid_output, velocity_output=None, env=None,
                             offset=0.0, shift=None):
    if env:
        os.environ.update(env)
        prefixes = env.get("AMENT_PREFIX_PATH", "").split(os.pathsep)
        library_paths = [str(Path(prefix) / "lib") for prefix in prefixes if prefix and (Path(prefix) / "lib").is_dir()]
        old_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(library_paths + ([old_library_path] if old_library_path else []))
        for path in env.get("PYTHONPATH", "").split(os.pathsep):
            if path and path not in sys.path:
                sys.path.insert(0, path)
    import ctypes
    for prefix in prefixes if env else []:
        library_dir = Path(prefix) / "lib"
        libraries = list(library_dir.glob("libkinetics_observer_ros2*.so"))
        libraries += list(library_dir.parent.glob("lib/python*/site-packages/kinetics_observer_ros2/*.so"))
        for library in sorted(libraries):
            try:
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from kinetics_observer_ros2.msg import KineticsState
    storage_id = "mcap" if any(Path(bag).glob("*.mcap")) else "sqlite3"
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=storage_id), rosbag2_py.ConverterOptions("", ""))
    streams = {"floating": floating_output.open("w", encoding="utf-8"), "centroid": centroid_output.open("w", encoding="utf-8")}
    if velocity_output is not None:
        streams["velocity"] = velocity_output.open("w", encoding="utf-8")
    try:
        for name, stream in streams.items():
            stream.write(("# timestamp vx vy vz wx wy wz" if name == "velocity" else
                          "# timestamp tx ty tz qx qy qz qw") + chr(10))
        count = 0
        while reader.has_next():
            topic, serialized, _ = reader.read_next()
            if topic != "/kinetics_observer/estimated_state":
                continue
            state = deserialize_message(serialized, KineticsState)
            stamp = state.header.stamp.sec + state.header.stamp.nanosec * 1.0e-9 - offset
            if shift is not None and len(shift):
                # One bag message per log row, in order, so the row index is the message index.
                stamp += float(shift[count if count < len(shift) else -1])
            for key, kinematics in (("floating", state.global_floating_base_kinematics), ("centroid", state.global_centroid_kinematics)):
                values = (stamp, kinematics.position.x, kinematics.position.y, kinematics.position.z,
                          kinematics.orientation.x, kinematics.orientation.y, kinematics.orientation.z, kinematics.orientation.w)
                streams[key].write(" ".join(f"{value:.17g}" for value in values) + chr(10))
            if "velocity" in streams:
                velocity = state.global_floating_base_kinematics
                values = (stamp, velocity.linear_velocity.x, velocity.linear_velocity.y, velocity.linear_velocity.z,
                          velocity.angular_velocity.x, velocity.angular_velocity.y, velocity.angular_velocity.z)
                streams["velocity"].write(" ".join(f"{value:.17g}" for value in values) + chr(10))
            count += 1
    finally:
        for stream in streams.values():
            stream.close()
    if count == 0:
        raise RuntimeError(f"replay bag has no /kinetics_observer/estimated_state: {bag}")
    return count


def plot_trajectory_states(project, kinetics, mocap, riekf, destination, centroid=None):
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.spatial.transform import Rotation
    trajectories = {"Mocap": load_trajectory(mocap), "RI-EKF": load_trajectory(riekf), "Kinetics (floating base)": load_trajectory(kinetics)}
    if centroid is not None:
        trajectories["Kinetics (centroid)"] = load_trajectory(centroid)
    reference_time, reference_position, reference_quaternion = trajectories["Mocap"]
    reference_rotation = Rotation.from_quat(reference_quaternion[0]).as_matrix()
    for name, (time, position, quaternion) in list(trajectories.items()):
        if name == "Mocap":
            continue
        estimate_rotation = Rotation.from_quat(quaternion[0]).as_matrix()
        relative = estimate_rotation @ reference_rotation.T
        theta = np.pi / 2 - np.arctan2(relative[0, 0] + relative[1, 1], relative[0, 1] - relative[1, 0])
        alignment = Rotation.from_euler("z", theta)
        aligned_position = alignment.apply(position)
        aligned_position += reference_position[0] - aligned_position[0]
        aligned_quaternion = (alignment * Rotation.from_quat(quaternion)).as_quat()
        trajectories[name] = (time, aligned_position, aligned_quaternion)
    colors = {"Mocap": "#202124", "RI-EKF": "#e67e22", "Kinetics (floating base)": "#1769aa", "Kinetics (centroid)": "#8e44ad"}
    dashes = {"Mocap": "dash", "RI-EKF": "solid", "Kinetics (floating base)": "solid", "Kinetics (centroid)": "dot"}
    titles = ("XY trajectory", "Position X", "Position Y", "Position Z", "Roll", "Pitch", "Yaw", "")
    fig = make_subplots(rows=2, cols=4, subplot_titles=titles, horizontal_spacing=0.06, vertical_spacing=0.12)
    for name, (time, position, quaternion) in trajectories.items():
        line = {"color": colors[name], "dash": dashes[name], "width": 2}
        fig.add_trace(go.Scatter(x=position[:, 0], y=position[:, 1], mode="lines", name=name, legendgroup=name, line=line), row=1, col=1)
        for index, axis_name in enumerate(("x", "y", "z")):
            fig.add_trace(go.Scatter(x=time, y=position[:, index], mode="lines", name=f"{name} {axis_name}", legendgroup=name, showlegend=False, line=line), row=1, col=index + 2)
        for column, values in enumerate(rpy_degrees(quaternion).T, start=1):
            fig.add_trace(go.Scatter(x=time, y=values, mode="lines", name=f"{name} RPY", legendgroup=name, showlegend=False, line=line), row=2, col=column)
    for column in range(1, 4):
        fig.update_xaxes(title_text="time [s]", row=1, col=column + 1)
        fig.update_yaxes(title_text="position [m]", row=1, col=column + 1)
        fig.update_xaxes(title_text="time [s]", row=2, col=column)
        fig.update_yaxes(title_text="angle [deg]", row=2, col=column)
    fig.update_xaxes(title_text="x [m]", row=1, col=1)
    fig.update_yaxes(title_text="y [m]", scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_layout(title={"text": f"{project}: trajectory, position and attitude", "x": 0.5}, template="plotly_white", height=780, hovermode="x unified", legend={"orientation": "h", "y": 1.04, "x": 0, "groupclick": "togglegroup"}, margin={"l": 60, "r": 30, "t": 120, "b": 55})
    path = destination / "plots" / f"{project}_trajectory_states.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs=True)
    return path


def report(rows, output, plots, minimal=False):
    fields = ["project", "estimator", "distance", "metric", "statistic", "value"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if minimal:
        return
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    medians = [row for row in rows if row["statistic"] == "median" and row["metric"] in
               {"trans_perc", "yaw", "tilt"}]
    figure = make_subplots(rows=1, cols=3, subplot_titles=("Translation (%)", "Yaw (deg)", "Tilt (deg)"))
    for column, metric in enumerate(("trans_perc", "yaw", "tilt"), start=1):
        for estimator in ("Kinetics", "RI-EKF"):
            selected = [row for row in medians if row["metric"] == metric and row["estimator"] == estimator]
            figure.add_trace(go.Bar(name=estimator,
                                    x=[f'{row["project"]} @ {row["distance"]:g} m' for row in selected],
                                    y=[row["value"] for row in selected], legendgroup=estimator,
                                    showlegend=column == 1), row=1, col=column)
    figure.update_layout(barmode="group", title={"text": "Relative error medians", "x": 0.5}, template="plotly_white", height=520, margin={"l": 60, "r": 30, "t": 90, "b": 150}, legend={"orientation": "h", "y": 1.08, "x": 0})
    figure.update_xaxes(tickangle=-35)
    figure.write_html(output / "report.html")
    report_path = output / "report.html"
    gallery = ["<style>body{font-family:Arial,sans-serif;max-width:1500px;margin:auto;padding:0 20px} iframe{display:block;width:100%;height:780px;border:0;margin-bottom:24px}</style>", "<h1>Trajectory plots</h1>"]
    gallery.extend(f"<h2>{project}</h2><iframe src='{plot.relative_to(output)}' style='width:100%;height:780px;border:0;'></iframe><br>" for project, plot in plots)
    html = report_path.read_text(encoding="utf-8")
    report_path.write_text(html.replace("</body>", "\n".join(gallery) + "</body>"), encoding="utf-8")

    print("\nproject                              estimator  dist(m)  trans%    yaw°    tilt°")
    grouped = {(row["project"], row["estimator"], row["distance"]): {} for row in medians}
    for row in medians:
        grouped[(row["project"], row["estimator"], row["distance"])][row["metric"]] = row["value"]
    for (project, estimator, distance), values in sorted(grouped.items()):
        print(f"{project:36} {estimator:9} {distance:7g} {values['trans_perc']:7.3f} "
              f"{values['yaw']:7.3f} {values['tilt']:7.3f}")


def require_evaluation_dependencies():
    missing = [name for name in ("numba", "alive_progress") if importlib.util.find_spec(name) is None]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"evaluation needs {names}; install with: .venv/bin/python -m pip install -r requirements.txt")


def evaluate(args):
    require_evaluation_dependencies()
    env = ros_environment(args.workspace)
    dependencies = replay_dependencies(args.workspace, env)
    if args.config and args.covariance_overlay:
        raise ValueError("--config and --covariance-overlay are mutually exclusive")
    config_paths = ([args.config] if args.config else
                    [project_paths(name)[1] / "resolved_config.yaml" for name in args.projects])
    if any(not path.exists() for path in config_paths):
        missing = next(path for path in config_paths if not path.exists())
        raise FileNotFoundError(missing)
    effective = {}
    if args.covariance_overlay:
        overlay = yaml.safe_load(args.covariance_overlay.read_text(encoding="utf-8"))
        for name, path in zip(args.projects, config_paths):
            configuration = yaml.safe_load(path.read_text(encoding="utf-8"))
            merged = merge_covariance_overlay(configuration, overlay,
                                              observer_robot(project_robot(project_paths(name)[0])))
            effective[name] = yaml.safe_dump(merged, sort_keys=False).encode()
        config_bytes = args.covariance_overlay.read_bytes() + b"".join(
            name.encode() + effective[name] for name in args.projects
        )
    else:
        config_bytes = b"".join(path.read_bytes() for path in config_paths)
    digest = hashlib.sha256(config_bytes).hexdigest()[:10]
    label = args.label or time.strftime("%Y%m%d-%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ValueError("label may contain only letters, numbers, dot, underscore, and dash")
    output = ROOT / "results" / f"{label}-{digest}"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    if args.config:
        shutil.copy2(args.config, output / "kinetics.yaml")
    elif args.covariance_overlay:
        shutil.copy2(args.covariance_overlay, output / "covariance_overlay.yaml")
    else:
        shutil.copy2(DEFAULT_MCKINETICS_CONFIG.expanduser(), output / "MCKineticsObserver.yaml")

    rows = []
    plots = []
    for name in args.projects:
        project, cache = project_paths(name)
        input_bag = cache / "input_bag"
        if not input_bag.exists():
            raise FileNotFoundError(f"{name}: run prepare first")
        validate_cache(cache, dependencies)
        destination = output / name
        destination.mkdir()
        config = args.config if args.config else cache / "resolved_config.yaml"
        if args.covariance_overlay:
            config = destination / "resolved_config.yaml"
            config.write_bytes(effective[name])
        tuning_bag = destination / "configuration"
        run(["ros2", "run", "test_state_obs_ros2", "make_kinetics_config_bag.py",
             "--input-bag", input_bag, "--config", config, "--out", tuning_bag], env)
        trajectory = destination / "kinetics.txt"
        centroid = destination / "kinetics_centroid.txt"
        replay_bag = destination / "standalone_replay"
        run(["ros2", "launch", "test_state_obs_ros2", "test_kinetics_replay.launch.py",
             f"input_bag:={input_bag}", f"configuration_bag:={tuning_bag}",
             f"output_bag:={replay_bag}", "startup_delay:=0.5", "shutdown_delay:=0.5"], env)
        offset_path = cache / "time_offset.json"
        offset = json.loads(offset_path.read_text(encoding="utf-8"))["offset"] if offset_path.exists() else 0.0
        shift = skipped_iteration_shift(project, project_timestep(project))
        extract_ros_trajectories(replay_bag, trajectory, centroid, destination / "kinetics_velocity.txt",
                                 env, offset, shift)
        if args.no_plots:
            # Tuning inner loop: the bags are regenerable intermediates and cost GBs per trial.
            shutil.rmtree(replay_bag, ignore_errors=True)
            shutil.rmtree(tuning_bag, ignore_errors=True)
        lengths = sublengths(project)
        evaluation = destination / "eval"
        analyze(trajectory, cache / "reference/mocap.txt", evaluation, lengths)
        if not args.no_plots:
            plots.append((name, plot_trajectory_states(name, trajectory, cache / "reference/mocap.txt", project / "output_data/evals/Hartley/stamped_traj_estimate.txt", output, centroid)))
        for path in stats_files(evaluation):
            rows.extend(read_stats(path, name, "Kinetics"))
        rows.extend(existing_riekf_stats(project, lengths))

    report(rows, output, plots, args.no_plots)
    if not args.no_latest:
        latest = ROOT / "results/latest"
        if latest.is_symlink():
            latest.unlink()
        elif latest.exists():
            raise FileExistsError(f"refusing to replace non-symlink {latest}")
        latest.symlink_to(output.name, target_is_directory=True)
    report_path = output / "report.html"
    if not args.no_open and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        webbrowser.open(report_path.as_uri())
    print(f"\nResults: {output}\nPlotly report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare and evaluate Kinetics on MultiContact and Slippage datasets")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--config", type=Path, help="Optional ROS2 Kinetics YAML override; default uses mc_rtc MCKineticsObserver config")
    parser.add_argument("--covariance-overlay", type=Path, help="Covariances merged into each robot-specific replay config")
    parser.add_argument("--mc-rtc-config", type=Path, default=DEFAULT_MC_RTC_CONFIG)
    # The observer config was hardcoded, so an experiment on anything outside the covariance
    # overlay -- contact detection, which sensors feed which contact -- had no way in except
    # editing the user's global mc_rtc config in place.
    parser.add_argument("--observer-config", type=Path, default=None,
                        help="MCKineticsObserver YAML to use instead of the mc_rtc one")
    parser.add_argument("--passthrough-config", type=Path, default=DEFAULT_PASSTHROUGH_CONFIG,
                        help="Controller YAML holding the inline MCKineticsObserver config layer")
    parser.add_argument("--projects", help="Comma-separated subset of the datasets to process")
    parser.add_argument("--cache-suffix", default="",
                        help="Use an isolated output_data/kinetics_eval<SUFFIX> cache")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.set_defaults(no_plots=False, no_latest=False, no_open=False)
    prepare_parser.add_argument("--no-regenerate", action="store_true",
                                help="Fail instead of re-ticking raw_data/controllerLog.bin")
    # Kept for compatibility: regenerating is now the default.
    prepare_parser.add_argument("--regenerate-missing", action="store_true", help=argparse.SUPPRESS)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--label")
    run_parser.add_argument("--no-plots", action="store_true", help="Skip the plotly report (tuning inner loop)")
    run_parser.add_argument("--no-latest", action="store_true", help="Do not move the results/latest symlink")
    run_parser.add_argument("--no-open", action="store_true", help="Never open the report in a browser")
    args = parser.parse_args()
    global CACHE_SUFFIX
    if args.cache_suffix and not re.fullmatch(r"_[A-Za-z0-9_.-]+", args.cache_suffix):
        parser.error("invalid --cache-suffix")
    CACHE_SUFFIX = args.cache_suffix
    # Applies to every subcommand: 'run' never calls prepare(), so setting this there did
    # nothing and the experiment silently used the default config.
    if args.observer_config is not None:
        global DEFAULT_MCKINETICS_CONFIG
        DEFAULT_MCKINETICS_CONFIG = args.observer_config.expanduser().resolve()
        if not DEFAULT_MCKINETICS_CONFIG.exists():
            raise FileNotFoundError(DEFAULT_MCKINETICS_CONFIG)
    args.workspace = args.workspace.resolve()
    args.config = args.config.resolve() if args.config else None
    args.projects = selected_projects(args.projects)
    prepare(args) if args.command == "prepare" else evaluate(args)


if __name__ == "__main__":
    try:
        main()
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"kinetics_eval: {error}", file=sys.stderr)
        raise SystemExit(2)

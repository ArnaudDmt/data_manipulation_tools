"""Re-tick a project the way mainRoutine.sh does, keeping the mocap plugin.

kinetics_eval.py rewrites the plugin list to `[HartleyIEKF]` (line 362) because the tuning loop
reads the ground truth elsewhere.  That is why its logs carry no MocapAligner_worldBodyKine_*
channels and cannot feed the routine.  Everything else -- controller config, robot, timestep,
pinned log destination -- is taken from kinetics_eval so the observer that writes this log is the
one the routine runs.
"""
import re, sys, time, shutil, tempfile, subprocess, pathlib

sys.path.insert(0, 'scripts')
import kinetics_eval as ke


def replay_config(source, robot, timestep, log_directory, template):
    text = source.read_text(encoding="utf-8")
    if not re.search(r"(?m)^MainRobot:\s*", text):
        raise RuntimeError(f"{source}: no active MainRobot entry")
    text = re.sub(r"(?m)^MainRobot:\s*.*$", f"MainRobot: {robot}", text, count=1)
    # The plugin line is deliberately left alone: MocapAligner must stay.
    rendered = f"Timestep: {timestep:.9g}"
    text = (re.sub(r"(?m)^Timestep:\s*.*$", rendered, text, count=1)
            if re.search(r"(?m)^Timestep:\s*", text) else text + f"\n{rendered}\n")
    text = re.sub(r"(?m)^LogDirectory:\s*.*$", "", text)
    text = re.sub(r"(?m)^LogTemplate:\s*.*$", "", text)
    text += f"\nLogDirectory: {log_directory}\nLogTemplate: {template}\n"
    handle = tempfile.NamedTemporaryFile(prefix="mc_rtc_retick_", suffix=".yaml",
                                         mode="w", delete=False)
    with handle:
        handle.write(text)
    return pathlib.Path(handle.name)



def set_mocap_body(project):
    """mainRoutine.sh rewrites the plugin's bodyName from projectConfig before every tick.

    RHPS1 logs it as `BODY`, HRP5-P as `Body`; leaving the wrong one makes MocapAligner abort
    with "Please give an available body" (MocapAligner.cpp:30).
    """
    import yaml as _yaml
    body = str((_yaml.safe_load((project / "projectConfig.yaml").read_text()) or {}).get("EnabledBody", "")).strip()
    if not body:
        raise RuntimeError(f"{project.name}: projectConfig.yaml has no EnabledBody")
    config = pathlib.Path.home() / ".config/mc_rtc/plugins/MocapAligner.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    previous = config.read_text() if config.exists() else ""
    if re.search(r"(?m)^bodyName:", previous):
        config.write_text(re.sub(r"(?m)^bodyName:.*$", f"bodyName: {body}", previous, count=1))
    else:
        config.write_text(f"bodyName: {body}\n" + previous)
    return body


def retick(name, destination):
    project = ke.ROOT / "Projects" / name
    controller_log = project / "raw_data/controllerLog.bin"
    if not controller_log.exists():
        raise FileNotFoundError(controller_log)
    body = set_mocap_body(project)
    robot = ke.project_robot(project)
    timestep = ke.project_timestep(project)
    tick_directory = pathlib.Path(tempfile.mkdtemp(prefix="retick_", dir=str(destination.parent)))
    config = replay_config(ke.DEFAULT_MC_RTC_CONFIG, robot, timestep, tick_directory, "replay")
    started = time.time()
    try:
        subprocess.run(["mc_rtc_ticker", "-f", str(config), "--no-sync", "--replay-outputs",
                        "-e", "-l", str(controller_log)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        produced = [p for p in tick_directory.glob("replay*.bin") if "latest" not in p.name]
        if not produced:
            produced = [ke.newest_replay(started)]
        shutil.move(str(max(produced, key=lambda p: p.stat().st_mtime)), destination)
    finally:
        config.unlink(missing_ok=True)
        shutil.rmtree(tick_directory, ignore_errors=True)
    return time.time() - started, destination.stat().st_size


if __name__ == "__main__":
    name, out = sys.argv[1], pathlib.Path(sys.argv[2])
    elapsed, size = retick(name, out)
    print(f"{name}: re-tick {elapsed:.1f} s, {size / 1024**3:.2f} Go -> {out}", flush=True)

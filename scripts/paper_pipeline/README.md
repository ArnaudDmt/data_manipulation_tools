# Rebuilding the paper's measured content

Everything the IJRR paper prints as a number or a data figure is produced from here.

    scripts/paper_pipeline/run.sh                     # routine, metrics, figures
    scripts/paper_pipeline/run.sh replay              # optional cached offline analysis
    scripts/paper_pipeline/run.sh routine hidehand    # one stage, one variant
    scripts/paper_pipeline/run.sh figures poseAndVel  # one stage, one figure

`manifest.py` is the single source of truth: datasets, categories and their sub-trajectory
lengths, observer variants, and which script produces which figure.

Paper sources: `~/Documents/ResearchNotes/Topics/KineticsObserver/Papers/IJRR/Third_submission/Paper/`
(`manifest.PAPER`), repository `ArnaudDmt/KineticsObserver`, branch `IJRR_third`.

## The Kinetics Observer of the paper, and how to check it is still the one installed

The published numbers come from the **routine runs of 2026-09-18 04:46**
(`results/paper-rebuild/runs/*/`), pooled into the paper's `metrics_results.tex` at 04:49. What
produced them is frozen under the tag **`ijrr-ko-2026-09-18`** in every repository involved, and
recorded in `paper_ko.lock.json`:

| what | where | identity |
|---|---|---|
| estimator library | `ArnaudDmt/state-observation` | commit `2b35f2d4`; `libstate-observation.so.1.6.1` md5 `d15f656e…` |
| mc_rtc observers (KO, VALINOR) | `ArnaudDmt/mc_state_observation`, branch `KO_IJRR_2026_09` | commit `04e8cf8`; `MCKineticsObserver.so` md5 `aeb58f43…`, `MCValinor.so` md5 `a99a1ab5…` |
| metric | `ArnaudDmt/rpg_trajectory_evaluation` (submodule) | commit `d767669` |
| RI-EKF input logging (mc_rtc plugin) | `ArnaudDmt/HartleyIEKF` | commit `27da130` |
| sensor noise injection (mc_rtc plugin, noise experiments) | `ArnaudDmt/NoisySensors` | commit `b36c644` |
| ROS 2 replay (verification and variant work only) | `ArnaudDmt/state_observation_ros2` with `kinetics_observer_ros2` and `test_state_obs_ros2` | commits `d7f3d6b`, `06325f9`, `00de2e6` |
| tuning and controller | `config_base/` in this repository | sha256 of every file in the lock |

A copy of the three binaries is kept in `results/paper-rebuild/so_paper_20260918/`.
(`so_backup_20260918_0219/` is an earlier copy taken before `MCValinor.so` was rebuilt with the
Schmitt thresholds; do not restore VALINOR from it.)

**Check before building anything on top of it:**

    .venv/bin/python scripts/paper_pipeline/verify_paper_ko.py            # hashes, instant
    .venv/bin/python scripts/paper_pipeline/verify_paper_ko.py --replay   # + two replays, minutes

`--replay` replays `HRP5_MultiContact_1` (0.3 m) and `KO_TRO2024_RHPS1_1` (1 m) in an isolated
cache (`output_data/kinetics_eval_verif`, built from the existing `logReplay_full.bin`, the original
bag is never touched) and compares the rpg means with the reference values of the lock file. On
2026-09-28 they agreed to 5-6 digits. If it says NON CONFORME, stop and find what moved.

Reference values (`results_summary/relative_errors.json`, key `clean`, `sum_abs / count`):

| | Transxy | Transz | yaw | tilt | rot | gravity |
|---|---|---|---|---|---|---|
| HRP5_MultiContact_1, 0.3 m, 8315 | 0.004342 | 0.001292 | 0.153389 | 0.319940 | 0.374121 | 0.321643 |
| KO_TRO2024_RHPS1_1, 1 m, 78755 | 0.017551 | 0.005291 | 0.371475 | 0.588188 | 0.735381 | 0.587415 |

Values dated 2026-09-14 still circulate (MultiContact_1 Transxy 0.004452): they predate the
change of the initial covariances of 2026-09-16 (velocities 1e-12/0 -> 1e-6, disturbance wrench
0 -> 100, first-contact rest pose 0 -> 5e-6 / 2e-4) and are NOT the paper's. The one rounded paper
value that tells the two runs apart is LongWalk Transz: 0.037 (0.03747) against 0.03755.

## Environment

- Two Python interpreters, both ignored by git. `env/bin/python` (3.10) runs the pipeline
  scripts; `.venv/bin/python` (3.12, `requirements.txt`) is the only one that can import
  `rosbag2_py`, so `kinetics_eval.py` and `verify_paper_ko.py` need it.
- mc_rtc and the observers are installed in `/home/arnaud/devel/install`. Rebuilding
  `mc_state_observation` or `state-observation` changes the md5s above: check them before and
  after, and rebuild BOTH the mc_rtc side and the ROS 2 replay from the same revision before any
  A/B (a build skew once produced a 2.5 mm "estimator difference").
- ROS 2 replay: `~/devel/src/catkin_ws/src/state_observation_ros2/` (no dependency on
  mc_state_observation). Build with
  `colcon build --packages-select kinetics_observer_ros2 test_state_obs_ros2` from
  `~/devel/src/catkin_ws`, then `source install/setup.zsh`. Use `ROS_DOMAIN_ID=77`.
- The MocapAligner mc_rtc plugin is built from `mc_rtc_plugin/` (`CMakeLists.txt`, `build` is a
  symlink to `/home/arnaud/devel/build/data_manipulation_tools`).

## Stages

| stage | what it runs | what it leaves |
|---|---|---|
| `replay` | `kinetics_eval.py` per variant over the 13 datasets | `results/var-<label>-<hash>/` — analysis only |
| `routine` | one mc_rtc tick per dataset running KO, KO-ZPC, KO-PC and KONOANG together; separate ticks for flexibility / hidden-hand / orientation-error / KO-Lin | `results/paper-rebuild/runs/<variant>/<project>/`; and `Projects/*/output_data`, which every figure reads |
| `metrics` | pools the routine snapshots | `results/paper-rebuild/macros/*.tex`, folded into the paper's `metrics_results.tex` |
| `figures` | every figure script of `manifest.FIGURES` | the figures next to `main.tex`, plus PNG **and** PDF of all of them in `figures-export/` |

Stages are independently runnable and each leaves its snapshots behind. After `metrics`, run
`env/bin/python scripts/paper_pipeline/distill.py` to refresh `results_summary/` (the committed,
lossless summary; plain keys = routine = paper, `var-*` keys = replay = analysis).

Figures outside `manifest.FIGURES`: `injectedGyroBias.pdf` comes from `fig_gyrobias.py`, which
reads `results/bmi-b8/` and `results/kolin-bias/`; the injected-bias study is in `gyrobias/`
(its own README). Not automated: `compute_time.pdf` (no producing script),
`slipping-odom-traj.pdf` (its inset is added by hand, see `TODO-figures.md` next to `main.tex`),
and the VALINOR macros of `metrics_results.tex`.

## Routine versus replay

- **The paper's relative errors come from the ROUTINE** (`metrics.py` reads
  `results/paper-rebuild/runs/`), since the replay's evaluated window once started 50 ms early on
  `KO_TRO2024_RHPS1_5`. That window was fixed on 2026-09-16; the two now agree to 5-6 digits on
  every 200 Hz dataset.
- **Not on `HRP5P_LongWalk`** (500 Hz, 1.47 M iterations): round-off accumulates to 89 mm, and the
  routine evaluates it at 250 Hz by design (`initialize_datas.py` halves logs with dt <= 3 ms).
  Never A/B LongWalk across pipelines. It is also 70 % of the whole evaluation cost.
- **Use the replay for a new estimator variant**: faster, isolated from the controller, driven by
  YAML. Use the routine for anything that changes the robot model, the controller, the logged
  channels, or that must end up in a figure.
- Two replays of the same cache are not bit-identical (states differ by ~1e-12): compare metrics.
- The replay cache (`output_data/kinetics_eval*/`) bakes the contact configuration and kinematics
  in at conversion; `prepare` fingerprints the inputs by content and refuses a stale cache.
  **Never modify the original `input_bag`**: use `--cache-suffix _<name>` for an isolated cache.
- `kinetics_eval.py` prints MEDIANS; the paper reports MEANS of absolute values pooled per
  category. Read the pickles (`eval/saved_results/traj_est/cached/cached_rel_err.pickle`).
- Report translation as the paper does: `rel_trans_x_y_norm` (Transxy) and `rel_trans_z` (Transz)
  in metres, plus `rel_tilt` and `rel_yaw` in degrees. Not `rel_trans_perc`.
- Always score with rpg (`posyaw` alignment per sub-trajectory), never with an ad-hoc scorer.

## Configuration layers

`config_base/` is the versioned configuration of the paper. Every routine pass materialises a
private HOME (`config_home.py`, `KO_CONFIG_HOME`), because mc_rtc reads everything from
`$HOME/.config/mc_rtc`: it COPIES the whole real `~/.config/mc_rtc`, then overwrites it with every
file of `config_base/`. So `~/.config/mc_rtc` is never written, but any file it holds that
`config_base/` does not is still read by the run. That is why every file the paper's tick reads is
in `config_base/`: mc_rtc.yaml, Passthrough.yaml, the Kinetics Observer AND VALINOR observer files
(VALINOR is the mocap alignment reference and the RI-EKF's contact source), and the plugins. The
Encoder observer has no configuration file. Add a file here as soon as a run starts depending on
it. (`KO_LIVE_CONFIG=1` restores the old install-and-restore behaviour.)

Not versioned anywhere: the robot models. Both URDFs in `~/devel/src/catkin_data_ws` (isri-aist
repositories) carry local mass changes (HRP-5P root body 9.8635 -> 1e-6 kg, RHPS1 chest
24.326 -> 18.7902 kg) that the paper ran with.

Precedence, lowest first:

1. the package's own configuration, `mc_state_observation/etc/`, as INSTALLED in
   `/home/arnaud/devel/install/lib/mc_observers/` (`etc/MCKineticsObserver.yaml`,
   `MCKineticsObserver/<robot>.yaml`, `etc/MCValinor.yaml`); mc_rtc reads these copies, not the
   sources. Their sha256 are in the lock file, since a reinstall would change them silently;
2. `observers/MCKineticsObserver.yaml`;
3. `observers/MCKineticsObserver/<robot>.yaml` — **wins over 2**. Keys declared in both (e.g.
   `surfacesForContactDetection`) must be edited in the robot file;
4. the inline `config:` block of each instance in `controllers/Passthrough.yaml` — highest.

Traps in those layers:
- The flexibilities (`linStiffness`, `angStiffness`, `linDamping`, `angDamping`) are read ONLY
  through `config("contacts")`. Declared at the root of a robot file they are silently ignored and
  the package's values are used (that made the flexibility variants inert on RHPS1 until
  2026-09-17). A result identical to the reference usually means an edit did not take.
- `retick_routine.py` rewrites `plugins/MocapAligner.yaml`'s `bodyName` per dataset (`Body` for
  HRP-5P, `BODY` for RHPS1).
- **`~/.config/mc_rtc` is kept equal to `config_base/`** (synced 2026-09-28; the previous files are
  in `results/paper-rebuild/backups/dotconfig-mc_rtc-20260928/`). The replay reads `~/.config/mc_rtc`
  unless given `--observer-config` / `--passthrough-config`, any manual mc_rtc run reads it, and
  every routine pass starts from a copy of it. `verify_paper_ko.py` reports any file that drifts
  (except `MocapAligner.yaml`, whose `bodyName` is per dataset). Edit `config_base/` first, then copy
  to `~/.config/mc_rtc`, never the other way round. `stage_replay.sh` copies `configs/clean` into
  the real `~/.config`.
- In the replay the contact wrench covariance comes from the BAG, not the configuration: a
  `contact_wrench` covariance overlay is silently inert there. Test wrench trust in the routine.
- The retained tuning is also kept in `results/paper-rebuild/configs/clean/` (read by
  `variant_install.py` and `make_overlays.py`); keep it equal to `config_base/`.

## Estimator instances and variants

One routine tick runs, in `Passthrough.yaml`: VALINOR (`update: true`, the realRobot the mocap is
cross-correlated against; also the contact source of the RI-EKF plugin), then the Kinetics
Observer instances, all `update: false`, all with Schmitt thresholds 0.1 / 0.12:

| instance | paper name | variant | macro |
|---|---|---|---|
| unnamed `MCKineticsObserver` | KO | `clean` | `Kineticsobserver` |
| `KOZPC` | KO-ZPC (contact rest pose frozen) | `zpc` | `KoZpc` |
| `KOWWS` | KO-PC, "without wrench sensors" (`pinContacts: true`) | `pc` | `Kowithoutwrenchsensors` |
| `KONOANG` | `noAngularFlexibility: true`; its column is not read | — | — |

`snapshot_shared.py` maps each named instance to its own `runs/<variant>/` directory, and
`observersInfos.yaml` maps each `Observers_MainObserverPipeline_<name>_*` channel family to its
abbreviation. The RI-EKF baseline is the offline parse in `results/paper-rebuild/hartley/`, never
the in-tick plugin (one row every two iterations on LongWalk); it is rebuilt only when
`KO_REBUILD_HARTLEY_CLEAN=1` is set, which the 2026-09-18 run did.

Variants that change the shared robot or sensor configuration need their own tick
(`variant_install.py`, from a pristine copy of the retained tuning every time):
`flexdiv10` / `flexmul10` (stiffness and damping /10, x10), `hidehand` (left-hand sensor ignored,
`extForces` figure), `orierror` (30 deg rest-orientation error, seed 1, `RightFootRoll` figure),
`noangclean` = **KO-Lin** (option `noAngularFlexibility`: no angular stiffness, no angular
damping, no contact torque measurement nor process; macro `Kolinear`). Analysis-only variants
known to `variant_install.py`: `noangular` (retired: zeroed the angular law but kept the torque
measurement), `pointcontact`, `noangstiff`, `nogyrobias`, `nounmodeled`, `tightyaw`, `freezebz`,
`handinput`, `imunoise`, `noisygyro`, `gyronoise`, `gyrobias`, `gyrobias_<deg/s>[_asis]`,
`biasinit_<value>`, and the suffix `+uw<value>` (disturbance-wrench process).

**Adding a variant** = one entry in `manifest.VARIANTS` and one branch in `variant_install.py`,
nothing else; a variant is the retained tuning plus ONE deliberate change. If it must run inside
the shared tick, add a named block to `config_base/controllers/Passthrough.yaml` and its mirror
in `observersInfos.yaml`. If it needs a new observer option, it goes into
`MCKineticsObserver.cpp` (and the ROS 2 bridge for the replay): rebuild, then re-run
`verify_paper_ko.py` to see what moved.

## Things that bite

- **Every pass overwrites `Projects/<p>/output_data` in place.** A result not copied out by
  `stage_routine.sh` is gone (velocity pickles of the first disturbance-wrench sweep).
- **Never leave a variant installed.** With private HOMEs this cannot happen any more unless
  `KO_LIVE_CONFIG=1`, where `stage_routine.sh` traps TERM to reinstall `clean`. To stop a pass,
  kill `stage_routine.sh` by PID (not by name pattern), then the orphan `chain.sh` and
  `mc_rtc_ticker`.
- **Killing a replay leaves its `ros2 bag record` alive**, sometimes writing gigabytes. Check
  `pgrep -af "ros2 bag record"`.
- **`metrics.py` refuses to run while a chain is in flight** (`--force` overrides): pooling during
  a regeneration once produced a table mixing two runs.
- **Run one computation at a time.** Parallel ticks and replays corrupt shared state.
- **`_WO_LeftHand` projects re-tick byte-identical raw data to their numbered twin** (md5), hence
  they borrow the twin's RI-EKF parse; same for the orientation-error project.
- **Result labels must not start with `var-`**: `metrics.py` would take them for replay variants.
- **Duplicated messages in a replay masquerade as divergence**: check the row count of
  `kinetics.txt`.
- The unmodeled (disturbance) wrench process covariance is load-bearing (it absorbs the RHPS1
  force-sensor artefacts): do not tighten it. Its settled value is 0.09.

## Environment variables

`KO_CONFIG_HOME` (private HOME of the pass), `KO_LIVE_CONFIG` (write the real `~/.config`),
`KO_PROJECTS` (restrict a routine pass to named datasets), `KO_OBSERVERS` (tick only the named
estimators, e.g. `KO,Tilt`), `KO_DROP_PLUGINS`, `KO_REBUILD_HARTLEY_CLEAN` (rebuild the RI-EKF
parse), `KO_REAL_HOME`.

## Where the data lives (none of it is in git)

- `Projects/<dataset>/raw_data/controllerLog.bin` — the raw input; `output_data/` — everything
  derived, rewritten by the next pass; `output_data/kinetics_eval/` — the replay cache.
- `results/paper-rebuild/` — `runs/<variant>/<project>/` (`cached_rel_err.pickle`,
  `riekf_rel_err.pickle`, `*_loc_vel.pickle`, `*_traj10.npz`, the pass's `config/` and its
  `config.sha256`), `hartley/` (RI-EKF baseline), `macros/`, `configs/clean/`, `figures/`, the
  binary copies, `retick/` logs.
- `results/<label>-<hash>/` — replay runs; `results/var-*/` are summarised by `distill.py`.
- Investigation material kept out of git on purpose (listed in `.gitignore`): `analysis/`,
  `ablations/`, the `KINETICS_*.md` notes, `yawsearch/`, the `fixup_*` and overnight scripts.

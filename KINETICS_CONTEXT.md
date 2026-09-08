# Kinetics Observer — shared working brief

Common ground for anyone (human or agent) picking up this work. Rewritten 2026-09-08.
It records the goal and acceptance criteria, the two pipelines, how evaluation and the covariance
search work, the conventions that have already produced wrong conclusions, what is settled, and
what is still open.

Companion documents:
- `analysis/README.md` — the scripts, overlays, configs and logs behind everything asserted here,
  plus an index of which `results/` directories matter and what they contain. Start there to
  reproduce or check any number in this brief.
- `analysis/findings/` — one file per durable finding, with the measurement and the "why".
- `KINETICS_INVESTIGATION.md` — the older investigation log. **Stale (last edited 2026-08-28)**;
  superseded by this file and `analysis/`.
- `~/Documents/ResearchNotes/Thesis/Manuscript/Sections/Appendix/Appendix_Analyt_Jacs.tex` — the
  analytical Jacobian derivations the C++ implements. **Do not edit the .tex.**

---

## 1. The goal and the acceptance criteria

Show that **MCKineticsObserver (KO)** — a centroidal EKF fusing IMU, kinematics and contact
force/torque through a visco-elastic contact model — beats the **RI-EKF baseline** (Hartley's
invariant EKF, run as the standalone `InEkfLogParser`) across 13 mocap datasets, without giving
the KO an advantage the baseline does not get.

Everything is scored as **relative pose error (RPE)**, never ATE, and quoted as a `KO / RI-EKF`
ratio where < 1 is a win.

| criterion | threshold | where it must hold |
|---|---|---|
| `trans_xy` | <= 0.80 | RHPS1 walk and RHPS1 slippage. MultiContact and LongWalk carry no floor (their best ever is ~0.95) but are still minimised. |
| `yaw` | <= 0.95 | every category |
| slippage robustness | <= 0.90 | `trans_xy` and `yaw`, as the walk->slippage error growth compared with the RI-EKF's |
| `trans_z` | <= 1.05 | every category |
| `tilt` | <= 1.05 | every category |
| `vel_xy` | <= 1.10 | every category |
| `vel_z` | no cap | ranked lowest |

**Per category, not pooled.** A pooled average hides a category that is badly violated — that is
exactly how the RHPS1 yaw deficit stayed invisible behind LongWalk for a whole campaign.

**Fairness.** Anything the RI-EKF also consumes must change on both sides or not at all, so the
gyroscope and accelerometer noise scales are excluded from the search. The contact wrench is *not*
shared: the RI-EKF uses force only for contact detection, never in its dynamics. A model
correction that fixes a genuine physical error is fair even when it is KO-only; a tuning knob only
one side gets is not.

### Datasets (13) and their RPE sub-trajectory length

```
HRP5_MultiContact_1..4          HRP5P,  5 ms, 3 contacts (2 feet + LeftHandCloseContact)   0.3 m
HRP5P_LongWalk                  HRP5P,  4 ms, 320 m walked inside a ~6x5 m box              10 m
KO_TRO2024_RHPS1_1..5           RHPS1,  5 ms                                                 1 m
KO_TRO_2024_RHPS1_SLIPPAGE_1..3 RHPS1,  5 ms                                                 1 m
```

The per-dataset sub-length matters more than it looks: the KO's yaw advantage only appears beyond
about 2 m, so RHPS1 at 1 m and LongWalk at 10 m are not measuring the same thing. See section 6.

Per-project settings: `Projects/<name>/projectConfig.yaml` (`predefined_sublengths`,
`Body_vel_eval`, `EnabledRobot`, `EnabledBody`).

---

## 2. The two pipelines

Both start from the same recorded `raw_data/controllerLog.bin` and end in the same rpg metric
code. They differ in **how the observer is executed**, and they must agree.

### Pipeline A — "the routine" (`scripts/routine_scripts/mainRoutine.sh`)

Runs the observer **inside mc_rtc**, as a live observer. This is the reference behaviour and the
source of the inputs pipeline B replays.

```
raw_data/controllerLog.bin
  |- mc_rtc_ticker --no-sync --replay-outputs -e -l controllerLog.bin
  |    Passthrough controller; ObserverPipelines instantiates MCKineticsObserver alongside
  |    MCValinor, MCKineticsObserverFG and the HartleyIEKF plugin (the RI-EKF baseline).
  |    Contact detection, kinematics and wrenches are computed live from the robot model.
  |- output_data/logReplay.bin -> logReplay.csv -> lightData.csv
  |- resampleMocapAndExtractPose.py   mocap reference
  |  crossCorrelation.py              time sync, by cross-correlating local linear velocity
  |  matchInitPose.py                 frame alignment
  |- finalDataCSV.csv     KO_position_*, Hartley_*, Mocap_*, Control_*
  |- plotAndFormatResults.py -> output_data/evals/<Observer>/stamped_traj_estimate.txt
  |                          -> output_data/formattedMocap_Traj.txt   (ground truth)
  |- computeMetrics.sh -> rpg analyze_trajectory_single.py
                       -> evals/<Observer>/saved_results/*/relative_error_statistics_<sub>.yaml
```

### Pipeline B — "the replay" (`scripts/kinetics_eval.py`)

Runs the **bare `KineticsObserver` C++ class** standalone over ROS 2, fed the inputs A recorded.
This is the fast tuning loop and the only thing the search uses.

```
prepare:
  raw_data/controllerLog.bin
    |- mc_rtc_ticker again -> kinetics_eval/logReplay_full.bin
    |    Needed because the routine lightens logReplay.bin, stripping MEKF_* debug channels.
    |    This file is the conversion INPUT and never changes when you re-run the observer.
    |- bin_to_rosbag_kinetics.py  (+ kinetics_bag_common.py)
    |    reads MEKF_inputs_*, MEKF_measurements_*, debug_contactKine_*, MEKF_initialState_*
    |    translates the mc_rtc YAML into a KineticsConfiguration message
    |    writes a ROS 2 bag: KineticsConfiguration (once) + KineticsInput (per tick)
    |- kinetics_eval/reference/{mocap,riekf,mocap_velocity,riekf_velocity}.txt  from finalDataCSV
run:
  make_kinetics_config_bag.py     applies the covariance overlay as a separate config bag
  test_kinetics_replay.launch.py -> kinetics_offline_replay node
    |- kinetics.txt / kinetics_centroid.txt / kinetics_velocity.txt
    |- analyze() -> same rpg code vs reference/mocap.txt -> summary.csv
```

|  | A (routine) | B (replay) |
|---|---|---|
| observer runs | inside mc_rtc | standalone node, bare estimator |
| kinematics / wrenches | computed live | replayed verbatim from A's log |
| contact detection | Schmitt trigger, live (>=10% of body weight) | replays the `isSet` flag A recorded |
| cost (LongWalk) | full re-tick, 30 min+ | 8.6 min prepare + 7 min run |
| cost (other 12) | — | ~13 s each, 159 s to prepare all twelve |
| covariance changes | edit live mc_rtc YAML | overlay file, mc_rtc untouched |

**Parity is verified**, not assumed: on HRP5_MultiContact_4 with both sides built from one
revision, position agrees to 0.0137 mm mean / 0.112 mm max and the rpg metrics to 0.04%. If they
ever disagree, suspect build skew first — rebuild *both* sides before drawing any conclusion.

### What lives where

- `summary.csv` — one row per (project, estimator, distance, metric, statistic). Metrics are
  `trans_xy`, `trans_z`, `yaw`, `tilt`, `trans`, `trans_perc`, `rot`; statistics include `mean`,
  `median`, `rmse`, `std`, quartiles. **The search reads `mean`.** There is no ATE anywhere.
- `scripts/.ko_cache.json` — the RI-EKF's and the installed tuning's own error per dataset and
  metric, built by `ko.reference()` from `INSTALLED_RUN`. Also read from `mean`. Rebuild it after
  changing which statistic is used, or the slippage term mixes statistics.

### Useful commands

```bash
# prepare (mandatory after ANY observer-config change - see trap 9)
.venv/bin/python scripts/kinetics_eval.py --projects <NAMES> prepare --force

# evaluate with an overlay, optionally against a different observer config tree
.venv/bin/python scripts/kinetics_eval.py --projects <NAMES> \
    --covariance-overlay <file>.yaml [--observer-config <tree>/MCKineticsObserver.yaml] \
    run --label <tag> --no-plots --no-latest --no-open
```

Overlay schema (`merge_covariance_overlay`): `covariances` (absolute), `covariance_scales`
(multiplicative, for per-robot quantities like `contact_wrench`), `contact_model`,
`contact_model_scales`, `*_per_contact`, `per_robot`, and `settings` (top-level scalar switches).

---

## 3. The covariance search (`scripts/kinetics_tune.py`)

CMA-ES over log-scale dimensions, anchored on the installed mc_rtc config.

```bash
.venv/bin/python scripts/kinetics_tune.py \
  --workers 16 --popsize 16 --screen-trials 0 --full-trials 256 --domain-base 20 \
  --refine-sampler cmaes --tracking results/<name> --only <dim> <dim> ...
```

- **`--screen-trials 0` is mandatory.** `SCREEN_PROJECTS` is referenced at the phase-A call site
  but defined nowhere, so any non-zero value raises `NameError` before a single trial runs. The
  two-stage screen was abandoned anyway: it twice produced points that screened clean and failed
  on all thirteen, because `trans_z` is invisible on short sub-trajectories.
- **`--popsize` decouples the generation size from the worker count** (0 keeps the historical
  2x workers). CMA-ES needs >= 4+3*ln(n) samples per generation; at 17 dimensions that is 12.5, so
  `popsize == workers` clears it and makes a generation exactly one wave with no idle worker.
- Cost: ~13 s per dataset idle, ~2.1x that under 16-way contention. 13 datasets = ~1360 s/trial,
  so 256 trials at 16 workers is ~6 h. Excluding LongWalk cuts it to ~2.5 h.

### The objective

Lower is better; 0 is parity. Every term is computed on `ln(KO/RI-EKF)` of the **mean** RPE.

```
weighted mean   trans_xy x3, yaw x3, trans_z x1, tilt x1, vel_xy x0.75, vel_z x0.40
                a metric above parity is charged again (HARD_REGRESSION_PENALTY = 2.0)
soft worst      0.15 weight, beta 6      catches divergence without dictating the ranking
datasets lost   0.60 x weighted fraction of cases above parity   (breadth counts)
slippage        3.0 x penalty, target 0.90, on trans_xy and yaw
caps            1.0 x, per category: vel_xy <= 1.10, tilt <= 1.05, trans_z <= 1.05
floors          1.0 x, per category: trans_xy <= 0.80 (RHPS1 only), yaw <= 0.95
failure         50.0
```

Two properties exist specifically because the search exploited their absence:

1. **The slippage denominator is capped at the installed tuning's nominal error.** Degradation is
   `slip / nominal`, so inflating your own nominal improves it with no robustness whatsoever. A
   256-trial search found exactly that: it scored 0.888 on yaw and cleared the 0.90 target purely
   by degrading the nominal walk 1.027 -> 1.120 while its slippage error was unchanged. Crossing
   that threshold was worth 2.49 of its 2.69 apparent advantage.
2. **Floor credit is awarded only once every floor is met**, and `trans_xy` earns none at all
   below its floor. Summing credits let `trans_xy` at 0.552 pay down a yaw miss at 1.120. Beating
   a floor is a requirement, not a currency.

With both closed, that winner's advantage fell from 2.69 to 0.045. **Any objective value from
before 2026-09-08 is not comparable** — the statistic changed from median to mean, the barriers
became per-category, and the floors became two-sided.

---

## 4. Conventions and traps (each has already produced a wrong conclusion)

1. **mc_rtc logs at the END of an iteration.** Row `i` holds the inputs used in iteration `i` and
   the state produced by it. `MEKF_initialState_*` is a separate constant channel holding the true
   pre-run state; that is what the converter restores.
2. **The logger stores INVERSE quaternions.** `MEKF_estimatedState_ori`,
   `debug_contactKine_*_orientation`, `MocapAligner_worldBodyKine_ori`. Comparing without
   inverting gives ~76 deg of pure artefact.
3. **`MEKF_estimatedState_position` is the CENTROID**, not the floating base (`mcko_fb_posW_*`).
   They differ by ~200 mm.
4. **`kinetics.txt` already has the time offset applied.** Do not subtract `time_offset.json`
   again.
5. **`logReplay_full.bin` is the conversion INPUT, not the replay output.** It never changes when
   you re-run the observer. Verifying a config change by reading it will always show "no change".
   The replay's own state is in the output bag (`standalone_replay`), which `--no-plots` deletes.
6. **Frames:** contact forces in the state are in the CONTACT frame and are rotated into the
   centroid frame by `contact.centroidContactKine.orientation` (the *input* kinematics, logged as
   `debug_contactKine_*_inputCentroidContactKine_orientation`) — not by the state's rest
   orientation. Using the wrong rotation changes a force sum from 24 N to 147 N.
7. **A run is corrupt if `kinetics.txt` has the wrong row count** or any non-monotonic timestamp;
   duplicated messages masquerade as filter divergence. Expected rows: LongWalk 1474096, MC_1
   11151, MC_2 10249, MC_3 10581, MC_4 10438, RHPS1_1 81792, _2 104897, _3 80813, _4 97257,
   _5 96021, SLIPPAGE_1 20800, _2 22600, _3 21200.
8. **Concurrent runs need distinct `ROS_DOMAIN_ID`s, and a hung replay poisons its domain.** A
   dead publisher keeps feeding the next trial on that domain another dataset's inputs until it
   also times out; they pile up and the domain is dead for the rest of the run. Measured: 20 of
   256 trials lost, every failure on the same three domains. `DOMAIN_CYCLE = 60` now gives a
   domain a long rest before reuse. A sweeper that kills replay processes older than ~900 s (twice
   the median trial) is worth running alongside any long search.
9. **Any observer-config change invalidates every prepared bag.** The contact configuration is
   baked into the bag at conversion time, so `prepare --force` is mandatory — otherwise every
   trial fails in 0 s with "wrapper changed". 159 s for twelve, 679 s including LongWalk.
10. **`cmake --build . --target install` on state-observation runs `cmake_uninstall.cmake` first.**
    If `CMAKE_INSTALL_PREFIX` is wrong it deletes the installed library and then fails. It must be
    `/home/arnaud/devel/install`. `build/` is a symlink to `/home/arnaud/devel/build/state-observation`.
11. **Everything now builds RelWithDebInfo** (state-observation and both ROS 2 packages). Debug
    was 6x slower. `BOOST_ASSERT`/`CheckNaN` are therefore NOT live — they caught two real bugs in
    the past, so keep a Debug tree available when modifying the observer.
12. **Contact IDs differ between pipelines.** mc_rtc orders LeftFootCenter=0, RightFootCenter=1;
    the converter assigns from the YAML list order (Right=0). Harmless for the trajectory but the
    slots do not mean what they say.
13. **Stale duplicate install trees** exist under `catkin_ws/src/install/` and
    `catkin_ws/src/state_observation_ros2/install/`. The live one is `catkin_ws/install/`.
14. **`pgrep -f` matches your own shell.** A guard like `until ! pgrep -f kinetics_tune.py` never
    terminates, because the waiting shell's command line contains that string. Cost 3 h of idle
    machine once. Bracketing the pattern does not help if the real path appears in the same line.

---

## 5. Settled — do not re-investigate

- **RHPS1 foot force sensors carry an 18-19 deg calibration tilt.** Each foot reports a
  load-proportional tangential force of ~0.29x the normal load, fixed in direction per foot,
  consistent to 1 deg across all eight RHPS1 datasets, giving 147 N of net horizontal force while
  the robot stands still. Confirmed against mocap attitude (not just the estimator's), and HRP5-P
  is 10x cleaner. Correcting the FORCE only (the torque's centre of pressure is already inside the
  sole) improves the estimated perturbation wrench 4.4x on force and 7.8x on torque against a
  known truth of zero, matching the ~10 N/N.m the paper reports for HRP5-P. Configured as
  `wrenchCalibration` in `MCKineticsObserver/rhps1.yaml`.
- **Identified sensor noise** (unloaded noise floor, 0.2 s windows, 10th percentile):
  RHPS1 foot force sigma 1.08 N -> 1e0, torque sigma 0.029 N.m -> 9e-4 (was 1e-2, 12x too loose);
  HRP5-P foot force sigma 0.67 N -> 5e-1, torque 9e-4; HRP5-P hand 1e-2 / 2.5e-5 (it inherited the
  foot values, 100x and 36x too loose). Note a foot is only ever unloaded mid-swing, so its floor
  still carries some inertial content; the hand is genuinely at rest.
- **The unmodeled wrench must keep its slack.** Tightening `unmodeledForceProcessVariance` from
  0.09 degrades everything monotonically (RHPS1 trans_xy 0.629 -> 1.601 at x100, -> 2.120 at
  x10000). It is load-bearing, and it is NOT motion-dependent (135 N standing still vs 153 N
  walking), so a large unmodeled force is not evidence of a bad dynamic model.
- **HRP5-P linear stiffness is 3e5 and always has been** (git, since Feb 2026). The
  `# [4e4, 4e4, 4e4]` comment beside it is boilerplate present in every robot file, not a former
  value. 4e4 makes LongWalk clearly worse (trans_z 2.064 -> 3.058).
- **A Coulomb limit with return mapping on the contact rest pose breaks the EKF.** Capping the
  tangential force at mu*f_n and advancing the rest position to match degrades monotonically
  (trans_xy slip degradation 1.005 -> 1.074 -> 1.126) and diverges outright under finite
  differences (trans_xy 819). The return mapping moves the state discontinuously while the
  covariance propagates as if the dynamics were smooth. Capping WITHOUT the return mapping is
  worse than useless: it makes the prediction agree with the saturated measurement and removes the
  innovation that corrects the rest pose.
- **Jacobians verified.** All appendix Jacobians check against finite differences except one typo:
  `eq:jac_vec_RRR` ends `R^T R0^T`, must be `R0^T R^T`. Not fixed in the .tex (owner's choice).
  `computeAMatrix`/`computeCMatrix` agree with the library's own FD path to 0.166 um over 10438
  iterations. The factor graph's `ContactMomentFactor::JTr` had the same typo, fixed 2026-09-02;
  the FG has zero `numericalDerivative` calls, which is why it went unnoticed.
- **The mocap lever-arm bug is fixed** (`resampleMocapAndExtractPose.py:324`, the offset was
  rotated by a constant instead of the body's world orientation). It changed LongWalk's ground
  truth path length by 14%, so pre-2026-09-02 numbers are not comparable.

---

## 6. Open questions

**RHPS1 yaw (~1.05-1.08 against the RI-EKF's 0.317 deg) has no working explanation.** Levers
measured and found inert: the tau_z measurement variance at 1e6x; the contact yaw process
covariance (owner's own ablations); RHPS1 angular stiffness over a 50x sweep (727 is already the
optimum, and the response is flat — 0.339 at best against 0.340 installed); and 218 CMA-ES trials
that never got RHPS1-walk yaw below 1.029. A measured correlate exists — RHPS1's foot yaw-torque
residual has a p90 of 8.5-11.9 N.m against HRP5-P's 3.1-4.4, confined to the yaw axis while
`|dTxy|` is identical between robots — and the natural reading is that the visco-elastic model
treats a ground friction moment as elastic compliance. But the prediction that follows from it
(spurious yaw scales as tau_z / K_ang, so tripling K_ang should cut it threefold) **failed**: yaw
moved 0.3%. So the correlate is real and the mechanism is not established.

**Velocity is untouched by anything.** `vel_xy` sits at 1.24-1.51 in every category, 0/13 datasets
won, unchanged across ~1400 historical evaluations and 460 more this week. The decomposition puts
the KO-specific velocity error at 28.2 mm/s on the RHPS1 walk against the RI-EKF's *entire* 23.2,
with only 0.40 correlation between them, so it is the KO's own failure mode and not shared
difficulty. No mechanism identified.

**Slippage robustness is not reachable by covariances.** Best genuine (non-gameable) degradation
across 218 trials is 0.985 against a 0.90 target. The RI-EKF absorbs slip through its own contact
random walk (`contactVariance` 1e-4, sigma 1 cm/step), so there is no easy asymmetry to exploit;
the KO's unique asset is that it sees the contact force, but force saturation above 0.4 occurs in
only ~2% of samples per sub-trajectory and does not correlate with where the KO loses ground.

**The vertical RPE deficit on RHPS1 is a start-up transient.** Sub-trajectories grouped by
position in the recording: the KO is 3.5 mm worse in the first 20% and *better* by the 60-80% mark
(7.84 vs 9.24), because the RI-EKF's vertical error climbs steadily (2.4 -> 9.9 mm) while the KO's
is nearly flat (5.9 -> 9.9). That is why the KO wins vertical ATE (0.670) while losing vertical
RPE. It points at the INITIAL covariances (`contact_initial`, `statePositionInitVariance`), a
family the recent searches did not touch. Yaw shows the opposite pattern — the KO is better for
the first 60% and degrades after — so yaw is not an initialisation problem.

**LongWalk is the canary and must stay in the objective.** The previous campaign, which scored
only 12 datasets, silently regressed it from 0.991 to 1.048 on `trans_xy` and 1.308 to 2.091 on
`trans_z`. Putting it back in recovered both (0.946 / 1.119) at the cost of RHPS1 translation
falling to exactly the 0.80 floor. That trade runs through `contact_process_position_xy`: 3e-6
buys RHPS1 slip robustness and wrecks LongWalk vertical; 1e-9 is the reverse; the search settles
near 3e-8.

**`VELOCITY_FILTER` hardcodes 200 Hz** (`kinetics_eval.py:389`). Twelve datasets are 200 Hz, but
LongWalk is 250 Hz and so has its mocap reference filtered at 18.8 Hz instead of 15. Real bug,
small, LongWalk-only — it cannot explain a velocity deficit present on the twelve where the filter
is exactly right.

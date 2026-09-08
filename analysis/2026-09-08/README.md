# Working files from the 2026-09-07/08 session

Copied out of `/tmp`, which is cleared on boot. Nothing here is required by the pipeline; it is
the material behind the numbers quoted in `KINETICS_CONTEXT.md`.

## configs/
- `tuned_a1_posx30.yaml` — the covariance overlay in use before this session. Every "previous
  tuning" comparison refers to it, and it existed nowhere else.
- `calibrated-observer-config/` — a full MCKineticsObserver config tree carrying the RHPS1
  `wrenchCalibration` block, used with `--observer-config` for opt-in A/B before the calibration
  was installed.
- `kang*.yaml`, `flex_*.yaml` — the ten single-parameter contact-model variants swept on
  2026-09-08 (angular stiffness 100/300/2000/5000, angular damping 5/60, linear stiffness
  1e4/1e5, linear damping 40/600).

## scripts/
Analysis run against the mc_rtc logs and the evaluation output. The ones worth keeping:

| script | what it establishes |
|---|---|
| `sensfloor.py` | sensor noise floor from unloaded windows — the source of the identified covariances |
| `calib4.py`, `calib3.py` | the per-foot wrench rotation, from single support where the GRF must be vertical |
| `balance2.py`, `swing.py`, `tilt.py` | the 147 N standstill imbalance, load-proportionality, and that the contact frame is flat |
| `extw_all.py` | perturbation wrench against a known truth of zero, per dataset |
| `rpecorr.py` | RPE harness that reproduces the benchmark to three digits (0.595 vs 0.596) |
| `ate.py` | posyaw-aligned ATE — note this is my own alignment, not rpg's |
| `transient.py` | RPE split by position in the recording; shows the vertical deficit is a start-up transient |
| `yawdist.py` | yaw RPE vs sub-trajectory length, both robots |
| `cone.py`, `compete.py` | friction-cone violation of the visco-elastic model, and who absorbs the innovation |
| `feas.py`, `degrad.py` | per-category criteria check and slippage degradation |
| `objtest.py`, `sens.py` | objective decomposition and per-metric marginal value |

Several read `logReplay_full.bin` directly and take 1-2 minutes per dataset; LongWalk can exhaust
memory, so exclude it or run it alone.

## backups/
Pre-session copies of `kinetics_tune.py` and `ko.py` (both untracked in git when edited, so this
is the only record of the objective before the mean/median switch and the per-category barriers),
and of `KINETICS_CONTEXT.md` before its rewrite.

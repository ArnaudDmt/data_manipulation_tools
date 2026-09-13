# Rebuilding the paper's measured content

Everything the IJRR paper prints as a number or a data figure is produced from here.

    scripts/paper_pipeline/run.sh                     # all four stages, in order
    scripts/paper_pipeline/run.sh routine hidehand    # one stage, one variant
    scripts/paper_pipeline/run.sh figures poseAndVel  # one stage, one figure

`manifest.py` is the single source of truth: datasets, categories and their sub-trajectory
lengths, observer variants, and which script produces which figure. Adding a dataset or a variant
means editing that file and nothing else.

## Stages

| stage | what it runs | what it leaves |
|---|---|---|
| `replay` | `kinetics_eval.py` per variant over the 13 datasets | `results/var-<label>-<hash>/` — the relative errors |
| `routine` | the mc_rtc chain per variant | `results/paper-rebuild/runs/<variant>/<project>/` — velocity pickles, RPE caches, disturbance-wrench logs; and `Projects/*/output_data`, which every figure reads |
| `metrics` | pools all three families | `results/paper-rebuild/macros/*.tex`, folded into the paper's `metrics_results.tex` |
| `figures` | every figure script | the figures next to `main.tex`, plus PNG **and** PDF of all of them in `figures-export/` |

Stages are independently runnable and each leaves its snapshots behind, so a failed stage can be
rerun without redoing the ones before it.

## Things that bite

- **Every pass overwrites `Projects/<p>/output_data` in place.** A result not copied out by
  `stage_routine.sh` is gone. This is how the velocity pickles were lost from the first
  disturbance-wrench sweep.
- **`metrics.py` refuses to run while a chain is in flight** (`--force` overrides). Pooling the
  RI-EKF while its caches were being regenerated trial by trial once produced a table that mixed
  two runs, and nothing in the output said so.
- **The RI-EKF baseline is the offline parse** in `results/paper-rebuild/hartley/`, never the
  in-tick plugin, which emits one row every two iterations on the 500 Hz LongWalk log. It does not
  depend on the Kinetics Observer's tuning, so it is a fixed reference and is not regenerated.
- **The KO-ZPC curve needs a second observer instance** in the controller pipeline. `chain.sh`
  installs it for `manifest.NEEDS_KOZPC` only and restores the plain controller on exit, including
  on failure. Two instances re-register their logger keys every iteration: harmless over 11k
  iterations, a 37.8 GB runaway on LongWalk.
- **Never leave a variant installed.** `stage_routine.sh` reinstalls `clean` at the end; a
  leftover 30 deg orientation error or a hidden hand silently contaminates every later tick.
- **`_WO_LeftHand` projects re-tick byte-identical raw data to their numbered twin** (verified by
  md5), which is why they borrow the twin's RI-EKF parse. Same for the orientation-error project
  and `HRP5_MultiContact_1`.

## Not automated

- `compute_time.pdf` — no producing script exists; the figure is carried through unchanged.
- `slipping-odom-traj.pdf` — its caption promises an inset the figure does not have; to be added
  by hand. See `TODO-figures.md` next to `main.tex`.
- The VALINOR macros in `metrics_results.tex` are not remeasured by any stage here.

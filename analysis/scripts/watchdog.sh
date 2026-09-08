#!/bin/bash
# Keeps the search alive without supervision.
#
# A trial that dies is already handled by the tuner (failed trial, penalty score, generation
# closes). This covers the case the tuner cannot: the tuner itself going away -- an OOM kill, a
# crash, or the machine rebooting. Completed trials live in trials.csv, so a relaunch recycles
# every evaluation and only loses the sampler's learned state.
TRACK=/home/arnaud/devel/src/data_manipulation_tools/results/kinetics-yaw-20260904
SCRIPTS=/home/arnaud/devel/src/data_manipulation_tools/scripts
PY=/home/arnaud/devel/src/data_manipulation_tools/.venv/bin/python
LOG=/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/287daf9e-ad1c-4931-a07e-923142412c80/scratchpad/search-yaw.log
TARGET=320
restarts=0

while true; do
  sleep 120
  # done?
  if [ -f "$TRACK/report.md" ]; then echo "WATCHDOG: search completed, report written"; break; fi
  done_count=$( [ -f "$TRACK/trials.csv" ] && echo $(( $(wc -l < "$TRACK/trials.csv") - 1 )) || echo 0 )
  [ "$done_count" -ge "$TARGET" ] && { echo "WATCHDOG: $done_count trials reached"; break; }

  # Match the python process specifically. Grepping the tracking-dir name matches this script's
  # own command line, so the watchdog was confirming its own existence and never fired.
  if ! pgrep -f "python.*kinetics_tune\.py --tracking" >/dev/null 2>&1; then
    if [ "$restarts" -ge 5 ]; then
      echo "WATCHDOG: tuner died and 5 restarts already used -- stopping, needs a human"
      break
    fi
    restarts=$(( restarts + 1 ))
    echo "WATCHDOG: tuner is gone at $done_count/$TARGET trials -- relaunch $restarts of 5"
    # clean whatever the dead run left behind so the disk does not fill
    for p in $(ps -eo pid,comm --no-headers | awk '$2=="ros2"{print $1}'); do
      g=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' '); [ -n "$g" ] && kill -9 -"$g" 2>/dev/null
    done
    sleep 3
    rm -rf /home/arnaud/devel/src/data_manipulation_tools/results/fin-*
    # keep the ledger to recycle from, start a fresh study beside it
    mv "$TRACK" "$TRACK-part$restarts" 2>/dev/null
    remaining=$(( TARGET - done_count ))
    seeds=""
    for d in "$TRACK"-part*; do [ -f "$d/trials.csv" ] && seeds="$seeds --seed-from $d/trials.csv"; done
    ( cd "$SCRIPTS" && nohup "$PY" kinetics_tune.py --tracking "$TRACK" --prefix yaw --workers 16 \
        --screen-trials 0 --full-trials "$remaining" --refine-sampler cmaes --timeout 1800 \
        $seeds >> "$LOG" 2>&1 & )
    sleep 60
    pgrep -f "python.*kinetics_tune\.py --tracking" >/dev/null 2>&1 \
      && echo "WATCHDOG: relaunched with $remaining trials remaining, seeded from $done_count evaluations" \
      || echo "WATCHDOG: RELAUNCH FAILED -- needs a human"
  fi
done

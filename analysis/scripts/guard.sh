#!/bin/bash
# Overnight guard. Three failure modes seen in this project, each with a specific response:
#   orphaned recorders   -- 22 of them once held 130 GB of deleted files and filled the disk
#   memory exhaustion    -- 16 workers swapped 17 GB in two minutes and the machine locked up
#   search process death -- optuna resumes from study.db, so restarting is safe and loses nothing
ROOT=/home/arnaud/devel/src/data_manipulation_tools
SP="$(dirname "$0")"
TRACK=$ROOT/results/kinetics-cell-20260905
LOG="$SP/guard.log"
RESTARTS=0
say(){ echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
say "guard started"
while true; do
  sleep 120
  # 1. reap recorders whose parent evaluation is gone (reparented to init)
  for pid in $(pgrep -f 'bag rec[o]rd' 2>/dev/null); do
    ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ "$ppid" = "1" ] && { kill -9 "$pid" 2>/dev/null; say "reaped orphan recorder $pid"; }
  done
  # 2. disk
  avail=$(df --output=avail -BG "$ROOT" | tail -1 | tr -dc 0-9)
  if [ "${avail:-999}" -lt 25 ]; then
    find /dev/shm -maxdepth 1 \( -name 'fastrtps_*' -o -name '*_el' \) -delete 2>/dev/null
    say "WARNING disk ${avail}G -- cleared shm"
  fi
  # 3. memory: only act when both RAM and swap are nearly gone
  memav=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  swtot=$(awk '/SwapTotal/{print $2}' /proc/meminfo); swfree=$(awk '/SwapFree/{print $2}' /proc/meminfo)
  swpct=$(( swtot>0 ? (swtot-swfree)*100/swtot : 0 ))
  if [ "${memav:-99}" -lt 2 ] && [ "$swpct" -gt 90 ]; then
    victim=$(pgrep -f 'kinetics_ev[a]l.py' | tail -1)
    [ -n "$victim" ] && { kill -- -"$(ps -o pgid= -p "$victim" | tr -d ' ')" 2>/dev/null; \
                          say "CRITICAL mem ${memav}G swap ${swpct}% -- killed trial group of $victim"; }
  fi
  # 4. liveness. study.db resumes, so a restart continues rather than starting over.
  if ! pgrep -f 'kinetics_tu[n]e' >/dev/null; then
    done=$(( $(wc -l < "$TRACK/trials.csv" 2>/dev/null || echo 1) - 1 ))
    if [ "$done" -ge 240 ]; then say "search finished with $done trials"; break; fi
    if [ "$RESTARTS" -ge 3 ]; then say "search dead at $done trials; 3 restarts used, giving up"; break; fi
    RESTARTS=$((RESTARTS+1))
    say "search died at $done trials -- restart $RESTARTS/3"
    cd "$ROOT/scripts" || break
    SEEDS=""
    for d in "$ROOT"/results/kinetics-*/; do [ -f "$d/trials.csv" ] && SEEDS="$SEEDS --seed-from $d/trials.csv"; done
    nohup ../.venv/bin/python kinetics_tune.py --tracking "$TRACK" --prefix cel --workers 12 \
      --screen-trials 0 --full-trials 240 --refine-sampler cmaes --timeout 1800 \
      --seed-count 12 $SEEDS >> "$SP/search-cell.log" 2>&1 &
    sleep 60
  fi
done
say "guard exiting"

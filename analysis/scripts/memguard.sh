#!/bin/bash
# Graceful degradation under memory pressure.
#
# The kernel's OOM killer picks the largest consumer, which can be anything, and a hard OOM on
# this machine has already taken the whole desktop down once. This instead sheds one trial before
# it gets that far: the tuner records a killed trial as failed with a penalty score, tells the
# sampler so the generation still closes, and keeps going. Cost is one trial out of 300.
#
# It also biases the kernel's own choice, in case pressure arrives faster than this can react:
# ROS nodes get a high oom_score_adj so they are picked before the tuner, whose death would end
# the run. Raising an adjustment needs no privileges; lowering one would.
SHED_BELOW_GB=5           # MemAvailable at which we drop a trial
RESUME_ABOVE_GB=9         # and stop shedding once it recovers
LOG=/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/287daf9e-ad1c-4931-a07e-923142412c80/scratchpad/memguard.log
shed=0

avail_gb() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }

while true; do
  pgrep -f "python.*kinetics_tune\.py --tracking" >/dev/null 2>&1 || { echo "MEMGUARD: search gone, stopping"; break; }

  # make ROS nodes the kernel's preferred victims, never the tuner
  for p in $(pgrep -f "kinetics_observer_node|rosbag_publish_kinetics|ros2 bag record" 2>/dev/null); do
    echo 600 > /proc/$p/oom_score_adj 2>/dev/null
  done
  for p in $(pgrep -f "kinetics_tune.py --tracking" 2>/dev/null); do
    echo 0 > /proc/$p/oom_score_adj 2>/dev/null
  done

  a=$(avail_gb)
  if [ "$a" -lt "$SHED_BELOW_GB" ]; then
    # newest in-flight trial: least work lost
    newest=$(ls -dt /home/arnaud/devel/src/data_manipulation_tools/results/fin-* 2>/dev/null | head -1)
    label=$(basename "$newest" 2>/dev/null)
    victim=$(pgrep -f "run --label ${label%-*}" 2>/dev/null | head -1)
    [ -z "$victim" ] && victim=$(pgrep -f "scripts/kinetics_eval.py" | tail -1)
    if [ -n "$victim" ]; then
      g=$(ps -o pgid= -p "$victim" 2>/dev/null | tr -d ' ')
      [ -n "$g" ] && kill -9 -"$g" 2>/dev/null
      shed=$((shed+1))
      echo "MEMGUARD: MemAvailable ${a}GB -- shed trial $label (pgid $g). total shed: $shed" | tee -a "$LOG"
      sleep 60
    fi
  fi
  sleep 20
done

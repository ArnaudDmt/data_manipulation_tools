#!/bin/bash
SP="$(dirname "$0")"
while pgrep -f 'lwful[l]' >/dev/null; do sleep 30; done
"$SP/tuned.sh" >> "$SP/tuned.log" 2>&1

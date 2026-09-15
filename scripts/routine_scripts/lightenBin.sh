#!/bin/bash

# Extracts a subset of keys from an mc_rtc binary log without exhausting the memory.
#
# "mc_bin_utils extract --keys" builds a FlatLog of the *whole* input (every key, every
# iteration, decoded in memory) before writing anything, which takes tens of GB on the
# multi-GB replay logs. Splitting the log first is a streaming operation, so extracting
# chunk by chunk keeps the peak memory to roughly what one chunk costs. The lightened
# chunks are then concatenated back into a single log.
#
# Usage: lightenBin.sh <input.bin> <output.bin> <key> [key...]
# The output may be the input: it is only overwritten once everything else is done.

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $(basename "$0") <input.bin> <output.bin> <key> [key...]" >&2
    exit 1
fi

input=$1
output=$2
shift 2
keys=("$@")

if [ ! -f "$input" ]; then
    echo "$input does not exist." >&2
    exit 1
fi

# Logs bigger than this are split before the extraction. One chunk of this size costs a
# few GB of RAM, lower it if the machine has less memory to spare.
chunkBytes=${LIGHTEN_BIN_CHUNK_BYTES:-1610612736}  # 1.5 GB

tmpDir=$(mktemp -d "$(dirname "$output")/.lightenBin.XXXXXX")
trap 'rm -rf "$tmpDir"' EXIT

# Lists the logs produced by one "extract --keys" call. It writes <template>.bin, plus
# <template>_2.bin, <template>_3.bin... if the set of logged keys changes along the log.
extractedLogs() {
    local template=$1
    if [ -f "${template}.bin" ]; then
        echo "${template}.bin"
    fi
    ls -1 "${template}"_*.bin 2>/dev/null | sort -V || true
}

pieces=()

inputSize=$(stat -Lc%s "$input")
if (( inputSize <= chunkBytes )); then
    mc_bin_utils extract "$input" "$tmpDir/light" --keys "${keys[@]}"
    while IFS= read -r piece; do pieces+=("$piece"); done < <(extractedLogs "$tmpDir/light")
else
    parts=$(( (inputSize + chunkBytes - 1) / chunkBytes ))
    echo "The log weighs $(( inputSize / 1024 / 1024 )) MB, splitting it into $parts parts before the extraction."
    mc_bin_utils split "$input" "$tmpDir/part" "$parts"

    for part in "$tmpDir"/part_*.bin; do
        template="${part%.bin}_light"
        echo "Extracting the keys from $(basename "$part")."
        # A part holding none of the keys is not an error: the observers may simply not
        # have been running yet over that portion of the log.
        mc_bin_utils extract "$part" "$template" --keys "${keys[@]}" \
            || echo "None of the keys were found in $(basename "$part"), skipping this part." >&2
        rm -f "$part"   # frees the disk as we go, the parts are as heavy as the input
        while IFS= read -r piece; do pieces+=("$piece"); done < <(extractedLogs "$template")
    done
fi

if [ ${#pieces[@]} -eq 0 ]; then
    echo "No data could be extracted from $input." >&2
    exit 1
fi

merged="$tmpDir/merged.bin"
if [ ${#pieces[@]} -eq 1 ]; then
    mv "${pieces[0]}" "$merged"
else
    "$(dirname "$0")/mergeBinLogs.py" "$merged" "${pieces[@]}"
    rm -f "${pieces[@]}"
fi

# Makes sure the merged log is readable before it possibly overwrites the input.
mc_bin_utils show "$merged" > /dev/null

mv "$merged" "$output"

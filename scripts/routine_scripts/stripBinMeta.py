#!/usr/bin/env python3
"""Removes from an mc_rtc binary log the metadata that breaks mc_rtc_ticker.

The metadata (timestep, main robot, its module parameters, the initial floating base,
the initial mbc().q of every robot, and the force sensor calibrations) is written as a
"start event" in the events of the first entry of a log.

mc_rtc_ticker prefers the recorded init_q over the logged qIn to initialise the robots,
and it indexes it by the *current* joint index in the mbc without checking its size:
replaying a log whose robot description has since gained joints segfaults in
Ticker.cpp:get_initial_encoders. Emptying init_q puts the ticker back on the qIn path,
which is sized by the reference joint order and stays valid.

Only init_q is dropped by default: the calibrations are the ones that were in effect when
the log was recorded, and the ticker applies them over the robot module's calib files
whenever the inputs are replayed (which --replay-outputs alone does NOT turn off). Losing
them silently swaps in whatever calibration is on disk today. Pass --all to drop the whole
metadata anyway, e.g. to pick up a calibration that was corrected after the recording.

This is also why the logs cut out of a bigger one with "mc_bin_utils split" replay fine:
only the first part carries the metadata.

Usage: stripBinMeta.py [--all] <input.bin> [output.bin]
Without an output, the input is rewritten in place once everything else is done.
"""

import os
import shutil
import struct
import sys
import tempfile

from mergeBinLogs import (
    MAGIC,
    SUPPORTED_VERSION,
    LogFormatError,
    array_header_length,
    pack_array_header,
    read_object,
)

START_EVENT = 3
# Index of init_q among the fields of a start event, see Logger.cpp:306-316.
INIT_Q_FIELD = 5
EMPTY_MAP = b"\x80"


def array_count(payload, i):
    tag = payload[i]
    if 0x90 <= tag <= 0x9F:
        return tag & 0x0F
    if tag == 0xDC:
        return int.from_bytes(payload[i + 1 : i + 3], "big")
    if tag == 0xDD:
        return int.from_bytes(payload[i + 1 : i + 5], "big")
    raise LogFormatError(f"The events of a log entry are not an array (tag {tag:#04x})")


def spans_of(payload, i):
    """Returns the (start, end) of every element of the array at payload[i:], and its end."""
    count = array_count(payload, i)
    i += array_header_length(payload, i)
    out = []
    for _ in range(count):
        start = i
        _, i = read_object(payload, i)
        out.append((start, i))
    return out, i


def without_init_q(payload, start, end):
    """Returns the start event at payload[start:end] with an emptied init_q."""
    fields, _ = spans_of(payload, start)
    if len(fields) <= INIT_Q_FIELD:
        return payload[start:end]  # a log written before init_q existed
    # Every other field is copied as it is, only init_q is replaced.
    kept = [EMPTY_MAP if n == INIT_Q_FIELD else payload[s:e] for n, (s, e) in enumerate(fields)]
    return pack_array_header(len(fields)) + b"".join(kept)


def strip_entry(payload, drop_all):
    """Returns the entry without its metadata, or None if it carries none."""
    if payload[0] != 0x92:
        raise LogFormatError("A log entry is not an array of two elements (events, records)")
    if payload[1] == 0xC0:  # nil events list
        return None
    events, end = spans_of(payload, 1)
    kept = []
    found = False
    for start, stop in events:
        event, _ = read_object(payload, start)
        if isinstance(event, list) and event and event[0] == START_EVENT:
            found = True
            if not drop_all:
                kept.append(without_init_q(payload, start, stop))
        else:
            kept.append(payload[start:stop])
    if not found:
        return None
    return b"\x92" + pack_array_header(len(kept)) + b"".join(kept) + payload[end:]


def strip(input_path, output, drop_all):
    with open(input_path, "rb") as log:
        magic = log.read(4)
        if len(magic) != 4 or magic[:3] != MAGIC[:3]:
            raise LogFormatError(f"{input_path} is not an mc_rtc binary log (invalid magic number)")
        version = magic[3] - MAGIC[3]  # the version is stored as magic[3] + version
        if version != SUPPORTED_VERSION:
            raise LogFormatError(f"{input_path} is in the log format version {version}, expected {SUPPORTED_VERSION}")
        output.write(magic)
        while True:
            header = log.read(8)
            if len(header) != 8:
                return False
            size = struct.unpack("<Q", header)[0]
            payload = log.read(size)
            if len(payload) != size:
                return False  # truncated entry, the log was interrupted
            entry = strip_entry(payload, drop_all)
            if entry is None:
                output.write(header)
                output.write(payload)
            else:
                output.write(struct.pack("<Q", len(entry)))
                output.write(entry)
                # The metadata is only ever written once, at the start of a log.
                shutil.copyfileobj(log, output)
                return True


def main():
    arguments = sys.argv[1:]
    drop_all = "--all" in arguments
    paths = [a for a in arguments if a != "--all"]
    if len(paths) not in (1, 2):
        print(f"Usage: {sys.argv[0]} [--all] <input.bin> [output.bin]", file=sys.stderr)
        return 1
    input_path = paths[0]
    output_path = paths[1] if len(paths) == 2 else input_path
    in_place = os.path.realpath(output_path) == os.path.realpath(input_path)
    directory = os.path.dirname(os.path.abspath(output_path))
    handle, temporary = tempfile.mkstemp(dir=directory, prefix=".stripBinMeta.", suffix=".bin")
    try:
        with os.fdopen(handle, "wb") as output:
            stripped = strip(input_path, output, drop_all)
        if not stripped and in_place:
            print(f"{input_path} carries no metadata, left untouched.")
        else:
            if not stripped:
                print(f"{input_path} carries no metadata, copied as it is.")
            os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return 0


if __name__ == "__main__":
    sys.exit(main())

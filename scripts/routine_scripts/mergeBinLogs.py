#!/usr/bin/env python3
"""Concatenates mc_rtc binary logs that were obtained by splitting a single log.

An mc_rtc log (format version 1) declares its keys through incremental "add key" /
"remove key" events rather than through a header, and the reader keeps that key list
alive from one entry to the next. Appending the raw bytes of a log to another one
therefore adds keys that are already declared, and the reader ends up with twice as many
keys as values: it crashes. This script rewrites the events of the first entry of every
appended log so that it first removes the keys inherited from the previous log. The
records themselves are copied byte for byte.

Usage: mergeBinLogs.py <output.bin> <input.bin> [input.bin...]
"""

import struct
import sys

MAGIC = b"ANNE"
SUPPORTED_VERSION = 1

ADD_KEY_EVENT = 0
REMOVE_KEY_EVENT = 1


class LogFormatError(Exception):
    pass


def read_object(buffer, i):
    """Reads the MessagePack object at buffer[i:], returns (value, index after it).

    Only the values that can appear in the events of a log are given a meaningful value,
    the others are read to be skipped over so a None is enough for them.
    """
    b = buffer[i]
    i += 1
    if b <= 0x7F or b >= 0xE0:  # positive / negative fixint
        return (b if b <= 0x7F else b - 0x100), i
    if 0xA0 <= b <= 0xBF:  # fixstr
        n = b & 0x1F
        return buffer[i : i + n].decode("utf-8"), i + n
    if 0x90 <= b <= 0x9F:  # fixarray
        return read_sequence(buffer, i, b & 0x0F)
    if 0x80 <= b <= 0x8F:  # fixmap
        return read_sequence(buffer, i, 2 * (b & 0x0F))
    if b == 0xC0:  # nil
        return None, i
    if b in (0xC2, 0xC3):  # false / true
        return b == 0xC3, i
    if b in (0xC4, 0xC5, 0xC6):  # bin 8 / 16 / 32
        width = 1 << (b - 0xC4)
        n = int.from_bytes(buffer[i : i + width], "big")
        return None, i + width + n
    if 0xC7 <= b <= 0xC9:  # ext 8 / 16 / 32
        width = 1 << (b - 0xC7)
        n = int.from_bytes(buffer[i : i + width], "big")
        return None, i + width + 1 + n
    if b in (0xCA, 0xCB):  # float 32 / 64
        n = 4 if b == 0xCA else 8
        return struct.unpack_from(">f" if b == 0xCA else ">d", buffer, i)[0], i + n
    if 0xCC <= b <= 0xCF:  # uint 8 / 16 / 32 / 64
        n = 1 << (b - 0xCC)
        return int.from_bytes(buffer[i : i + n], "big"), i + n
    if 0xD0 <= b <= 0xD3:  # int 8 / 16 / 32 / 64
        n = 1 << (b - 0xD0)
        return int.from_bytes(buffer[i : i + n], "big", signed=True), i + n
    if 0xD4 <= b <= 0xD8:  # fixext 1 / 2 / 4 / 8 / 16
        return None, i + 1 + (1 << (b - 0xD4))
    if b in (0xD9, 0xDA, 0xDB):  # str 8 / 16 / 32
        width = 1 << (b - 0xD9)
        n = int.from_bytes(buffer[i : i + width], "big")
        return buffer[i + width : i + width + n].decode("utf-8"), i + width + n
    if b in (0xDC, 0xDD):  # array 16 / 32
        width = 2 if b == 0xDC else 4
        n = int.from_bytes(buffer[i : i + width], "big")
        return read_sequence(buffer, i + width, n)
    if b in (0xDE, 0xDF):  # map 16 / 32
        width = 2 if b == 0xDE else 4
        n = int.from_bytes(buffer[i : i + width], "big")
        return read_sequence(buffer, i + width, 2 * n)
    raise LogFormatError(f"Unknown MessagePack tag {b:#04x}")


def read_sequence(buffer, i, count):
    out = []
    for _ in range(count):
        value, i = read_object(buffer, i)
        out.append(value)
    return out, i


def pack_string(value):
    data = value.encode("utf-8")
    n = len(data)
    if n < 32:
        return bytes([0xA0 | n]) + data
    if n < 256:
        return b"\xd9" + bytes([n]) + data
    if n < 65536:
        return b"\xda" + n.to_bytes(2, "big") + data
    return b"\xdb" + n.to_bytes(4, "big") + data


def pack_array_header(count):
    if count < 16:
        return bytes([0x90 | count])
    if count < 65536:
        return b"\xdc" + count.to_bytes(2, "big")
    return b"\xdd" + count.to_bytes(4, "big")


def events_of(payload):
    """Returns the events of an entry and the index at which its records start."""
    if payload[0] != 0x92:
        raise LogFormatError("A log entry is not an array of two elements (events, records)")
    return read_object(payload, 1)


def updated_keys(keys, events):
    """Applies the key events of an entry to the key list held by the reader."""
    for event in events or []:
        if not event:
            continue
        if event[0] == ADD_KEY_EVENT:
            keys.append(event[2])
        elif event[0] == REMOVE_KEY_EVENT and event[1] in keys:
            keys.remove(event[1])
    return keys


def rewrite_first_entry(payload, inherited_keys):
    """Prefixes the events of an entry with the removal of the inherited keys."""
    events, _ = events_of(payload)
    if not events:
        return payload
    removals = b"".join(b"\x92\x01" + pack_string(key) for key in inherited_keys)
    # The original events are copied as they are, only their array header is replaced.
    return (
        b"\x92"
        + pack_array_header(len(inherited_keys) + len(events))
        + removals
        + payload[1 + array_header_length(payload, 1) :]
    )


def array_header_length(payload, i):
    tag = payload[i]
    if 0x90 <= tag <= 0x9F:
        return 1
    if tag == 0xDC:
        return 3
    if tag == 0xDD:
        return 5
    raise LogFormatError(f"The events of a log entry are not an array (tag {tag:#04x})")


def entries_of(log):
    """Iterates over the (size, payload) of a log, past its magic number."""
    magic = log.read(4)
    if len(magic) != 4 or magic[:3] != MAGIC[:3]:
        raise LogFormatError(f"{log.name} is not an mc_rtc binary log (invalid magic number)")
    version = magic[3] - MAGIC[3]  # the version is stored as magic[3] + version
    if version != SUPPORTED_VERSION:
        raise LogFormatError(f"{log.name} is in the log format version {version}, expected {SUPPORTED_VERSION}")
    while True:
        header = log.read(8)
        if len(header) != 8:
            return
        size = struct.unpack("<Q", header)[0]
        payload = log.read(size)
        if len(payload) != size:
            return  # truncated entry, the log was interrupted
        yield size, payload


def merge(output_path, input_paths):
    keys = []
    with open(output_path, "wb") as out:
        for index, input_path in enumerate(input_paths):
            with open(input_path, "rb") as log:
                if index == 0:
                    out.write(MAGIC[:3] + bytes([MAGIC[3] + SUPPORTED_VERSION]))
                for entry_index, (size, payload) in enumerate(entries_of(log)):
                    if index > 0 and entry_index == 0:
                        payload = rewrite_first_entry(payload, keys)
                        size = len(payload)
                    if payload[1] != 0xC0:  # 0xc0 is a nil events list: nothing changes
                        keys = updated_keys(keys, events_of(payload)[0])
                    out.write(struct.pack("<Q", size))
                    out.write(payload)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <output.bin> <input.bin> [input.bin...]", file=sys.stderr)
        return 1
    try:
        merge(sys.argv[1], sys.argv[2:])
    except LogFormatError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

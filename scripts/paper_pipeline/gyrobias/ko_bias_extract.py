#!/usr/bin/env python3
"""Extrait le biais gyrometrique estime par le Kinetics Observer du bag de sortie du replay.

Les fichiers texte du replay (kinetics.txt et compagnie) ne portent que les poses ; le biais ne
vit que dans le message KineticsState. On relit donc le bag une fois et on le range en .npz.
"""
import sys
from pathlib import Path

import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message

TOPIC = '/kinetics_observer/estimated_state'


def read(bag):
    reader = SequentialReader()
    reader.open(StorageOptions(uri=str(bag), storage_id='mcap'), ConverterOptions('', ''))
    kind = get_message({t.name: t.type for t in reader.get_all_topics_and_types()}[TOPIC])
    times, biases = [], []
    while reader.has_next():
        _, data, _ = reader.read_next()
        message = deserialize_message(data, kind)
        times.append(message.header.stamp.sec + 1e-9 * message.header.stamp.nanosec)
        b = message.gyro_biases[0].bias
        biases.append((b.x, b.y, b.z))
    return np.array(times), np.array(biases)


if __name__ == '__main__':
    bag, destination = Path(sys.argv[1]), Path(sys.argv[2])
    t, b = read(bag)
    np.savez_compressed(destination, t=t, b=b)
    print(f'{len(t)} messages, t de {t[0]:.2f} a {t[-1]:.2f} s -> {destination}')
    print(f'biais final {np.degrees(b[-1])} deg/s')

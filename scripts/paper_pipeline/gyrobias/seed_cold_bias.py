#!/usr/bin/env python3
"""Ablation B : cache froid dont l'etat initial reprend le BIAIS GYRO de la course normale a t=0.

Usage : seed_cold_bias.py <projet> <run normal avec bag>
Copie kinetics_eval_cold en kinetics_eval_coldbias ; seule la partie biais (13:16) de initial_state change.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_cold_state import ROOT, read  # noqa: E402


def main(project, warm_run):
    times, states = read(warm_run, project)
    bias = states[np.abs(times).argmin(), 13:16]
    source = ROOT / f'Projects/{project}/output_data/kinetics_eval_cold'
    target = ROOT / f'Projects/{project}/output_data/kinetics_eval_coldbias'
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir()
    for name in ('manifest.json', 'resolved_config.yaml', 'time_offset.json'):
        shutil.copy(source / name, target / name)
    (target / 'reference').symlink_to((source / 'reference').resolve())
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(source / 'input_bag'), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    topics = reader.get_all_topics_and_types()
    types = {t.name: t.type for t in topics}
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(target / 'input_bag'), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    for topic in topics:
        writer.create_topic(topic)
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic == '/kinetics_observer/configuration':
            message = deserialize_message(data, get_message(types[topic]))
            state = list(message.initial_state)
            state[13:16] = [float(v) for v in bias]
            message.initial_state = state
            data = serialize_message(message)
        writer.write(topic, data, stamp)
    del writer
    print(f'{project}: biais initial seme {np.degrees(bias) * 1000} mdeg/s')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

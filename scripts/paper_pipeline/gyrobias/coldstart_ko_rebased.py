#!/usr/bin/env python3
"""Bag du Kinetics Observer : tronque au premier instant de la mocap, rampe de biais REBASEE.

Pendant de coldstart_rebased.py. On repart du cache NON BIAISE (kinetics_eval_ref) pour n'injecter
la derive qu'une seule fois, avec la meme rebase : b(t) = model.at(t - START). Le bag source n'est
jamais modifie.

L'etat initial est le meme froid que la premiere version -- pose du centroide au premier instant
note, zero partout ailleurs -- pour que les deux experiences ne different que par la rebase.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
SCRATCH = Path('/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/'
               '287daf9e-ad1c-4931-a07e-923142412c80/scratchpad/bias')
PROJECT = 'HRP5P_LongWalk'
SOURCE = ROOT / f'Projects/{PROJECT}/output_data/kinetics_eval_ref'
TARGET = ROOT / f'Projects/{PROJECT}/output_data/kinetics_eval_init2'


def main():
    start = json.loads((SOURCE / 'time_offset.json').read_text())['offset']
    warm = np.load(SCRATCH / 'ko_state_t0.npy')
    grid_t, grid_b = model.trajectory()

    TARGET.mkdir(parents=True, exist_ok=True)
    for name in ('manifest.json', 'resolved_config.yaml', 'time_offset.json'):
        shutil.copy(SOURCE / name, TARGET / name)
    for name in ('reference', 'logReplay_full.bin'):
        link, source = TARGET / name, SOURCE / name
        if not link.exists() and source.exists():
            link.symlink_to(source.resolve())

    destination = TARGET / 'input_bag'
    if destination.exists():
        sys.exit(f'ABANDON: {destination} existe deja')

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(SOURCE / 'input_bag'), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    topics = reader.get_all_topics_and_types()
    types = {t.name: t.type for t in topics}
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(destination), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    for topic in topics:
        writer.create_topic(topic)

    kept = dropped = 0
    first = last = None
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic == '/kinetics_observer/configuration':
            message = deserialize_message(data, get_message(types[topic]))
            cold = np.zeros(len(message.initial_state))
            cold[0:7] = warm[0:7]
            message.initial_state = cold.tolist()
            print(f'  initial_state froid ({len(cold)} valeurs), pos {cold[0:3]}')
            writer.write(topic, serialize_message(message), stamp)
            continue
        message = deserialize_message(data, get_message(types[topic]))
        t = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        if t < start:
            dropped += 1
            continue
        for imu in message.imus:
            imu.angular_velocity.x += float(np.interp(t - start, grid_t, grid_b[:, 0]))
            imu.angular_velocity.y += float(np.interp(t - start, grid_t, grid_b[:, 1]))
            imu.angular_velocity.z += float(np.interp(t - start, grid_t, grid_b[:, 2]))
        if first is None:
            first = t
        last = t
        kept += 1
        writer.write(topic, serialize_message(message), stamp)
        if kept % 200000 == 0:
            print(f'    {kept} messages', flush=True)
    del writer
    print(f'  {kept} gardes, {dropped} jetes '
          f'({100 * dropped / (kept + dropped):.1f}%), t de {first:.3f} a {last:.3f} s')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Redemarre le Kinetics Observer au premier instant de la mocap, a froid.

Pendant de coldstart_riekf.py. Le bag source (kinetics_eval_bias, deja porteur de la derive
BMI088) n'est jamais modifie : tout va dans un nouveau cache kinetics_eval_init.

Deux changements, et deux seulement :

1. les messages d'entree anterieurs au premier instant note sont jetes -- 34.0 % d'entre eux,
   exactement la meme proportion que pour le RI-EKF ;
2. initial_state passe de l'etat de debut de log a un etat FROID pris au premier instant note.

Ce que veut dire froid : on recopie le motif que le pipeline utilise deja en debut de log --
position du centroide, orientation, et ZERO partout ailleurs (vitesses, biais gyro, torseur non
modelise, poses et torseurs de contact, qui sont poses quand les contacts sont declares). Seule la
geometrie change, elle est prise a t=0 au lieu de t=-1002 s.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
SCRATCH = Path('/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/'
               '287daf9e-ad1c-4931-a07e-923142412c80/scratchpad/bias')
PROJECT = 'HRP5P_LongWalk'
SOURCE = ROOT / f'Projects/{PROJECT}/output_data/kinetics_eval_bias'
TARGET = ROOT / f'Projects/{PROJECT}/output_data/kinetics_eval_init'


def cold_state(warm, size):
    """Le motif du demarrage a froid du pipeline, mais avec la geometrie de l'instant vise."""
    state = np.zeros(size)
    state[0:3] = warm[0:3]      # position du centroide
    state[3:7] = warm[3:7]      # orientation
    return state


def main():
    start = json.loads((SOURCE / 'time_offset.json').read_text())['offset']
    warm = np.load(SCRATCH / 'ko_state_t0.npy')

    TARGET.mkdir(parents=True, exist_ok=True)
    for name in ('manifest.json', 'resolved_config.yaml', 'time_offset.json'):
        shutil.copy(SOURCE / name, TARGET / name)
    for name in ('reference', 'logReplay_full.bin'):
        link = TARGET / name
        if not link.exists():
            link.symlink_to((SOURCE / name).resolve())

    destination = TARGET / 'input_bag'
    if destination.exists():
        sys.exit(f'ABANDON: {destination} existe deja -- l\'effacer sciemment avant de refaire')

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
            before = list(message.initial_state)
            message.initial_state = cold_state(warm, len(before)).tolist()
            print(f'  initial_state reecrit ({len(before)} valeurs)')
            print(f'    avant : pos {np.array(before[0:3])}')
            print(f'    apres : pos {np.array(message.initial_state[0:3])}')
            writer.write(topic, serialize_message(message), stamp)
            continue
        message = deserialize_message(data, get_message(types[topic]))
        t = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        if t < start:
            dropped += 1
            continue
        if first is None:
            first = t
        last = t
        kept += 1
        writer.write(topic, data, stamp)
        if kept % 200000 == 0:
            print(f'    {kept} messages gardes', flush=True)
    del writer
    total = kept + dropped
    print(f'  {kept} gardes, {dropped} jetes ({100 * dropped / total:.1f}%), '
          f't de {first:.3f} a {last:.3f} s')
    print(f'\n=== rejouer avec :\n  .venv/bin/python scripts/kinetics_eval.py --projects {PROJECT} '
          f'--cache-suffix _init run --label kocold --no-latest --no-open')


if __name__ == '__main__':
    main()

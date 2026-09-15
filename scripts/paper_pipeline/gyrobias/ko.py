#!/usr/bin/env python3
"""Injecte la MEME derive de biais dans le bag du replay, pour le Kinetics Observer.

Le bag d'origine n'est jamais touche : tout va dans un cache suffixe `kinetics_eval_bias`, que
`kinetics_eval.py --cache-suffix _bias` sait lire. Le manifeste est recopie tel quel -- il pointe
les fichiers sources en chemin ABSOLU, qui eux ne changent pas, donc le garde-fou passe sans
qu'on ait a regenerer quoi que ce soit.

Le biais est interpole depuis la meme realisation que celle injectee dans HartleyInput.txt, sur
l'axe du temps : les deux estimateurs voient la meme derive au meme instant.
"""
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
PROJECT = sys.argv[1] if len(sys.argv) > 1 else 'HRP5P_LongWalk'
SUFFIX = '_bias'
CACHE = ROOT / f'Projects/{PROJECT}/output_data/kinetics_eval'
TARGET = ROOT / f'Projects/{PROJECT}/output_data/kinetics_eval{SUFFIX}'


def prepare_cache():
    TARGET.mkdir(parents=True, exist_ok=True)
    # Les gros fichiers sources ne sont pas recopies : seul le bag est reecrit, le reste est
    # identique et le manifeste pointe de toute facon les originaux.
    for name in ('manifest.json', 'resolved_config.yaml'):
        if (CACHE / name).exists():
            shutil.copy(CACHE / name, TARGET / name)
    link = TARGET / 'reference'
    if not link.exists():
        link.symlink_to(CACHE / 'reference')


def rewrite_bag():
    destination = TARGET / 'input_bag'
    if destination.exists():
        print('  bag deja ecrit'); return
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(CACHE / 'input_bag'), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    topics = reader.get_all_topics_and_types()
    types = {t.name: t.type for t in topics}
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(destination), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    for t in topics:
        writer.create_topic(t)
    grid_t, grid_b = model.trajectory()
    kind = get_message(types['/kinetics_observer/input'])
    n = 0
    first = last = None
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic == '/kinetics_observer/input':
            msg = deserialize_message(data, kind)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            for imu in msg.imus:
                w = imu.angular_velocity
                w.x += float(np.interp(t, grid_t, grid_b[:, 0]))
                w.y += float(np.interp(t, grid_t, grid_b[:, 1]))
                w.z += float(np.interp(t, grid_t, grid_b[:, 2]))
            data = serialize_message(msg)
            n += 1
            if first is None:
                first = t
            last = t
            if n % 100000 == 0:
                print(f'    {n} messages', flush=True)
        writer.write(topic, data, stamp)
    del writer
    print(f'  {n} messages IMU modifies, t de {first:.1f} a {last:.1f} s', flush=True)


if __name__ == '__main__':
    print(f'=== cache suffixe : {TARGET}')
    prepare_cache()
    rewrite_bag()
    print('=== rejouer avec :')
    print(f'  .venv/bin/python scripts/kinetics_eval.py --projects {PROJECT} '
          f'--cache-suffix {SUFFIX} run --label kobias --no-plots --no-latest --no-open')

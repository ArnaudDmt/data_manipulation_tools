#!/usr/bin/env python3
"""LongWalk : retire, pied par pied et sur les six composantes, le biais mesure pied en l'air.

Usage : apply_swing_offset.py <suffixe source> <suffixe cible>
Offsets dans le REPERE DU CONTACT, medianes sur 34 phases de vol entre 1039 et 1100 s du log re-tick
(`swing_bias.py`), c'est-a-dire juste apres le lancement de la mocap. Appliques constants a toutes les
mesures d'un contact actif ; la derive qui continue pendant la marche n'est pas corrigee. Le cache source
n'est pas modifie.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import rosbag2_py
import yaml
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
PROJECT = 'HRP5P_LongWalk'
OFFSET = {  # force x y z (N), couple x y z (N.m), repere du contact
    'LeftFootCenter': np.array([1.49, 1.83, 22.09, -0.392, 0.297, -0.026]),
    'RightFootCenter': np.array([1.71, 1.79, 20.52, 0.100, 0.315, -0.006]),
}


def main(source_suffix, target_suffix):
    source = ROOT / f'Projects/{PROJECT}/output_data/kinetics_eval{source_suffix}'
    target = ROOT / f'Projects/{PROJECT}/output_data/kinetics_eval{target_suffix}'
    names = {c['id']: c['name'] for c in yaml.safe_load(open(source / 'resolved_config.yaml'))['contacts']}
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
    kind = get_message(types['/kinetics_observer/input'])
    counts = {n: 0 for n in OFFSET}
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic == '/kinetics_observer/input':
            message = deserialize_message(data, kind)
            for contact in message.contacts:
                w = contact.measured_wrench
                name = names.get(contact.id)
                if name in OFFSET and contact.active and (w.force.x or w.force.y or w.force.z):
                    o = OFFSET[name]
                    w.force.x -= o[0]; w.force.y -= o[1]; w.force.z -= o[2]
                    w.torque.x -= o[3]; w.torque.y -= o[4]; w.torque.z -= o[5]
                    counts[name] += 1
            data = serialize_message(message)
        writer.write(topic, data, stamp)
    del writer
    print(f'{PROJECT}{target_suffix}: mesures corrigees par contact {counts}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

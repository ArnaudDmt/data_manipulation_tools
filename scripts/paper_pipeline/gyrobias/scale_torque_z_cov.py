#!/usr/bin/env python3
"""Cache copie ou la variance du COUPLE DE LACET mesure est multipliee, message par message.

Usage : scale_torque_z_cov.py <projet> <facteur> <suffixe>
Dans le replay, la covariance contact_wrench de la configuration est ignoree : le noeud prend
`contact.wrench_covariance` porte par chaque message, deja exprimee dans le REPERE DU CONTACT
(kinetics_observer_bridge.cpp:445-449). L'element (5,5), indice 35 en ligne, est donc la variance du
couple autour de la normale au pied. On le multiplie ; augmenter un terme diagonal garde la matrice
definie positive. Le cache par defaut n'est pas modifie.
"""
import shutil
import sys
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')


def main(project, factor, suffix):
    factor = float(factor)
    source = ROOT / f'Projects/{project}/output_data/kinetics_eval'
    target = ROOT / f'Projects/{project}/output_data/kinetics_eval{suffix}'
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
    changed = 0
    before = after = None
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic == '/kinetics_observer/input' and factor != 1.0:
            message = deserialize_message(data, get_message(types[topic]))
            for contact in message.contacts:
                covariance = list(contact.wrench_covariance)
                if covariance[35] != 0.0:
                    if before is None:
                        before = covariance[35]
                    covariance[35] *= factor
                    after = covariance[35]
                    contact.wrench_covariance = covariance
                    changed += 1
            data = serialize_message(message)
        writer.write(topic, data, stamp)
    del writer
    print(f'{project}{suffix}: facteur {factor:g}, {changed} covariances modifiees'
          + (f', variance couple z {before:.6g} -> {after:.6g}' if before is not None else ''), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:4])

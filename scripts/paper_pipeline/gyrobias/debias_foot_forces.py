#!/usr/bin/env python3
"""LongWalk : retire la derive des capteurs d'effort des pieds, mesuree entre deux phases statiques.

Usage : debias_foot_forces.py <suffixe source> <suffixe cible> [rampe]

Robot debout sur ses deux pieds au debut du log [0, 10] s et juste avant la mocap [992, 1002] s : la
charge vraie est la meme. La SOMME des forces mesurees passe de 928.0 N (m.g = 931.7 N) a 963.0 N, soit
une derive de (-4.00, -1.08, +34.98) N. Les ecarts PAR PIED (+45 / -10 N en z) melangent derive et report
de poids : inutilisables. La derive est partagee a parts egales (mesure pied en l'air du 2026-09-01 :
les deux capteurs derivaient pareil). Force seulement, repere du contact.
Sans `rampe` : offset CONSTANT -- pour un cache deja coupe au lancement de la mocap (depart a froid).
Avec `rampe` : 0 au debut du log -> valeur au lancement de la mocap, constant ensuite.
La derive qui continue pendant la marche n'est PAS corrigee. Le cache source n'est pas modifie.
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
PROJECT = 'HRP5P_LongWalk'
PER_FOOT = np.array([-4.00, -1.08, 34.98]) / 2.0


def main(source_suffix, target_suffix, ramp):
    source = ROOT / f'Projects/{PROJECT}/output_data/kinetics_eval{source_suffix}'
    target = ROOT / f'Projects/{PROJECT}/output_data/kinetics_eval{target_suffix}'
    ramp_end = json.loads((source / 'time_offset.json').read_text())['offset']
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
    corrected = 0
    first = None
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic == '/kinetics_observer/input':
            message = deserialize_message(data, kind)
            t = message.header.stamp.sec + 1e-9 * message.header.stamp.nanosec
            first = t if first is None else first
            offset = PER_FOOT * (min(t / ramp_end, 1.0) if ramp else 1.0)
            for contact in message.contacts:
                f = contact.measured_wrench.force
                if contact.active and (f.x or f.y or f.z):
                    f.x -= float(offset[0]); f.y -= float(offset[1]); f.z -= float(offset[2])
                    corrected += 1
            data = serialize_message(message)
        writer.write(topic, data, stamp)
    del writer
    print(f'{PROJECT}{target_suffix}: {corrected} mesures de force corrigees, premiere entree a t_log={first:.3f}, '
          f'offset {"en rampe" if ramp else "constant"} par pied {np.round(PER_FOOT, 2)} N', flush=True)


if __name__ == '__main__':
    main(sys.argv[1] if sys.argv[1] != '-' else '', sys.argv[2], len(sys.argv) > 3 and sys.argv[3] == 'rampe')

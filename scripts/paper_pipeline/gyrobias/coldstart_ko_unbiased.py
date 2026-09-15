#!/usr/bin/env python3
"""Cache KO pour un redemarrage a froid au premier instant de la mocap, SANS derive injectee.

Usage : coldstart_ko_unbiased.py <projet> <dossier du run chaud de reference>

Le cache par defaut (kinetics_eval) n'est jamais modifie : tout va dans kinetics_eval_cold.
Deux changements seulement :
  1. les entrees anterieures au decalage de la fenetre notee sont jetees ;
  2. initial_state devient l'etat froid du pipeline -- pose du centroide, zero partout ailleurs --
     avec la pose prise a t=0 dans le kinetics_centroid.txt du run chaud (verifie : ce fichier
     porte la pose d'etat du centroide).
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
sys.path.insert(0, str(ROOT / 'scripts'))
import kinetics_eval as ke  # noqa: E402


def main(project, warm_run):
    source = ROOT / f'Projects/{project}/output_data/kinetics_eval'
    target = ROOT / f'Projects/{project}/output_data/kinetics_eval_cold'
    offset = json.loads((source / 'time_offset.json').read_text())['offset']
    step = ke.project_timestep(ROOT / f'Projects/{project}')
    skipped = ke.skipped_iteration_shift(ROOT / f'Projects/{project}', step)
    centroid = np.loadtxt(Path(warm_run) / project / 'kinetics_centroid.txt', comments='#')
    row = centroid[np.abs(centroid[:, 0]).argmin()]
    print(f'{project}: pose chaude prise a t={row[0]:.4f} s de la fenetre, decalage {offset:.3f} s')

    target.mkdir(parents=True, exist_ok=True)
    for name in ('manifest.json', 'resolved_config.yaml', 'time_offset.json'):
        shutil.copy(source / name, target / name)
    link = target / 'reference'
    if not link.exists():
        link.symlink_to((source / 'reference').resolve())
    destination = target / 'input_bag'
    if destination.exists():
        sys.exit(f'ABANDON: {destination} existe deja')

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(source / 'input_bag'), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    topics = reader.get_all_topics_and_types()
    types = {t.name: t.type for t in topics}
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(destination), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    for topic in topics:
        writer.create_topic(topic)
    kept = dropped = 0
    first = None
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        message = deserialize_message(data, get_message(types[topic]))
        if topic == '/kinetics_observer/configuration':
            size = len(message.initial_state)
            if size == 0:
                sys.exit('ABANDON: initial_state vide dans le cache source, refaire prepare --force')
            cold = np.zeros(size)
            cold[0:3] = row[1:4]
            cold[3:7] = row[4:8]
            message.initial_state = cold.tolist()
            writer.write(topic, serialize_message(message), stamp)
            continue
        t = message.header.stamp.sec + 1e-9 * message.header.stamp.nanosec
        # Le decalage est sur l'horloge CORRIGEE de la routine (kinetics_eval, commit d05499f) : on
        # compare donc temps brut + iterations sautees, pas le temps brut seul.
        row = min(int(round(t / step)), len(skipped) - 1) if skipped is not None else 0
        if t + (skipped[row] if skipped is not None else 0.0) < offset:
            dropped += 1
            continue
        if first is None:
            first = t
        kept += 1
        writer.write(topic, data, stamp)
    del writer
    print(f'{project}: {kept} entrees gardees, {dropped} jetees '
          f'({100 * dropped / (kept + dropped):.1f} %), premiere a t_log={first:.3f}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

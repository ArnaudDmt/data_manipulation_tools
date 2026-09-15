#!/usr/bin/env python3
"""Variante du KO sur l'experience a froid rebasee, sans toucher aux sources partagees.

Usage : ko_variant.py <nom> <cle=valeur> ...
Les cles sont celles de la configuration RESOLUE (with_unmodeled_wrench, with_gyro_bias),
plus angular_stiffness / angular_damping / linear_stiffness / linear_damping du contact_model,
plus wrench_sensors (true/false), qui n'est pas dans le YAML : merge_tuning le recopie du bag
d'entree, donc on corrige le bag de configuration apres sa fabrication.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
SCRATCH = Path('/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/'
               '287daf9e-ad1c-4931-a07e-923142412c80/scratchpad/bias')
PROJECT = 'HRP5P_LongWalk'
CACHE = ROOT / f'Projects/{PROJECT}/output_data/kinetics_eval_init2'
REFERENCE = ROOT / f'results/kocold2-ba73545d4c/{PROJECT}/resolved_config.yaml'
MODEL_KEYS = {'angular_stiffness', 'angular_damping', 'linear_stiffness', 'linear_damping'}


def build(name, settings):
    data = yaml.safe_load(REFERENCE.read_text())
    sensors = settings.pop('wrench_sensors', None)
    for key, value in settings.items():
        if key in MODEL_KEYS:
            data['contact_model'][key] = value
        elif key in data:
            data[key] = value
        else:
            sys.exit(f'ABANDON: cle inconnue {key}')
    work = SCRATCH / name
    work.mkdir(parents=True, exist_ok=True)
    config = work / 'resolved_config.yaml'
    config.write_text(yaml.safe_dump(data, sort_keys=False))
    return config, work, sensors


def set_sensors(bag, enabled):
    import rosbag2_py
    from rclpy.serialization import deserialize_message, serialize_message
    from kinetics_observer_ros2.msg import KineticsConfiguration
    topic = '/kinetics_observer/configuration'
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    _, data, stamp = reader.read_next()
    message = deserialize_message(data, KineticsConfiguration)
    del reader
    for contact in message.contacts:
        contact.has_wrench_sensor = enabled
    shutil.rmtree(bag)
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    writer.create_topic(rosbag2_py.TopicMetadata(
        id=0, name=topic, type='kinetics_observer_ros2/msg/KineticsConfiguration',
        serialization_format='cdr'))
    writer.write(topic, serialize_message(message), stamp)
    del writer
    print(f'  has_wrench_sensor force a {enabled}')


def main():
    name = sys.argv[1]
    settings = {}
    for item in sys.argv[2:]:
        key, _, raw = item.partition('=')
        settings[key] = yaml.safe_load(raw)
    config, work, sensors = build(name, settings)
    print(f'{name}: {config}')
    tuning = work / 'configuration'
    replay = ROOT / f'results/{name}/{PROJECT}/standalone_replay'
    for path in (tuning, replay):
        shutil.rmtree(path, ignore_errors=True)
    replay.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['ros2', 'run', 'test_state_obs_ros2', 'make_kinetics_config_bag.py',
                    '--input-bag', str(CACHE / 'input_bag'), '--config', str(config),
                    '--out', str(tuning)], check=True)
    if sensors is not None:
        set_sensors(tuning, bool(sensors))
    subprocess.run(['ros2', 'launch', 'test_state_obs_ros2', 'test_kinetics_replay.launch.py',
                    f'input_bag:={CACHE / "input_bag"}', f'configuration_bag:={tuning}',
                    f'output_bag:={replay}', 'startup_delay:=0.5', 'shutdown_delay:=0.5'],
                   check=True)
    print(f'replay ecrit dans {replay}')


if __name__ == '__main__':
    main()

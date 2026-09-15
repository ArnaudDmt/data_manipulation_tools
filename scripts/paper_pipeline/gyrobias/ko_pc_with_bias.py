#!/usr/bin/env python3
"""KO-PC avec estimation du biais gyrometrique, sur l'experience a froid et rebasee.

KO-PC ("Kinetics Observer sans capteurs d'effort") desactive d'un bloc quatre choses :
mesures de torseur, raideur angulaire de contact, torseur non modelise ET biais gyro
(MCKineticsObserver.cpp:139-143, et kinetics_bag_common.py:166-167 cote replay). Sur une experience
dont tout l'objet est un biais injecte, la derniere desactivation vide la question de son sens.

Ce script remonte KO-PC a l'identique mais en gardant le biais dans l'etat, SANS toucher aux
sources partagees :

  - la configuration resolue est reprise de la course kocold2 puis corrigee sur quatre cles
    (with_unmodeled_wrench, raideur et amortissement angulaires, biais garde a true) ;
  - has_wrench_sensor ne vient PAS du YAML : merge_tuning() le recopie du bag d'entree
    (make_kinetics_config_bag.py:47). On corrige donc le bag de configuration APRES sa fabrication,
    ce qui evite de dupliquer les 3.8 Go du bag d'entree.

Rien n'est ecrit hors du scratchpad et de results/.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
WORKSPACE = Path('/home/arnaud/devel/src/catkin_ws')
SCRATCH = Path('/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/'
               '287daf9e-ad1c-4931-a07e-923142412c80/scratchpad/bias')
PROJECT = 'HRP5P_LongWalk'
CACHE = ROOT / f'Projects/{PROJECT}/output_data/kinetics_eval_init2'
REFERENCE = ROOT / f'results/kocold2-ba73545d4c/{PROJECT}/resolved_config.yaml'
OUTPUT = SCRATCH / 'kopc'


def patched_configuration():
    data = yaml.safe_load(REFERENCE.read_text())
    data['with_unmodeled_wrench'] = False          # KO-PC : pas de torseur non modelise
    data['with_gyro_bias'] = True                  # <- la seule difference avec KO-PC
    model = data['contact_model']
    model['angular_stiffness'] = [0.0, 0.0, 0.0]   # KO-PC : pas de raideur angulaire
    model['angular_damping'] = [0.0, 0.0, model['angular_damping'][2]]   # amortissement en lacet garde
    for contact in data['contacts']:
        contact['has_wrench_sensor'] = False
    destination = OUTPUT / 'resolved_config.yaml'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(data, sort_keys=False))
    return destination


def fix_wrench_sensors(bag):
    """merge_tuning recopie has_wrench_sensor du bag d'entree ; on le remet a false ici."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message, serialize_message
    from kinetics_observer_ros2.msg import KineticsConfiguration
    topic = '/kinetics_observer/configuration'
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    name, data, stamp = reader.read_next()
    message = deserialize_message(data, KineticsConfiguration)
    del reader
    print(f'  avant : gyro_bias={message.with_gyro_bias} wrench={message.with_unmodeled_wrench} '
          f'capteurs={[c.has_wrench_sensor for c in message.contacts]} '
          f'raideur_ang={list(message.contacts[0].angular_stiffness)}')
    for contact in message.contacts:
        contact.has_wrench_sensor = False
    shutil.rmtree(bag)
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    writer.create_topic(rosbag2_py.TopicMetadata(
        id=0, name=topic, type='kinetics_observer_ros2/msg/KineticsConfiguration',
        serialization_format='cdr'))
    writer.write(topic, serialize_message(message), stamp)
    del writer
    print(f'  apres : capteurs={[c.has_wrench_sensor for c in message.contacts]}')


def main():
    config = patched_configuration()
    print(f'configuration corrigee : {config}')
    tuning = OUTPUT / 'configuration'
    replay = ROOT / f'results/kopc-bias/{PROJECT}/standalone_replay'
    for path in (tuning, replay):
        shutil.rmtree(path, ignore_errors=True)
    replay.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(['ros2', 'run', 'test_state_obs_ros2', 'make_kinetics_config_bag.py',
                    '--input-bag', str(CACHE / 'input_bag'), '--config', str(config),
                    '--out', str(tuning)], check=True)
    fix_wrench_sensors(tuning)

    subprocess.run(['ros2', 'launch', 'test_state_obs_ros2', 'test_kinetics_replay.launch.py',
                    f'input_bag:={CACHE / "input_bag"}', f'configuration_bag:={tuning}',
                    f'output_bag:={replay}', 'startup_delay:=0.5', 'shutdown_delay:=0.5'],
                   check=True)
    print(f'\nreplay ecrit dans {replay}')


if __name__ == '__main__':
    main()

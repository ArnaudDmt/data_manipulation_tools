#!/usr/bin/env python3
"""Variante du KO rejouee sur un cache existant (a froid ou biaise), scoree par rpg comme kinetics_eval.

Usage : cold_variant.py <projet> <suffixe de cache> <label> [section.cle=valeur ...]

Reprend le coeur de kinetics_eval.evaluate (bag de configuration, replay, extraction avec decalage
de fenetre et correction des iterations sautees, analyse rpg) sans sa validation de cache : les
caches a froid sont des derives faits a la main dont le manifeste ne suit plus la config installee.
Le bag de sortie est garde le temps d'en extraire le biais gyro et le torseur non modelise
(raw_state 13:22) dans states.npz, puis supprime.
"""
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
sys.path.insert(0, str(ROOT / 'scripts'))
import kinetics_eval as ke  # noqa: E402

KEYS = [('Transxy m', 'rel_trans_x_y_norm'), ('Transz m', 'rel_trans_z'), ('tilt', 'rel_tilt'),
        ('lacet', 'rel_yaw')]


def means(pickle_path):
    data = pickle.load(open(pickle_path, 'rb'))
    return {d: [float(np.mean(np.abs(v[k]))) for _, k in KEYS] for d, v in data.items()}


def states(bag, destination):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from kinetics_observer_ros2.msg import KineticsState
    storage = 'mcap' if any(Path(bag).glob('*.mcap')) else 'sqlite3'
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=storage), rosbag2_py.ConverterOptions('', ''))
    rows = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != '/kinetics_observer/estimated_state':
            continue
        m = deserialize_message(data, KineticsState)
        rows.append([m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec, *m.raw_state[13:22]])
    a = np.array(rows)
    np.savez_compressed(destination, t=a[:, 0], bias=a[:, 1:4], force=a[:, 4:7], torque=a[:, 7:10])


def main():
    name, suffix, label = sys.argv[1:4]
    project = ROOT / f'Projects/{name}'
    cache = project / f'output_data/kinetics_eval{suffix}'
    config = yaml.safe_load((cache / 'resolved_config.yaml').read_text())
    for item in sys.argv[4:]:
        key, _, raw = item.partition('=')
        section, _, leaf = key.partition('.')
        target = config[section] if leaf else config
        leaf = leaf or section
        if leaf not in target:
            sys.exit(f'ABANDON: cle inconnue {key}')
        target[leaf] = yaml.safe_load(raw)

    destination = ROOT / f'results/{label}/{name}'
    if destination.exists():
        sys.exit(f'ABANDON: {destination} existe deja')
    destination.mkdir(parents=True)
    (destination / 'resolved_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (destination / 'settings.txt').write_text(f'cache {cache}\n' + '\n'.join(sys.argv[4:]) + '\n')

    env = ke.ros_environment(ke.DEFAULT_WORKSPACE)
    env['ROS_DOMAIN_ID'] = '77'
    tuning, replay = destination / 'configuration', destination / 'standalone_replay'
    input_bag = cache / 'input_bag'
    ke.run(['ros2', 'run', 'test_state_obs_ros2', 'make_kinetics_config_bag.py', '--input-bag', input_bag,
            '--config', destination / 'resolved_config.yaml', '--out', tuning], env)
    ke.run(['ros2', 'launch', 'test_state_obs_ros2', 'test_kinetics_replay.launch.py', f'input_bag:={input_bag}',
            f'configuration_bag:={tuning}', f'output_bag:={replay}', 'startup_delay:=0.5',
            'shutdown_delay:=0.5'], env)
    offset = json.loads((cache / 'time_offset.json').read_text())['offset']
    step = ke.project_timestep(project)
    shift = ke.skipped_iteration_shift(project, step)
    trajectory = destination / 'kinetics.txt'
    ke.extract_ros_trajectories(replay, trajectory, destination / 'kinetics_centroid.txt',
                                destination / 'kinetics_velocity.txt', env, offset, shift, step)
    states(replay, destination / 'states.npz')
    shutil.rmtree(replay)
    shutil.rmtree(tuning)
    ke.analyze(trajectory, cache / 'reference/mocap.txt', destination / 'eval', ke.sublengths(project))
    result = means(destination / 'eval/saved_results/traj_est/cached/cached_rel_err.pickle')
    for distance, values in result.items():
        print(f'RESULTAT {label:26s} {name:22s} ({distance} m) '
              + '  '.join(f'{k} {v:.4f}' for (k, _), v in zip(KEYS, values)), flush=True)


if __name__ == '__main__':
    main()

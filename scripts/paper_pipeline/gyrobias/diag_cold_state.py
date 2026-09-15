#!/usr/bin/env python3
"""Pourquoi le KO se degrade-t-il au depart a froid ? Etat interne, course normale contre froide.

Usage : diag_cold_state.py <projet> <run normal> <run froid>
Lit raw_state dans les deux bags de replay (garde sans --no-plots) : biais gyro, torseur non modelise,
et compare l'orientation de la base entre les deux courses au fil de la fenetre notee.
Disposition de raw_state : pos 0:3, ori 3:7, linVel 7:10, angVel 10:13, biais gyro 13:16,
force non modelisee 16:19, couple non modelise 19:22, contacts 22+13k (pos 3, ori 4, force 3, couple 3).
"""
import json
import sys
from pathlib import Path

import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation as R

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
sys.path.insert(0, str(ROOT / 'scripts'))
import kinetics_eval as ke  # noqa: E402


def read(run, project):
    directory = ROOT / f'Projects/{project}'
    suffix = '_cold' if 'cold' in run else ''
    offset = json.loads((directory / f'output_data/kinetics_eval{suffix}/time_offset.json').read_text())['offset']
    step = ke.project_timestep(directory)
    skipped = ke.skipped_iteration_shift(directory, step)
    reader = SequentialReader()
    reader.open(StorageOptions(uri=str(next(ROOT.glob(f'results/{run}-*')) / project / 'standalone_replay'),
                               storage_id='mcap'), ConverterOptions('', ''))
    kind = get_message('kinetics_observer_ros2/msg/KineticsState')
    times, states = [], []
    while reader.has_next():
        _, data, _ = reader.read_next()
        message = deserialize_message(data, kind)
        raw = message.header.stamp.sec + 1e-9 * message.header.stamp.nanosec
        row = min(int(round(raw / step)), len(skipped) - 1)
        times.append(raw - offset + skipped[row])
        states.append(message.raw_state)
    return np.array(times), np.array(states)


def at(times, values, moments):
    return [values[np.abs(times - m).argmin()] for m in moments]


def main(project, warm_run, cold_run):
    tw, sw = read(warm_run, project)
    tc, sc = read(cold_run, project)
    moments = [0.0, 1.0, 5.0, 15.0, 30.0, 60.0, 120.0, float(min(tw[-1], tc[-1]))]
    print(f'== {project}   (temps dans la fenetre notee, s)')
    print(f"{'':28s}" + ''.join(f'{m:>9.0f}' for m in moments))
    rows = [('biais gyro z (mdeg/s)', lambda s: np.degrees(s[:, 15]) * 1000),
            ('biais gyro |xy| (mdeg/s)', lambda s: np.degrees(np.linalg.norm(s[:, 13:15], axis=1)) * 1000),
            ('force non modelisee z (N)', lambda s: s[:, 18]),
            ('|force non mod. xy| (N)', lambda s: np.linalg.norm(s[:, 16:18], axis=1)),
            ('couple non modelise z (Nm)', lambda s: s[:, 21]),
            ('|couple non mod. xy| (Nm)', lambda s: np.linalg.norm(s[:, 19:21], axis=1))]
    for label, f in rows:
        for name, t, s in (('normal', tw, sw), ('froid', tc, sc)):
            print(f'{label:20s} {name:7s}' + ''.join(f'{v:9.2f}' for v in at(t, f(s), moments)))
    # ecart d'orientation entre les deux courses, en repere relatif depuis t=0 (insensible au repere initial)
    grid = np.arange(0.0, moments[-1], 0.05)
    iw = np.clip(np.searchsorted(tw, grid), 0, len(tw) - 1)
    ic = np.clip(np.searchsorted(tc, grid), 0, len(tc) - 1)
    qw = R.from_quat(sw[iw, 3:7]); qc = R.from_quat(sc[ic, 3:7])
    rel = (qw[0].inv() * qw).inv() * (qc[0].inv() * qc)
    yaw = np.degrees(np.unwrap(rel.as_euler('ZYX')[:, 0]))
    tilt = np.degrees(np.linalg.norm(rel.as_euler('ZYX')[:, 1:], axis=1))
    print('ecart froid-normal, lacet cumule (deg)' + ''.join(f'{v:9.3f}' for v in at(grid, yaw, moments)))
    print('ecart froid-normal, tilt (deg)       ' + ''.join(f'{v:9.3f}' for v in at(grid, tilt, moments)))
    print(f'etat normal a t=0 : biais {np.degrees(sw[np.abs(tw).argmin(), 13:16])*1000} mdeg/s, '
          f'torseur non modelise {np.round(sw[np.abs(tw).argmin(), 16:22], 2)}')


if __name__ == '__main__':
    main(*sys.argv[1:4])

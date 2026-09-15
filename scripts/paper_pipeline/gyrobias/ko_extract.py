#!/usr/bin/env python3
"""Sort d'un bag de replay la trajectoire de la base et le biais gyro estime.

Meme mise en forme que kinetics_eval : kinetics.txt porte l'horloge de la fenetre notee, donc le
decalage du log est deja retranche.
"""
import sys
from pathlib import Path

import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
SCRATCH = Path('/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/'
               '287daf9e-ad1c-4931-a07e-923142412c80/scratchpad/bias')
OFFSET = 1002.2759999762993
PROJECT = 'HRP5P_LongWalk'


def main(name):
    reader = SequentialReader()
    reader.open(StorageOptions(uri=str(ROOT / f'results/{name}/{PROJECT}/standalone_replay'),
                               storage_id='mcap'), ConverterOptions('', ''))
    kind = get_message('kinetics_observer_ros2/msg/KineticsState')
    times, biases, poses = [], [], []
    while reader.has_next():
        _, data, _ = reader.read_next()
        message = deserialize_message(data, kind)
        stamp = message.header.stamp.sec + 1e-9 * message.header.stamp.nanosec
        k = message.global_floating_base_kinematics
        times.append(stamp)
        biases.append((message.gyro_biases[0].bias.x, message.gyro_biases[0].bias.y,
                       message.gyro_biases[0].bias.z))
        poses.append((stamp - OFFSET, k.position.x, k.position.y, k.position.z,
                      k.orientation.x, k.orientation.y, k.orientation.z, k.orientation.w))
    np.savez_compressed(SCRATCH / f'ko_bias_{name}.npz', t=np.array(times), b=np.array(biases))
    destination = ROOT / f'results/{name}/{PROJECT}/kinetics.txt'
    with destination.open('w') as stream:
        stream.write('# timestamp tx ty tz qx qy qz qw\n')
        for row in poses:
            stream.write(' '.join(f'{v:.17g}' for v in row) + '\n')
    print(f'{name}: {len(times)} messages, biais final '
          + ' '.join(f'{v:+.5f}' for v in np.degrees(np.array(biases)[-1])) + ' deg/s')


if __name__ == '__main__':
    main(sys.argv[1])

#!/usr/bin/env python3
"""Biais des capteurs d'effort des pieds de LongWalk, lu pied en l'air, forces ET couples.

Source : voie `MCKineticsObserver_debug_forceSensor_<capteur>_measured{Force,Torque}` du log re-tick
(MCKineticsObserver.cpp:1431-1434) = `forceSensorMeasurements_` = wrenchWithoutGravity, l'entree exacte
de l'observateur, repere du capteur, a chaque iteration. Pied en l'air, cette entree devrait etre nulle :
ce qu'elle vaut est le biais. Passage au repere du contact par la transformation que porte le bag,
T = [[R, 0], [skew(p) R, R]] (bin_to_rosbag_kinetics.py:296, MCKineticsObserver.cpp:1135-1139).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
CSV = Path('/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/'
           '287daf9e-ad1c-4931-a07e-923142412c80/scratchpad/swing/swing.csv')
O = 'Observers_MainObserverPipeline_MCKineticsObserver_debug_forceSensor_'
FEET = {'LeftFootForceSensor': 'LeftFootCenter', 'RightFootForceSensor': 'RightFootCenter'}
TRIM = 0.06          # s rognes a chaque bord de phase de vol
MIN_SWING = 0.15     # s de vol minimum


def swings(t, fz, threshold):
    low = fz < threshold
    edges = np.flatnonzero(np.diff(low.astype(int)))
    bounds = np.split(np.arange(len(t)), edges + 1)
    keep = []
    for idx in bounds:
        if not low[idx[0]] or t[idx[-1]] - t[idx[0]] < MIN_SWING:
            continue
        inner = idx[(t[idx] >= t[idx[0]] + TRIM) & (t[idx] <= t[idx[-1]] - TRIM)]
        if len(inner):
            keep.append(inner)
    return keep


def contact_transforms():
    """T par contact (nom -> matrice 6x6), lu dans le premier message d'entree du bag."""
    import yaml
    cache = ROOT / 'Projects/HRP5P_LongWalk/output_data/kinetics_eval'
    names = {c['id']: c['name'] for c in yaml.safe_load(open(cache / 'resolved_config.yaml'))['contacts']}
    reader = SequentialReader()
    reader.open(StorageOptions(uri=str(cache / 'input_bag'), storage_id='sqlite3'), ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != '/kinetics_observer/input':
            continue
        m = deserialize_message(data, get_message(types[topic]))
        return {names[c.id]: np.array(c.wrench_covariance_transform).reshape(6, 6) for c in m.contacts if c.active}


def main():
    frame = pd.read_csv(CSV, sep=';')
    t = frame['t'].to_numpy()
    transforms = contact_transforms()
    total = np.zeros(6)
    print(f'fenetre {t[0]:.1f} - {t[-1]:.1f} s, {len(t)} echantillons\n')
    for sensor, contact in FEET.items():
        f = frame[[f'{O}{sensor}_measuredForce_{a}' for a in 'xyz']].to_numpy()
        tau = frame[[f'{O}{sensor}_measuredTorque_{a}' for a in 'xyz']].to_numpy()
        stance = np.median(f[f[:, 2] > 200, 2]) if (f[:, 2] > 200).any() else np.nan
        threshold = 0.1 * stance
        segments = swings(t, f[:, 2], threshold)
        if not segments:
            print(f'{sensor}: aucune phase de vol'); continue
        idx = np.concatenate(segments)
        w = np.hstack([f[idx], tau[idx]])
        med = np.median(w, axis=0)
        per_swing = np.array([np.median(np.hstack([f[s], tau[s]]), axis=0) for s in segments])
        raw = frame[[f'{sensor}_{a}' for a in ('fx', 'fy', 'fz', 'cx', 'cy', 'cz')]].to_numpy()
        print(f'== {sensor}  (appui median {stance:.0f} N, seuil {threshold:.0f} N)')
        print(f'   {len(segments)} phases de vol, {len(idx)} echantillons, de {t[segments[0][0]]:.1f} a {t[segments[-1][-1]]:.1f} s')
        print(f'   biais repere capteur : force {np.round(med[:3], 2)} N   couple {np.round(med[3:], 3)} N.m')
        print(f'   dispersion entre phases (ecart-type des medianes) : force {np.round(per_swing[:, :3].std(0), 2)}  couple {np.round(per_swing[:, 3:].std(0), 3)}')
        half = len(per_swing) // 2
        print(f'   1re moitie des phases : fz {per_swing[:half, 2].mean():.2f} N ; 2e moitie : {per_swing[half:, 2].mean():.2f} N')
        print(f'   lecture brute pied en l\'air (avec poids du pied) : force {np.round(np.median(raw[idx, :3], 0), 2)}  couple {np.round(np.median(raw[idx, 3:], 0), 3)}')
        T = transforms.get(contact)
        if T is not None:
            in_contact = T @ med
            total += in_contact
            print(f'   biais repere contact ({contact}) : force {np.round(in_contact[:3], 2)} N   couple {np.round(in_contact[3:], 3)} N.m')
        print()
    print(f'somme des deux pieds, repere contact : force {np.round(total[:3], 2)} N  (mesure statique : (-4.00, -1.08, +34.98))')


if __name__ == '__main__':
    main()

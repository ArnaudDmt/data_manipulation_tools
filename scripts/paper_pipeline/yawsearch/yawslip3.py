#!/usr/bin/env python3
"""Recherche de cause sur le pire glissement en lacet : KO_TRO_2024_RHPS1_SLIPPAGE_3.

Rapports KO/RI-EKF en lacet sur les trois glissements : SLIPPAGE_1 1.031, SLIPPAGE_2 0.982,
**SLIPPAGE_3 1.116**. C'est le seul ou le KO perd nettement, donc le seul ou il y a une cause a
trouver. Un seul dataset coute 20.9 s par essai contre 128 pour sept : six fois plus d'essais.

C'est une recherche de CAUSE, pas de reglage : optimiser sur un dataset de 106 s est le
surajustement maximal, et tout survivant devra etre repasse sur les douze.

Par rapport a yawsearch.py : les process du centroide et le biais gyro sont retires, les
composantes xy du wrench non modelise aussi, et les huit flexibilites de RHPS1 sont ajoutees.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path('/home/arnaud/devel/src/data_manipulation_tools/scripts')))
import kinetics_tune as kt

DATASET = 'KO_TRO_2024_RHPS1_SLIPPAGE_3'
MANTISSAS = (1.0, 4.0)


def ladder(low, high, reference=None):
    rungs = set()
    exponent = int(math.floor(math.log10(low)))
    while 10.0 ** exponent <= high * 1.0000001:
        for mantissa in MANTISSAS:
            value = mantissa * 10.0 ** exponent
            if low * 0.9999999 <= value <= high * 1.0000001:
                rungs.add(value)
        exponent += 1
    if reference is not None:
        rungs.add(reference)
    return tuple(sorted(rungs))


DECADES = {
    'contact_process_orientation_yaw': (1e-12, 1e-2, 1e-4),
    'contact_process_position_xy': (1e-6, 1e-4, 1e-5),
    'contact_new_orientation_yaw': (1e-6, 1e-2, 2e-4),
    'unmodeled_force_process_z': (1e-6, 0.09, 0.09),
    'unmodeled_torque_process_z': (1e-6, 0.09, 0.09),
}
FORCE = tuple(float(m * m) for m in (20, 25, 30, 35, 40))
TORQUE = tuple(float(m * m) for m in (10, 15, 25, 30, 35))
EXACT = {
    'contact_process_force_xy': FORCE, 'contact_process_force_z': FORCE,
    'contact_process_torque_xy': TORQUE, 'contact_process_torque_z': TORQUE,
}
RUNGS = {name: ladder(low, high, ref) for name, (low, high, ref) in DECADES.items()}
RUNGS.update(EXACT)

# --- le wrench non modelise : seulement les composantes z ---------------------------------------
SPLIT = (
    ('unmodeled_force_process_z', 'unmodeled_wrench_process', (2,)),
    ('unmodeled_torque_process_z', 'unmodeled_wrench_process', (5,)),
)
space = [e for e in kt.SPACE if e[0] != 'unmodeled_wrench_process']
for name, field, indices in SPLIT:
    space.append((name, field, indices,
                  math.log10(min(RUNGS[name])), math.log10(max(RUNGS[name]))))

kt.GRID.update(RUNGS)
kt.SQUARE_GRID.update(RUNGS)
kt.SPACE = tuple(
    (e[:-2] + (math.log10(min(RUNGS[e[0]])), math.log10(max(RUNGS[e[0]])))) if e[0] in RUNGS else e
    for e in space)

# --- flexibilites : +-1 exposant autour de la valeur installee ----------------------------------
# Les RAIDEURS sont en log10 absolu, donc on centre exactement sur 3e4 (lineaire) et 727
# (angulaire). Les AMORTISSEMENTS sont un ratio zeta, pas une valeur absolue : je ne peux pas
# centrer sans recalculer l'inertie du modele, donc la plage native est conservee -- elle fait
# deja 1.8 decade, soit a peu pres le +-1 exposant demande.
INSTALLED = {'linear_stiffness': 3e4, 'angular_stiffness': 727.0}
flexibility = []
for entry in kt.FLEXIBILITY_SPACE:
    name, robot, field = entry[0], entry[1], entry[2]
    if robot == 'rhps1' and field in INSTALLED:
        centre = math.log10(INSTALLED[field])
        entry = entry[:-2] + (centre - 1.0, centre + 1.0)
    flexibility.append(entry)
kt.FLEXIBILITY_SPACE = tuple(flexibility)

kt.ALL_SPACE = kt.SPACE + kt.RATIO_SPACE + kt.SCALE_SPACE + kt.FLEXIBILITY_SPACE
# La phase de criblage ignore --datasets et utilise une liste codee en dur qui contient LongWalk.
kt.SCREEN_PROJECTS = (DATASET,)

DIMS = [
    'contact_process_force_xy', 'contact_process_force_z',
    'contact_process_torque_xy', 'contact_process_torque_z',
    'contact_process_position_xy', 'contact_process_position_z',
    'contact_process_orientation_yaw', 'contact_process_orientation_rp',
    'contact_new_position_xy',
    'contact_new_orientation_yaw', 'contact_new_orientation_rp',
    'unmodeled_force_process_z', 'unmodeled_torque_process_z',
] + [e[0] for e in kt.FLEXIBILITY_SPACE if e[1] == 'rhps1']

if __name__ == '__main__':
    print(f'=== {len(DIMS)} dimensions sur {DATASET}')
    for name in DIMS:
        entry = next(e for e in kt.ALL_SPACE if e[0] == name)
        rungs = RUNGS.get(name)
        detail = (f'{len(rungs)} barreaux : {rungs[0]:g} .. {rungs[-1]:g}' if rungs
                  else f'continu 10^{entry[-2]:+.2f} .. 10^{entry[-1]:+.2f}')
        print(f'  {name:34s} {detail}')
    sys.stdout.flush()
    kt.main()

#!/usr/bin/env python3
"""Lance kinetics_tune.py avec les bornes, les barreaux et le decoupage qu'Arnaud a fixes.

Trois choses sont remplacees avant d'appeler kinetics_tune.main() :

1. **Les barreaux** (`GRID`). Le tuner propose une valeur continue dans l'exposant puis la rabat
   sur le barreau le plus proche. Ses mantisses natives sont 1/2/5 ; ici ce sont 1 et 4, plus la
   valeur de reference de chaque axe, pour que la configuration actuelle reste atteignable -- sans
   quoi la recherche ne pourrait pas reproduire son propre point de depart.

2. **Les bornes** (`SPACE`). Changer les barreaux seuls laisserait le sampler proposer hors plage.
   Son echelle native du lacet de contact s'arretait a 1e-4, donc le 1e-2 demande en etait exclu.

3. **Le decoupage du wrench non modelise.** Le tuner n'a qu'UNE dimension pour ses six composantes
   -- `(0,1,2,3,4,5)` -- donc il ne peut pas donner une confiance differente a la force et au
   couple, ni au lacet et au roulis-tangage. Elle est remplacee par quatre dimensions, comme pour
   le wrench de contact, qui lui est deja separe.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path('/home/arnaud/devel/src/data_manipulation_tools/scripts')))
import kinetics_tune as kt

MANTISSAS = (1.0, 4.0)


def ladder(low, high, reference=None):
    """Barreaux `MANTISSAS` par decade entre low et high, plus la valeur de reference."""
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


# --- axes en decades : (borne basse, borne haute, valeur actuelle) ------------------------------
DECADES = {
    'state_angular_velocity_process': (1e-12, 1e-8, 1e-10),
    'contact_process_orientation_yaw': (1e-12, 1e-2, 1e-4),
    'contact_process_position_xy': (1e-6, 1e-4, 1e-5),
    'contact_new_orientation_yaw': (1e-6, 1e-2, 2e-4),
    'gyro_bias_process': (1e-20, 1e-16, 1e-18),
    # Wrench non modelise : intermediaires et valeurs plus faibles, jusqu'a 1e-6.
    'unmodeled_force_process_xy': (1e-6, 0.09, 0.09),
    'unmodeled_force_process_z': (1e-6, 0.09, 0.09),
    'unmodeled_torque_process_xy': (1e-6, 0.09, 0.09),
    'unmodeled_torque_process_z': (1e-6, 0.09, 0.09),
}

# --- axes en carres parfaits : l'ecart-type doit etre un entier rond ----------------------------
# Une variance de 5e2 a un sigma de 22.36, qui ne se met pas dans un tableau ; 625 donne 25.
FORCE = tuple(float(m * m) for m in (20, 25, 30, 35, 40))      # 400 625 900 1225 1600
TORQUE = tuple(float(m * m) for m in (10, 15, 25, 30, 35))     # 100 225 625 900 1225
EXACT = {
    'contact_process_force_xy': FORCE,
    'contact_process_force_z': FORCE,
    'contact_process_torque_xy': TORQUE,
    'contact_process_torque_z': TORQUE,
}

RUNGS = {name: ladder(low, high, ref) for name, (low, high, ref) in DECADES.items()}
RUNGS.update(EXACT)

# --- 1. le decoupage du wrench non modelise ----------------------------------------------------
SPLIT = (
    ('unmodeled_force_process_xy', 'unmodeled_wrench_process', (0, 1)),
    ('unmodeled_force_process_z', 'unmodeled_wrench_process', (2,)),
    ('unmodeled_torque_process_xy', 'unmodeled_wrench_process', (3, 4)),
    ('unmodeled_torque_process_z', 'unmodeled_wrench_process', (5,)),
)
space = [e for e in kt.SPACE if e[0] != 'unmodeled_wrench_process']
for name, field, indices in SPLIT:
    low = math.log10(min(RUNGS[name]))
    high = math.log10(max(RUNGS[name]))
    space.append((name, field, indices, low, high))

# --- 2. les barreaux ---------------------------------------------------------------------------
kt.GRID.update(RUNGS)
kt.SQUARE_GRID.update(RUNGS)

# --- 3. les bornes de proposition, alignees sur les barreaux ------------------------------------
kt.SPACE = tuple(
    (e[:-2] + (math.log10(min(RUNGS[e[0]])), math.log10(max(RUNGS[e[0]])))) if e[0] in RUNGS else e
    for e in space
)
kt.SCALE_SPACE = tuple(
    (e[:-2] + (math.log10(min(RUNGS[e[0]])), math.log10(max(RUNGS[e[0]])))) if e[0] in RUNGS else e
    for e in kt.SCALE_SPACE
)

# active_space() valide --only contre ALL_SPACE, pas contre SPACE : sans cette ligne, les quatre
# dimensions ajoutees sont refusees comme "not searchable" -- c'est ce qui a fait echouer le
# lancement de 07:29.
kt.ALL_SPACE = kt.SPACE + kt.RATIO_SPACE + kt.SCALE_SPACE + kt.FLEXIBILITY_SPACE

# La phase de criblage ignore --datasets : `phase("screen", list(SCREEN_PROJECTS), ...)` utilise une
# liste CODEE EN DUR qui contient HRP5P_LongWalk. Non prepare -- et volontairement exclu -- il fait
# echouer le decompte de read_ratios, donc TOUS les essais : c'est ce qui donnait
# "failed J=+50 geomean=nan wins=0/1" a 10:42. On la remplace par les sept datasets demandes.
DATASETS = (
    'KO_TRO_2024_RHPS1_SLIPPAGE_1', 'KO_TRO_2024_RHPS1_SLIPPAGE_2', 'KO_TRO_2024_RHPS1_SLIPPAGE_3',
    'HRP5_MultiContact_1', 'HRP5_MultiContact_2', 'HRP5_MultiContact_3',
    'KO_TRO2024_RHPS1_1',
)
kt.SCREEN_PROJECTS = DATASETS

if __name__ == '__main__':
    print('=== barreaux imposes ===')
    for name in sorted(RUNGS):
        values = RUNGS[name]
        print(f'  {name:32s} {len(values):2d} : ' + ', '.join(f'{v:g}' for v in values))
    print('=== dimensions du wrench non modelise ===')
    for entry in kt.SPACE:
        if entry[0].startswith('unmodeled'):
            print(f'  {entry[0]:32s} indices {entry[2]}  bornes {entry[-2]:+.2f} .. {entry[-1]:+.2f}')
    sys.stdout.flush()
    kt.main()

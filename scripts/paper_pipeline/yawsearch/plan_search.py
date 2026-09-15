"""Choisit les dimensions de la recherche ciblee lacet, a partir du criblage de l'etape 2.

Un axe mesure inerte au criblage est retire : a 13 dimensions, chaque dimension morte divise la
densite d'echantillonnage sans rien apporter. Les echelles de wrench de contact sont retirees
d'office -- le replay ignore cette covariance, la valeur du bag gagne (memoire du 2026-09-14),
donc le tuner y chercherait dans un sous-espace plat.
"""
import re, sys
from pathlib import Path

SP = Path('/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/287daf9e-ad1c-4931-a07e-923142412c80/scratchpad/grid')

# mes cles de balayage -> dimensions du tuner
MAP = {
    'av': ['state_angular_velocity_process'],
    'oy': ['contact_process_orientation_yaw'],
    'px': ['contact_process_position_xy'],
    'oi': ['contact_new_orientation_yaw', 'contact_new_orientation_rp'],
    'gb': ['gyro_bias_process'],
    'cf': ['contact_process_force_xy'],
    'fz': ['contact_process_force_z'],
    'ct': ['contact_process_torque_xy'],
    'cz': ['contact_process_torque_z'],
    # yawsearch.py remplace l'unique dimension du tuner par ces quatre-la.
    'uf': ['unmodeled_force_process_xy'], 'un': ['unmodeled_force_process_z'],
    'ut': ['unmodeled_torque_process_xy'], 'uz': ['unmodeled_torque_process_z'],
}
# Dimensions du tuner qu'aucun axe du balayage ne couvre mais qui agissent sur l'orientation.
ALWAYS = ['contact_process_orientation_rp', 'state_orientation_process']
# Inertes par construction dans le replay : la covariance de wrench de contact vient du bag.
NEVER = ['contact_wrench_force_scale', 'contact_wrench_moment_xy_scale', 'contact_wrench_moment_z_scale']

log = SP / 'stage2.log'
active, inert = set(), set()
if log.exists():
    for line in log.read_text().splitlines():
        m = re.match(r'\s{2}(\w{2})=(\S+)\s', line)
        if m:
            (active if 'ACTIF' in line else inert).add(m.group(1))
# PAS d'elimination par le criblage pour cette campagne : le tuner fait un nombre FIXE d'essais,
# donc retirer un axe ne fait pas gagner de temps, seulement de la densite. Arnaud prefere garder
# tous les axes et lire le resultat en connaissant cette limite (1400^(1/13) = 1.9 point par axe).
# Le criblage reste rapporte ci-dessous, a titre indicatif.
active = set(MAP)
inert = inert - active

dims = []
for key in sorted(active):
    for d in MAP.get(key, []):
        if d not in dims and d not in NEVER:
            dims.append(d)
for d in ALWAYS:
    if d not in dims:
        dims.append(d)

print(' '.join(dims))
sys.stderr.write(f"actifs   : {sorted(active)}\ninertes retires : {sorted(inert)}\n"
                 f"dimensions du tuner : {len(dims)}\n")

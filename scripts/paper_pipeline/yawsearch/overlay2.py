"""Ecrit un overlay a partir de la config retenue et d'un dictionnaire de reglages.

Usage : overlay2.py <fichier> cle=valeur ...
Cles reconnues (tranches des vecteurs d'overlay, cf. MC_RTC_KEYS) :
    av  state_angular_velocity_process [0:3]   stateAngVelProcessVariance
    px  contact_process                [0:2]   contactPositionProcessVariance   x,y
    oy  contact_process                [5:6]   contactOrientationProcessVariance  z
    cf  contact_process                [6:8]   contactForceProcessVariance   x,y
    fz  contact_process                [8:9]   contactForceProcessVariance   z (normale)
    ct  contact_process                [9:11]  contactTorqueProcessVariance  x,y
    cz  contact_process                [11:12] contactTorqueProcessVariance  z (lacet)
    oi  contact_initial_new            [3:6]   contactOriInitVarianceNewContacts
    gb  gyro_bias_process              [0:3]   gyroBiasProcessVariance
    uf  unmodeled_wrench_process       [0:2]   unmodeledForceProcessVariance   x,y
    un  unmodeled_wrench_process       [2:3]   unmodeledForceProcessVariance   z (normale)
    ut  unmodeled_wrench_process       [3:5]   unmodeledTorqueProcessVariance  x,y
    uz  unmodeled_wrench_process       [5:6]   unmodeledTorqueProcessVariance  z (lacet)
Tout le reste vient de la config retenue. Chaque ecriture est relue et verifiee.
"""
import sys
from pathlib import Path
import yaml

sys.path.insert(0, '/home/arnaud/devel/src/data_manipulation_tools/scripts/paper_pipeline')
from make_overlays import retained_covariances

SLICES = {
    'av': ('state_angular_velocity_process', slice(0, 3)),
    'px': ('contact_process', slice(0, 2)),
    'oy': ('contact_process', slice(5, 6)),
    'cf': ('contact_process', slice(6, 8)),   # contactForceProcessVariance x,y
    'fz': ('contact_process', slice(8, 9)),   # contactForceProcessVariance z (normale)
    'ct': ('contact_process', slice(9, 11)),   # contactTorqueProcessVariance x,y
    'cz': ('contact_process', slice(11, 12)),  # contactTorqueProcessVariance z (lacet)
    'oi': ('contact_initial_new', slice(3, 6)),
    'gb': ('gyro_bias_process', slice(0, 3)),
    'uf': ('unmodeled_wrench_process', slice(0, 2)),   # unmodeledForceProcessVariance x,y
    'un': ('unmodeled_wrench_process', slice(2, 3)),   # unmodeledForceProcessVariance z (normale)
    'ut': ('unmodeled_wrench_process', slice(3, 5)),   # unmodeledTorqueProcessVariance x,y
    'uz': ('unmodeled_wrench_process', slice(5, 6)),   # unmodeledTorqueProcessVariance z (lacet)
}

target, *pairs = sys.argv[1:]
settings = dict(p.split('=', 1) for p in pairs)
unknown = set(settings) - set(SLICES)
if unknown:
    sys.exit(f"ABANDON: cles inconnues {sorted(unknown)}")

cov = retained_covariances()
ref = retained_covariances()
for key, raw in settings.items():
    field, sl = SLICES[key]
    cov[field][sl] = [float(raw)] * len(range(*sl.indices(len(cov[field]))))

Path(target).write_text(yaml.safe_dump(
    {'covariances': {k: [float(x) for x in v] for k, v in sorted(cov.items())}}, sort_keys=True))

back = yaml.safe_load(Path(target).read_text())['covariances']
for key, raw in settings.items():
    field, sl = SLICES[key]
    got = back[field][sl]
    if any(abs(g - float(raw)) > 1e-30 for g in got):
        sys.exit(f"ABANDON: {key} ({field}{sl}) = {got}, attendu {raw}")
# Aucune tranche non demandee ne doit avoir bouge.
touched = {}
for key in settings:
    field, sl = SLICES[key]
    touched.setdefault(field, set()).update(range(*sl.indices(len(back[field]))))
for field, vector in back.items():
    for i, value in enumerate(vector):
        if i not in touched.get(field, set()) and value != ref[field][i]:
            sys.exit(f"ABANDON: {field}[{i}] a bouge sans etre demande ({value} vs {ref[field][i]})")
print('ok')

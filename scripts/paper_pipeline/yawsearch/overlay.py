"""Ecrit l'overlay de covariances d'une combinaison, a partir de la config retenue.

Les quatre reglages balayes sont des tranches des vecteurs d'overlay (cf. MC_RTC_KEYS) :
    state_angular_velocity_process [0:3]   stateAngVelProcessVariance
    contact_process                [0:2]   contactPositionProcessVariance  x,y
    contact_process                [5:6]   contactOrientationProcessVariance  z (lacet)
    contact_initial_new            [3:6]   contactOriInitVarianceNewContacts
Tout le reste vient de la config retenue, donc une combinaison ne peut pas figer par megarde
un reglage perime -- c'est la raison d'etre de make_overlays.py, reprise ici.
"""
import sys
from pathlib import Path
import yaml

sys.path.insert(0, '/home/arnaud/devel/src/data_manipulation_tools/scripts/paper_pipeline')
from make_overlays import retained_covariances

angvel, oriyaw, posxy, oriinit, target = sys.argv[1:6]
cov = retained_covariances()
cov['state_angular_velocity_process'][0:3] = [float(angvel)] * 3
cov['contact_process'][0:2] = [float(posxy)] * 2
cov['contact_process'][5:6] = [float(oriyaw)]
cov['contact_initial_new'][3:6] = [float(oriinit)] * 3

overlay = {'covariances': {k: [float(x) for x in v] for k, v in sorted(cov.items())}}
Path(target).write_text(yaml.safe_dump(overlay, sort_keys=True))

# Relecture : on ne fait pas confiance a l'ecriture.
back = yaml.safe_load(Path(target).read_text())['covariances']
checks = [('state_angular_velocity_process', slice(0, 3), float(angvel)),
          ('contact_process', slice(0, 2), float(posxy)),
          ('contact_process', slice(5, 6), float(oriyaw)),
          ('contact_initial_new', slice(3, 6), float(oriinit))]
for field, sl, value in checks:
    got = back[field][sl]
    if any(abs(g - value) > 1e-30 for g in got):
        sys.exit(f"ABANDON: {field}{sl} = {got}, attendu {value}")
# Les tranches NON balayees doivent etre restees celles de la config retenue.
ref = retained_covariances()
if back['contact_process'][2] != ref['contact_process'][2] or back['contact_process'][3:5] != ref['contact_process'][3:5]:
    sys.exit("ABANDON: une tranche non balayee de contact_process a bouge")
print('ok')

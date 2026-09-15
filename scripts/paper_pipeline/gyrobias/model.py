#!/usr/bin/env python3
"""Derive de biais d'une centrale MEMS de quadrupede de milieu de gamme (Bosch BMI088).

Valeurs de la fiche technique Bosch, pas d'un ordre de grandeur invente :

    stabilite de biais            < 2 deg/h
    decalage a l'allumage (ZRO)   +- 1 deg/s        <- retire par la mise a zero au demarrage
    coefficient thermique du ZRO  +- 0.015 deg/s/K  <- LE terme dominant
    densite de bruit              0.014 deg/s/sqrt(Hz)

Le terme dominant sur une marche de 32 minutes n'est ni la marche aleatoire ni l'instabilite de
biais : c'est la TEMPERATURE. Le BMI088 ne s'auto-echauffe pas (15 mW), donc la montee vient du
chassis, et son profil est un premier ordre vers l'equilibre. ArduPilot exige au moins 0.5 K par
10 min pour qu'un etalonnage thermique aboutisse, et retient TMAX = 65 C pour les cartes chauffees.

Trois composantes, dans l'ordre de leur poids :

1. RAMPE THERMIQUE   dT(t) = DELTA_T * (1 - exp(-t/TAU)), biais = COEFF * dT * RESIDUAL
   RESIDUAL est le residu apres compensation thermique embarquee -- PX4 et ArduPilot la font
   toutes deux. Sans compensation le biais atteindrait 0.17 deg/s, ce qui noierait tout ; avec
   15 % de residu il atteint 0.026 deg/s. C'est une HYPOTHESE : je n'ai pas trouve de mesure
   publiee de montee en temperature sur un quadrupede.
2. INSTABILITE DE BIAIS  processus de Gauss-Markov d'ecart-type 2 deg/h et de constante de temps
   300 s. Ce n'est pas une marche aleatoire : le bruit en 1/f erre mais reste borne, ce que montre
   le plancher de la variance d'Allan.
3. MARCHE ALEATOIRE DE VITESSE, en residuel, 1 deg/h/sqrt(h).

Pas de biais d'allumage : le robot se met a zero a l'arret avant de partir.

La derive est generee sur l'AXE DU TEMPS, une fois, puis interpolee sur les horodatages de chaque
fichier : le bag du replay et HartleyInput.txt n'ont pas la meme cadence et les deux estimateurs
doivent voir la meme derive au meme instant.
"""
import numpy as np

DURATION = 2948.0            # s, la duree du log brut de LongWalk
GRID = 0.1                   # s, pas de generation
SEED = 20260915

# 1. thermique
COEFF = np.radians(0.015)    # rad/s par kelvin, fiche BMI088
DELTA_T = 12.0               # K de montee vers l'equilibre, chassis de robot
TAU = 900.0                  # s, constante de temps thermique
RESIDUAL = 0.15              # residu apres compensation thermique embarquee -- hypothese

# 2. instabilite de biais, Gauss-Markov
BIAS_INSTABILITY = np.radians(2.0 / 3600.0)   # 2 deg/h -> rad/s
TAU_BIAS = 300.0             # s

# 3. marche aleatoire de vitesse
RRW = np.radians(1.0 / 3600.0) / 60.0         # 1 deg/h/sqrt(h) -> rad/s/sqrt(s)

# Les trois axes ne chauffent pas identiquement ; la fiche donne +- sur le coefficient.
AXIS_GAIN = np.array([1.0, -0.6, 0.85])


def trajectory(seed=SEED):
    """b(t) sur une grille reguliere : (temps, biais 3 axes) en rad/s."""
    n = int(DURATION / GRID) + 1
    t = np.arange(n) * GRID
    rng = np.random.default_rng(seed)

    thermal = COEFF * DELTA_T * RESIDUAL * (1.0 - np.exp(-t / TAU))
    b = thermal[:, None] * AXIS_GAIN[None, :]

    # Gauss-Markov : x_{k+1} = a x_k + w, a = exp(-dt/tau), var(w) tel que var(x) = sigma^2.
    a = np.exp(-GRID / TAU_BIAS)
    sigma_w = BIAS_INSTABILITY * np.sqrt(1.0 - a * a)
    flicker = np.zeros((n, 3))
    noise = rng.normal(0.0, sigma_w, (n, 3))
    flicker[0] = rng.normal(0.0, BIAS_INSTABILITY, 3)
    for k in range(1, n):
        flicker[k] = a * flicker[k - 1] + noise[k]
    b += flicker

    b += np.cumsum(rng.normal(0.0, RRW * np.sqrt(GRID), (n, 3)), axis=0)
    return t, b


def at(times, seed=SEED):
    """Le biais aux instants demandes, interpole depuis la meme realisation."""
    t, b = trajectory(seed)
    return np.column_stack([np.interp(times, t, b[:, axis]) for axis in range(3)])


if __name__ == '__main__':
    t, b = trajectory()
    print(f'derive de biais BMI088, graine {SEED}, {len(t)} points sur {DURATION:.0f} s\n')
    print(f"{'axe':5s} {'depart':>10s} {'a 1002 s':>10s} {'fin':>10s} {'amplitude':>10s}   (deg/s)")
    k1 = np.searchsorted(t, 1002.28)
    for axis, name in enumerate('xyz'):
        d = np.degrees(b[:, axis])
        print(f'{name:5s} {d[0]:10.5f} {d[k1]:10.5f} {d[-1]:10.5f} {d.max()-d.min():10.5f}')
    yaw = np.degrees(b[k1:, 2]).mean() * (DURATION - 1002.28)
    print(f'\nlacet integre sur la fenetre evaluee : {yaw:+.1f} deg')
    print(f'sans compensation thermique il serait de {yaw/RESIDUAL:+.0f} deg')

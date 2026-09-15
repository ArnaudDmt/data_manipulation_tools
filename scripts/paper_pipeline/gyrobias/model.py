#!/usr/bin/env python3
"""Derive de biais d'une centrale inertielle standard, du type monte sur un quadrupede.

HRP5-P porte un gyro a fibre optique : son biais mesure 0.0003 deg/s sur l'axe de lacet aux deux
fenetres ou la verite est calculable, autrement dit il ne derive pas. Une centrale MEMS
industrielle -- Bosch BMI088, ADIS16470, ce qu'on trouve sur un ANYmal ou un Unitree -- derive de
deux a trois ordres de grandeur plus.

Le modele :
    b(t) = b0 + marche aleatoire d'ecart-type SIGMA_RW * sqrt(t)
avec SIGMA_RW cale pour que l'ecart-type de la derive atteigne DRIFT_1946 sur la duree de
LongWalk. C'est la composante que le gyro a fibre optique n'a pas, et c'est celle que l'etat de
biais des deux estimateurs doit rattraper.

La derive est generee sur l'AXE DU TEMPS, une fois, puis interpolee sur les horodatages de chaque
fichier : le bag du replay et HartleyInput.txt n'ont pas la meme cadence, et les deux estimateurs
doivent voir la meme derive au meme instant, sinon la comparaison ne vaut rien.
"""
import numpy as np

DURATION = 2948.0          # s, la duree du log brut de LongWalk
DRIFT_1946 = np.radians(0.03)   # rad/s, ecart-type de la derive sur la fenetre evaluee (1946 s)
B0 = np.radians(np.array([0.010, -0.008, 0.012]))   # biais d'allumage residuel, rad/s
SEED = 20260915
GRID = 0.1                 # s, pas de la grille de generation


def trajectory(seed=SEED):
    """b(t) sur une grille reguliere : (temps, biais 3 axes)."""
    n = int(DURATION / GRID) + 1
    t = np.arange(n) * GRID
    sigma_step = DRIFT_1946 / np.sqrt(1946.0) * np.sqrt(GRID)
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0.0, sigma_step, (n, 3)), axis=0)
    return t, B0 + walk


def at(times, seed=SEED):
    """Le biais aux instants demandes, interpole depuis la meme realisation."""
    t, b = trajectory(seed)
    return np.column_stack([np.interp(times, t, b[:, axis]) for axis in range(3)])


if __name__ == '__main__':
    t, b = trajectory()
    print(f'derive de biais, graine {SEED}, {len(t)} points sur {DURATION:.0f} s')
    for axis, name in enumerate('xyz'):
        deg = np.degrees(b[:, axis])
        print(f'  axe {name} : depart {deg[0]:+.4f}  fin {deg[-1]:+.4f}  '
              f'amplitude {deg.max()-deg.min():.4f} deg/s')
    k = np.searchsorted(t, 1946.0)
    print(f'\n  a 1946 s (fin de la fenetre evaluee) : '
          f'{np.degrees(b[k]).round(4)} deg/s')
    print(f'  a comparer au gyro a fibre optique de HRP5-P : 0.0003 deg/s mesure')
    yaw = np.degrees(b[:k, 2]).mean() * 1946.0
    print(f'  lacet que ce biais integre sur la fenetre : {yaw:+.1f} deg')

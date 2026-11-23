"""Validation commune de la plaque, de la carte et des positions."""

import numpy as np


def validate(c):
    if not (c.NX >= 2 and c.NY >= 2 and c.PLATE_LAYERS >= 1 and c.MAX_HEIGHT >= 1):
        raise ValueError("Maillage et nombre de couches invalides.")
    if not (0 <= c.LENS_X[0] < c.LENS_X[1] <= c.NX and
            0 <= c.LENS_Y[0] < c.LENS_Y[1] <= c.NY):
        raise ValueError("La zone de lentille doit être dans le maillage.")
    if not (isinstance(c.INITIAL_HEIGHT, int) and 0 <= c.INITIAL_HEIGHT <= c.MAX_HEIGHT):
        raise ValueError("Hauteur initiale hors limites.")
    values = (c.LENGTH, c.WIDTH, c.THICKNESS, c.YOUNG, c.DENSITY,
              c.STUB_DZ, c.STUB_YOUNG, c.STUB_DENSITY, c.FREQUENCY)
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("Les dimensions, matériaux et la fréquence doivent être positifs et finis.")
    if not (-1 < c.POISSON < .5 and -1 < c.STUB_POISSON < .5):
        raise ValueError("Coefficients de Poisson invalides.")
    if c.ITERATIONS < 0 or c.CANDIDATES_PER_ITERATION < 1 or c.MUTATION_PIXELS < 1:
        raise ValueError("Paramètres de recherche invalides.")
    if c.FOCUS[0] >= c.LENS_X[0]*c.LENGTH/c.NX or c.RETURN_SOURCE[0] <= c.LENS_X[1]*c.LENGTH/c.NX:
        raise ValueError("Placer le foyer à gauche et la source retour à droite de la lentille.")
    if not c.SENSORS:
        raise ValueError("Au moins un capteur est nécessaire.")
    for point in (c.RETURN_SOURCE, c.FOCUS, *c.SENSORS):
        node(c, point)


def node(c, point):
    x, y = point
    if not np.isfinite((x, y)).all() or not (0 <= x <= c.LENGTH and 0 <= y <= c.WIDTH):
        raise ValueError("Source, foyer et capteurs doivent être dans la plaque.")
    ix = int(np.rint(x*c.NX/c.LENGTH))
    iy = int(np.rint(y*c.NY/c.WIDTH))
    return iy*(c.NX+1)+ix, (ix*c.LENGTH/c.NX, iy*c.WIDTH/c.NY)


def geometry(c, design):
    shape = (c.LENS_Y[1]-c.LENS_Y[0], c.LENS_X[1]-c.LENS_X[0])
    d = np.asarray(design)
    if d.shape != shape or not np.isfinite(d).all() or np.any(d != np.round(d)):
        raise ValueError("Carte de hauteurs incompatible.")
    if np.any(d < 0) or np.any(d > c.MAX_HEIGHT):
        raise ValueError("Hauteur de plot hors limites.")
    if c.SYMMETRY_X and not np.array_equal(d, d[:, ::-1]):
        raise ValueError("Symétrie x non respectée.")
    if c.SYMMETRY_Y and not np.array_equal(d, d[::-1, :]):
        raise ValueError("Symétrie y non respectée.")
    return d.astype(int)

"""Nœuds et pondération de la ligne focale commune aux modèles 2D et 3D."""

import numpy as np

from reduced import node


def focal_line(c):
    if not np.isfinite(c.FOCAL_SIGMA) or c.FOCAL_SIGMA <= 0:
        raise ValueError("FOCAL_SIGMA doit être positif et fini (en mètres).")
    _, snapped = node(c, (c.FOCUS[0], c.WIDTH/2))
    y = np.linspace(0, c.WIDTH, c.NY+1)
    if not np.isfinite(y).all() or len(y) < 3 or c.WIDTH <= 0:
        raise ValueError("Largeur ou maillage de la ligne focale invalide.")
    # Subtract the smallest squared distance before division to avoid total underflow.
    distance = np.abs(y-c.WIDTH/2)
    relative = (distance-distance.min())*(distance+distance.min())
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(-0.5*(relative/c.FOCAL_SIGMA)/c.FOCAL_SIGMA)
    weights[[0, -1]] *= .5
    weights /= weights.sum()
    indices = np.arange(c.NY+1)*(c.NX+1)+int(np.rint(snapped[0]*c.NX/c.LENGTH))
    return snapped[0], y, indices, weights

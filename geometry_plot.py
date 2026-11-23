"""Représentation de la plaque et des hauteurs réelles de la métasurface."""

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from reduced import geometry


def plot_geometry_3d(path, c, design):
    heights = geometry(c, design)
    if not np.isfinite(c.GEOMETRY_Z_SCALE) or c.GEOMETRY_Z_SCALE <= 0:
        raise ValueError("GEOMETRY_Z_SCALE doit être positif et fini.")
    dx, dy = c.LENGTH/c.NX*1000, c.WIDTH/c.NY*1000
    scale = c.GEOMETRY_Z_SCALE
    layer = c.STUB_DZ*1000
    top = heights*layer*scale
    vertices, face_colors = [], []
    cmap = plt.get_cmap("viridis")
    norm = colors.Normalize(vmin=0, vmax=c.MAX_HEIGHT*layer)
    ny, nx = heights.shape
    for y, x in zip(*np.nonzero(heights)):
        x0 = (c.LENS_X[0]+x)*dx
        y0 = (c.LENS_Y[0]+y)*dy
        x1, y1 = x0+dx, y0+dy
        z = top[y, x]
        color = cmap(norm(heights[y, x]*layer))
        vertices.append([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)])
        face_colors.append(color)
        vertices.append([(x0, y0, -z), (x1, y0, -z), (x1, y1, -z), (x0, y1, -z)])
        face_colors.append(color)
        for xx, yy, edge in ((x-1, y, ((x0, y0), (x0, y1))),
                             (x+1, y, ((x1, y0), (x1, y1))),
                             (x, y-1, ((x0, y0), (x1, y0))),
                             (x, y+1, ((x0, y1), (x1, y1)))):
            neighbour = top[yy, xx] if 0 <= xx < nx and 0 <= yy < ny else 0
            if neighbour < z:
                a, b = edge
                vertices.append([(a[0], a[1], neighbour), (b[0], b[1], neighbour),
                                 (b[0], b[1], z), (a[0], a[1], z)])
                face_colors.append(color)
                vertices.append([(a[0], a[1], -neighbour), (b[0], b[1], -neighbour),
                                 (b[0], b[1], -z), (a[0], a[1], -z)])
                face_colors.append(color)

    fig = plt.figure(figsize=(11, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    plate = [[(0, 0, 0), (c.LENGTH*1000, 0, 0),
              (c.LENGTH*1000, c.WIDTH*1000, 0), (0, c.WIDTH*1000, 0)]]
    ax.add_collection3d(Poly3DCollection(plate, facecolors="lightsteelblue",
                                       edgecolors="steelblue", alpha=.35))
    if vertices:
        ax.add_collection3d(Poly3DCollection(vertices, facecolors=face_colors,
                                           edgecolors="black", linewidths=.15))
    ax.set_xlim(0, c.LENGTH*1000)
    ax.set_ylim(0, c.WIDTH*1000)
    max_z = c.MAX_HEIGHT*layer*scale
    ax.set_zlim(-max_z*1.1, max_z*1.1)
    ax.set_box_aspect((c.LENGTH*1000, c.WIDTH*1000, 2*max_z))
    ax.zaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/scale:g}"))
    ax.set(xlabel="x (mm)", ylabel="y (mm)", zlabel="Hauteur relative (mm)",
           title=f"Deux faces symétriques — hauteur amplifiée ×{scale:g}")
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=.65,
                 pad=.1, label="Hauteur réelle du plot (mm)")
    fig.savefig(path, dpi=160)
    plt.close(fig)

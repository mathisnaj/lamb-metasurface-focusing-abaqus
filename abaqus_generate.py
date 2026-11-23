"""Generate Abaqus/Explicit models from the V4 configuration and optional design."""

import argparse
import json
from pathlib import Path

import numpy as np

import config as c
from focal_line import focal_line
from main import signature
from reduced import geometry, node, validate


def build_model(design):
    """Return full-grid node labels and C3D8R connectivity for occupied elements."""
    nx, ny = c.NX, c.NY
    layer_size = (nx + 1)*(ny + 1)
    bottom, top = c.MAX_HEIGHT, c.MAX_HEIGHT+c.PLATE_LAYERS
    z_values = ([(-c.THICKNESS/2)-(bottom-z)*c.STUB_DZ for z in range(bottom)]
                + [-c.THICKNESS/2+z*c.THICKNESS/c.PLATE_LAYERS
                   for z in range(c.PLATE_LAYERS+1)]
                + [c.THICKNESS/2+z*c.STUB_DZ for z in range(1, c.MAX_HEIGHT+1)])

    def label(x, y, z):
        return z*layer_size+y*(nx+1)+x+1

    nodes, plate_elements, stub_elements = set(), [], []

    def add(x, y, z, group):
        ids = (label(x, y, z), label(x+1, y, z), label(x+1, y+1, z),
               label(x, y+1, z), label(x, y, z+1), label(x+1, y, z+1),
               label(x+1, y+1, z+1), label(x, y+1, z+1))
        nodes.update(ids)
        group.append(ids)

    for z in range(bottom, top):
        for y in range(ny):
            for x in range(nx):
                add(x, y, z, plate_elements)
    for iy, ix in np.ndindex(design.shape):
        y, x = iy+c.LENS_Y[0], ix+c.LENS_X[0]
        for z in range(int(design[iy, ix])):
            add(x, y, top+z, stub_elements)
            add(x, y, bottom-1-z, stub_elements)
    return nodes, plate_elements, stub_elements, z_values


def write_deck(path, design, line_indices):
    nodes, plate_elements, stub_elements, z_values = build_model(design)
    nx, ny = c.NX, c.NY
    layer_size = (nx+1)*(ny+1)
    source, _ = node(c, c.RETURN_SOURCE)
    top = c.MAX_HEIGHT+c.PLATE_LAYERS
    source_top = top*layer_size+source+1
    source_bottom = c.MAX_HEIGHT*layer_size+source+1
    line_ids = [top*layer_size+int(i)+1 for i in line_indices]
    with path.open("w", encoding="ascii") as f:
        f.write("*Heading\nV4 elastic plate and columns: SI units, Abaqus/Explicit\n")
        f.write("*Preprint, echo=NO, model=NO, history=NO, contact=NO\n")
        f.write("*Part, name=PLATE\n*Node\n")
        for label in sorted(nodes):
            z, remainder = divmod(label-1, layer_size)
            y, x = divmod(remainder, nx+1)
            f.write("{}, {:.16g}, {:.16g}, {:.16g}\n".format(
                label, x*c.LENGTH/nx, y*c.WIDTH/ny, z_values[z]))
        number = 1
        for name, elements in (("PLATE_E", plate_elements), ("STUB_E", stub_elements)):
            if not elements:
                continue
            f.write("*Element, type=C3D8R, elset={}\n".format(name))
            for ids in elements:
                f.write("{}, {}\n".format(number, ", ".join(map(str, ids))))
                number += 1
            f.write("*Solid Section, elset={}, material={}\n,\n".format(
                name, "PLATE_M" if name == "PLATE_E" else "STUB_M"))
        f.write("*End Part\n*Assembly, name=ASSEMBLY\n"
                "*Instance, name=PLATE-1, part=PLATE\n*End Instance\n")
        for name, label in [("SRC_TOP", source_top), ("SRC_BOTTOM", source_bottom)]:
            f.write("*Nset, nset={}, instance=PLATE-1\n{}\n".format(name, label))
        for i, label in enumerate(line_ids):
            f.write("*Nset, nset=LINE_{:04d}, instance=PLATE-1\n{}\n".format(i, label))
        f.write("*End Assembly\n")
        for name, young, poisson, density in (
            ("PLATE_M", c.YOUNG, c.POISSON, c.DENSITY),
            ("STUB_M", c.STUB_YOUNG, c.STUB_POISSON, c.STUB_DENSITY),
        ):
            f.write("*Material, name={}\n*Density\n{:.16g}\n"
                    "*Elastic\n{:.16g}, {:.16g}\n".format(name, density, young, poisson))
        f.write("*Amplitude, name=BURST, definition=TABULAR\n")
        duration = c.CYCLES/c.FREQUENCY
        for t in np.linspace(0, duration, max(2, int(np.ceil(c.CYCLES*100))+1)):
            amplitude = np.sin(2*np.pi*c.FREQUENCY*t)*np.sin(np.pi*t/duration)**2
            f.write("{:.16g}, {:.16g}\n".format(t, amplitude))
        f.write("*Step, name=PROPAGATION, nlgeom=NO\n*Dynamic, Explicit\n"
                ", {:.16g}\n*Cload, amplitude=BURST\n".format(c.DURATION))
        if c.SOURCE_MODE == "top":
            f.write("SRC_TOP, 3, {:.16g}\n".format(c.SOURCE_FORCE))
        else:
            f.write("SRC_TOP, 3, {:.16g}\nSRC_BOTTOM, 3, {:.16g}\n".format(
                c.SOURCE_FORCE/2, c.SOURCE_FORCE/2 if c.SOURCE_MODE == "A"
                else -c.SOURCE_FORCE/2))
        f.write("*Output, field, number interval=10\n*Node Output\nU\n"
                "*Output, history, time interval={:.16g}\n".format(c.ABAQUS_HISTORY_DT))
        for i in range(len(line_ids)):
            f.write("*Node Output, nset=LINE_{:04d}\nU3\n".format(i))
        f.write("*Energy Output\nALLAE, ALLIE, ALLKE, ETOTAL\n")
        f.write("*End Step\n")
    return line_ids, len(nodes), len(plate_elements)+len(stub_elements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--design", type=Path,
                        help="design.npz produit par la V4, avec une signature compatible")
    args = parser.parse_args()
    validate(c)
    if c.SOURCE_MODE not in ("top", "A", "S"):
        raise ValueError("SOURCE_MODE doit valoir 'top', 'A' ou 'S'.")
    if not np.isfinite((c.DURATION, c.CYCLES, c.SOURCE_FORCE, c.ABAQUS_HISTORY_DT)).all() or (
        c.DURATION <= 0 or c.CYCLES <= 0 or c.SOURCE_FORCE == 0 or c.ABAQUS_HISTORY_DT <= 0
        or not 0 <= c.FOCUS_WINDOW[0] < c.FOCUS_WINDOW[1] <= c.DURATION
    ):
        raise ValueError("Durée, source ou fenêtre temporelle invalide.")
    if args.design:
        with np.load(args.design, allow_pickle=False) as data:
            if str(data["signature"].item()) != json.dumps(signature(c), sort_keys=True):
                raise ValueError("Le design V4 ne correspond pas à la configuration actuelle.")
            design = geometry(c, data["heights"])
    else:
        design = None
    args.output.mkdir(parents=True, exist_ok=True)
    line_x, line_y, indices, weights = focal_line(c)
    shape = (c.LENS_Y[1]-c.LENS_Y[0], c.LENS_X[1]-c.LENS_X[0])
    line_ids, n, e = write_deck(args.output/"bare.inp", np.zeros(shape, dtype=int), indices)
    print("Plaque nue : {} nœuds, {} éléments".format(n, e))
    if design is not None:
        ids2, n, e = write_deck(args.output/"motifs.inp", design, indices)
        if ids2 != line_ids:
            raise RuntimeError("Les nœuds de mesure diffèrent entre modèles.")
        print("Plaque avec motifs : {} nœuds, {} éléments".format(n, e))
    manifest = {"instance": "PLATE-1", "step": "PROPAGATION",
                "line_node_labels": line_ids, "line_x_m": line_x,
                "line_y_m": line_y.tolist(), "weights": weights.tolist(),
                "window_s": list(c.FOCUS_WINDOW), "duration_s": c.DURATION,
                "history_interval_s": c.ABAQUS_HISTORY_DT, "has_design": design is not None,
                "signature": signature(c)}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("Fichiers Abaqus créés dans {}".format(args.output))


if __name__ == "__main__":
    main()

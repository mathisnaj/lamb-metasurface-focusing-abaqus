"""Compare focal-line histories using their actual Abaqus output times."""

import argparse
import json
from pathlib import Path

import numpy as np


def energy(history, window):
    a = np.asarray(history, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError("Historique U3 invalide.")
    times, displacement = a[:, 0], a[:, 1]
    if np.any(np.diff(times) <= 0):
        raise ValueError("Les instants de sortie doivent être strictement croissants.")
    start, end = window
    tolerance = max(1e-12, abs(end)*1e-7)
    if times[0] > start+tolerance or times[-1] < end-tolerance:
        raise ValueError("Historique incomplet sur la fenêtre demandée.")
    if times[0] > start:
        times = np.r_[start, times]
        displacement = np.r_[displacement[0], displacement]
    if times[-1] < end:
        times = np.r_[times, end]
        displacement = np.r_[displacement, displacement[-1]]
    t = np.r_[start, times[(times > start) & (times < end)], end]
    u = np.interp(t, times, displacement)
    return float(np.sum(np.diff(t)*(u[:-1]**2+u[:-1]*u[1:]+u[1:]**2)/3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    path = args.directory
    manifest = json.loads((path/"manifest.json").read_text())
    ref = json.loads((path/"bare_history.json").read_text())
    design = json.loads((path/"motifs_history.json").read_text())
    ids = manifest["line_node_labels"]
    if (ref["line_node_labels"] != ids or design["line_node_labels"] != ids
        or len(ref["histories"]) != len(ids) or len(design["histories"]) != len(ids)):
        raise ValueError("Historiques de ligne incompatibles avec le manifeste.")
    reference = np.array([energy(h, manifest["window_s"]) for h in ref["histories"]])
    patterned = np.array([energy(h, manifest["window_s"]) for h in design["histories"]])
    weights = np.array(manifest["weights"])
    if len(weights) != len(ids) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Poids de ligne invalides.")
    denom = float(weights@reference)
    if not np.isfinite(denom) or denom <= np.finfo(float).tiny:
        raise ValueError("Référence sur la ligne nulle ou non finie.")
    ratio = float(weights@patterned/denom)
    if not np.isfinite(ratio):
        raise ValueError("Rapport sur la ligne non fini.")
    result = {"rapport_uz2_ligne_ponderee": ratio,
              "ligne_y_m": manifest["line_y_m"],
              "poids": weights.tolist(), "reference_m2_s": reference.tolist(),
              "motifs_m2_s": patterned.tolist(),
              "fenetre_s": manifest["window_s"],
              "integration": "carré du signal linéaire par morceaux"}
    (path/"comparaison.json").write_text(json.dumps(result, indent=2)+"\n")
    print("Rapport Abaqus pondéré sur la ligne : {:.6g}".format(ratio))


if __name__ == "__main__":
    main()

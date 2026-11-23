# Compatible with the Python distributed with Abaqus 2023.
"""Extract U3 history at each focal-line node from an Abaqus ODB."""

from __future__ import print_function

import json
import re
import sys

from odbAccess import openOdb


def main():
    if len(sys.argv) != 4:
        raise SystemExit("Usage: abaqus python abaqus_extract.py job.odb manifest.json sortie.json")
    odb_path, manifest_path, output_path = sys.argv[1:]
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    instance = manifest["instance"].upper()
    wanted = set(manifest["line_node_labels"])
    found = {}
    odb = openOdb(path=odb_path, readOnly=True)
    try:
        step = odb.steps[manifest["step"]]
        for name, region in step.historyRegions.items():
            match = re.match(r"^Node\s+(.+)\.(\d+)$", name, re.IGNORECASE)
            if not match or match.group(1).upper() != instance:
                continue
            label = int(match.group(2))
            if label not in wanted:
                continue
            if label in found:
                raise ValueError("Plusieurs regions d'historique pour le noeud %d" % label)
            if "U3" not in region.historyOutputs:
                raise ValueError("U3 absent pour le noeud %d" % label)
            data = region.historyOutputs["U3"].data
            if len(data) < 2:
                raise ValueError("Historique trop court pour le noeud %d" % label)
            found[label] = [[float(t), float(u)] for t, u in data]
        missing = wanted-set(found)
        if missing:
            raise ValueError("Noeuds de ligne absents de l'ODB : %s" % sorted(missing))
        result = {"line_node_labels": manifest["line_node_labels"],
                  "histories": [found[n] for n in manifest["line_node_labels"]]}
        with open(output_path, "w") as f:
            json.dump(result, f, allow_nan=False)
        print("Historiques U3 extraits : %d noeuds" % len(found))
    finally:
        odb.close()


if __name__ == "__main__":
    main()

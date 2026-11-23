"""V4 : optimisation de géométrie par FEM élastique transitoire et adjoint."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import config as cfg
from focal_line import focal_line
from geometry_plot import plot_geometry_3d
from inverse_design import optimize
from phase_design import a0_wavenumber


def signature(c):
    names = ("LENGTH", "WIDTH", "THICKNESS", "NX", "NY", "PLATE_LAYERS",
             "YOUNG", "POISSON", "DENSITY", "STUB_YOUNG", "STUB_POISSON",
             "STUB_DENSITY", "STUB_DZ", "MAX_HEIGHT", "LENS_X", "LENS_Y",
             "SYMMETRY_X", "SYMMETRY_Y", "FREQUENCY", "RETURN_SOURCE", "FOCUS",
             "FOCAL_SIGMA", "SOURCE_MODE", "SOURCE_FORCE", "CYCLES", "DURATION",
             "FOCUS_WINDOW")
    return {"modele": "elastique_transitoire_ligne_v4",
            **{name: getattr(c, name) for name in names}}


def main():
    output = Path(cfg.OUTPUT_DIRECTORY)
    output.mkdir(parents=True, exist_ok=True)
    design, plate, history, gain, score, periods, step, device, state = optimize(cfg)
    points = 2*np.pi/a0_wavenumber(cfg)/(cfg.LENGTH/cfg.NX)
    if points < 10:
        print(f"Attention : {points:.1f} éléments par longueur d'onde A0 selon x ; "
              "affiner le maillage et vérifier la convergence.")
    if periods is None:
        print("Attention : aucune géométrie entière évaluée n'améliore l'objectif ; "
              "la plaque nue est conservée.")
    x, y, _, weights = focal_line(cfg)
    parameters = signature(cfg)
    np.savez_compressed(output/"design.npz", heights=design, history=history,
                        signature=json.dumps(parameters, sort_keys=True),
                        line_x_m=x, line_y_m=y, line_weights=weights, dt_s=plate.dt)
    network_path = output/"reseau.pt"
    if state is not None:
        torch.save({"state_dict": state, "hidden": cfg.NN_HIDDEN,
                    "period_x_range": cfg.V2_PERIOD_X_RANGE,
                    "period_y_range": cfg.V2_PERIOD_Y_RANGE,
                    "configuration": parameters}, network_path)
    elif network_path.exists():
        network_path.unlink()
    options = {name: getattr(cfg, name) for name in (
        "V4_STEPS", "V4_LEARNING_RATE", "V4_HEIGHT_PENALTY", "V4_DISCRETE_EVERY",
        "V4_VOID_FRACTION", "V4_CHECKPOINT_INTERVAL", "V2_PERIOD_X_RANGE",
        "V2_PERIOD_Y_RANGE", "V2_INITIAL_PERIOD_X", "V2_INITIAL_PERIOD_Y",
        "V2_GATE_SHARPNESS", "V2_FILL_FRACTION", "V2_INITIAL_FILL",
        "NN_HIDDEN", "NN_DEVICE", "SEED")}
    result = {"methode": "reseau_geometrie_adjoint_elastique_transitoire_v4",
              "gain_elastique_discret": gain, "objectif_penalise": score,
              "etape_retenue": step, "periodes_nominales_elements": periods,
              "periodes_nominales_m": ([periods[0]*cfg.LENGTH/cfg.NX,
                                         periods[1]*cfg.WIDTH/cfg.NY]
                                        if periods is not None else None),
              "plaque_nue_retenue": periods is None, "appareil_pytorch": device,
              "dt_s": plate.dt, "reference_uz2_ligne": plate.reference,
              "elements_par_longueur_onde_x": points,
              "ligne_focale_x_m": x, "ligne_focale_sigma_m": cfg.FOCAL_SIGMA,
              "parametres_modele": parameters, "parametres_optimisation": options}
    (output/"optimisation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    fig, ax = plt.subplots()
    ax.plot(history[:, 0], history[:, 1], label="Gain relaxé")
    ax.plot(history[:, 0], history[:, 3], label="Gain entier")
    ax.plot(history[:, 0], history[:, 4], label="Objectif entier")
    ax.axhline(1, linestyle="--", color="gray", label="Plaque nue")
    ax.set(xlabel="Étape", ylabel="Valeur sans dimension", title="Optimisation élastique 3D")
    ax.legend()
    fig.savefig(output/"convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    plot_geometry_3d(output/"metasurface_3d.png", cfg, design)
    print(f"Gain élastique entier pondéré : {gain:.3f} ; objectif {score:.3f}. Résultats : {output}")


if __name__ == "__main__":
    main()

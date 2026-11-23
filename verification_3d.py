"""Vérification élastique 3D de la carte optimisée, sans arguments."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
import numpy as np

import config as cfg
from elastic3d import Plate3D
from elastic_adjoint import common_timestep
from focal_line import focal_line
from geometry_plot import plot_geometry_3d
from main import signature
from reduced import geometry, node, validate


def load_design(path):
    with np.load(path, allow_pickle=False) as data:
        recorded = str(data["signature"].item())
        current = json.dumps(signature(cfg), sort_keys=True)
        if recorded != current:
            raise ValueError("Paramètres physiques ou géométriques modifiés depuis main.py : "
                             "relancer l'optimisation avant la vérification.")
        return geometry(cfg, data["heights"])


def figures(output, design, times, reference, filtered, frame_times, movies, source, sensors):
    extent = [0, cfg.LENGTH*1000, 0, cfg.WIDTH*1000]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.imshow(np.zeros((cfg.NY, cfg.NX)), origin="lower", extent=extent,
              vmin=0, vmax=1, cmap="Greys", alpha=.2)
    ax.imshow(design, origin="lower", cmap="magma", vmin=0, vmax=cfg.MAX_HEIGHT,
              extent=[cfg.LENS_X[0]*cfg.LENGTH/cfg.NX*1000,
                      cfg.LENS_X[1]*cfg.LENGTH/cfg.NX*1000,
                      cfg.LENS_Y[0]*cfg.WIDTH/cfg.NY*1000,
                      cfg.LENS_Y[1]*cfg.WIDTH/cfg.NY*1000])
    ax.scatter(*np.array(source)*1000, marker="*", color="red", s=150, label="Source retour")
    _, focus = node(cfg, cfg.FOCUS)
    ax.scatter(*np.array(focus)*1000, marker="x", color="blue", s=80, label="Foyer")
    line_x, _, _, _ = focal_line(cfg)
    ax.axvline(line_x*1000, color="blue", linestyle="--", label="Ligne focale")
    for i, p in enumerate(sensors):
        ax.scatter(*np.array(p)*1000, facecolors="none", edgecolors="cyan", s=100, linewidths=1.5)
        ax.annotate(str(i+1), np.array(p)*1000, color="blue")
    ax.set(xlabel="x (mm)", ylabel="y (mm)", title="Plaque, hauteur des plots et capteurs")
    ax.legend(loc="upper right")
    fig.savefig(output/"geometrie.png", dpi=150)
    plt.close(fig)
    plot_geometry_3d(output/"metasurface_3d.png", cfg, design)

    fig, axes = plt.subplots(len(sensors), 1, figsize=(10, max(3, 2.2*len(sensors))),
                             squeeze=False, constrained_layout=True)
    for i, (ax, p) in enumerate(zip(axes[:, 0], sensors)):
        for component, color in zip(range(3), ("tab:blue", "tab:orange", "tab:green")):
            label = ("u_x", "u_y", "u_z")[component]
            ax.plot(times*1e6, filtered[:, i, component]*1e9, color=color, label=label+" avec")
            ax.plot(times*1e6, reference[:, i, component]*1e9, color=color,
                    linestyle="--", alpha=.65, label=label+" sans")
        ax.set(ylabel="u (nm)", title=f"Capteur {i+1} : x={p[0]*1000:.1f}, y={p[1]*1000:.1f} mm")
        ax.grid(alpha=.25)
    axes[-1, 0].set_xlabel("Temps (µs)")
    axes[0, 0].legend(ncol=3, fontsize=8)
    fig.savefig(output/"capteurs.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    vmax = float(np.max(np.abs(movies[..., 2])))
    if vmax == 0:
        vmax = 1.0
    artists = []
    for ax, movie, title in zip(axes, movies, ("Plaque nue", "Plaque avec plots")):
        image = ax.imshow(movie[0, ..., 2]*1e9, origin="lower", extent=extent,
                          cmap="RdBu_r", vmin=-vmax*1e9, vmax=vmax*1e9)
        artists.append(image)
        ax.scatter(*np.array(source)*1000, marker="*", c="black", s=80)
        ax.scatter(*np.array(focus)*1000, marker="x", c="black", s=60)
        ax.axvline(line_x*1000, color="black", linestyle=":", linewidth=1)
        for i, p in enumerate(sensors):
            ax.scatter(*np.array(p)*1000, facecolors="none", edgecolors="black", s=50)
            ax.annotate(str(i+1), np.array(p)*1000, color="black", fontsize=8)
        ax.add_patch(Rectangle((cfg.LENS_X[0]*cfg.LENGTH/cfg.NX*1000,
                                cfg.LENS_Y[0]*cfg.WIDTH/cfg.NY*1000),
                               (cfg.LENS_X[1]-cfg.LENS_X[0])*cfg.LENGTH/cfg.NX*1000,
                               (cfg.LENS_Y[1]-cfg.LENS_Y[0])*cfg.WIDTH/cfg.NY*1000,
                               fill=False, edgecolor="black", linestyle="--"))
        ax.set(xlabel="x (mm)", ylabel="y (mm)", title=title)
    fig.colorbar(artists[0], ax=axes, label="Déplacement vertical u_z (nm)")
    title = fig.suptitle("")

    def update(index):
        for image, movie in zip(artists, movies):
            image.set_data(movie[index, ..., 2]*1e9)
        title.set_text(f"Face supérieure : t = {frame_times[index]*1e6:.1f} µs")
        return *artists, title

    animation = FuncAnimation(fig, update, frames=len(frame_times), interval=1000/cfg.FPS, blit=False)
    animation.save(output/"ondes_3d.gif", writer=PillowWriter(fps=cfg.FPS))
    if cfg.SHOW:
        plt.show()
    plt.close(fig)


def plot_line(output, y, weights, ref_energy, filtered_energy):
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(y*1000, ref_energy, label="Plaque nue")
    ax.plot(y*1000, filtered_energy, label="Avec motifs")
    ax.set(xlabel="y (mm)", ylabel="Intégrale temporelle de u_z² (m² s)",
           title="Réponse sur la ligne focale")
    ax.legend(loc="upper left")
    other = ax.twinx()
    other.plot(y*1000, weights, "k--", alpha=.6, label="Poids")
    other.set_ylabel("Poids normalisé par nœud")
    other.legend(loc="upper right")
    fig.savefig(output/"ligne_focale_3d.png", dpi=150)
    plt.close(fig)


def main():
    output = Path(cfg.OUTPUT_DIRECTORY)
    design = load_design(output/"design.npz")
    validate(cfg)
    if cfg.FPS <= 0 or not (0 <= cfg.FOCUS_WINDOW[0] < cfg.FOCUS_WINDOW[1] <= cfg.DURATION):
        raise ValueError("FPS ou fenêtre FOCUS_WINDOW invalide.")
    reference_plate = Plate3D(cfg, np.zeros_like(design))
    filtered_plate = Plate3D(cfg, design)
    dt = common_timestep(cfg)
    line_x, line_y, _, weights = focal_line(cfg)
    # Ajouter le foyer même s'il n'est pas dans la liste des capteurs utilisateur.
    original = cfg.SENSORS
    try:
        cfg.SENSORS = list(original) + [cfg.FOCUS] + [(line_x, y) for y in line_y]
        times, ref_all, frame_times, ref_movie, source, positions = reference_plate.simulate(dt)
        times2, filtered_all, frame_times2, filtered_movie, _, positions2 = filtered_plate.simulate(dt)
    finally:
        cfg.SENSORS = original
    if not (np.array_equal(times, times2) and np.array_equal(frame_times, frame_times2)
            and positions == positions2):
        raise RuntimeError("Les deux simulations ne sont pas alignées.")
    n = len(original)
    sensors = positions[:n]
    ref, filtered = ref_all[:, :n], filtered_all[:, :n]
    focus_ref, focus_filtered = ref_all[:, n, 2], filtered_all[:, n, 2]
    line_ref, line_filtered = ref_all[:, n+1:, 2], filtered_all[:, n+1:, 2]
    window = (times >= cfg.FOCUS_WINDOW[0]) & (times <= cfg.FOCUS_WINDOW[1])
    if window.sum() < 2:
        raise ValueError("FOCUS_WINDOW doit contenir au moins deux pas de temps.")
    energy_ref = float(np.sum(focus_ref[window]**2)*dt)
    energy_filtered = float(np.sum(focus_filtered[window]**2)*dt)
    ratio = None if energy_ref <= np.finfo(float).tiny else energy_filtered/energy_ref
    if ratio is None:
        print("Attention : rapport ponctuel au foyer indisponible (référence nulle).", flush=True)
    line_reference_energy = np.sum(line_ref[window]**2, axis=0)*dt
    line_filtered_energy = np.sum(line_filtered[window]**2, axis=0)*dt
    reference_weighted = float(np.dot(weights, line_reference_energy))
    filtered_weighted = float(np.dot(weights, line_filtered_energy))
    if not np.isfinite(reference_weighted) or reference_weighted <= np.finfo(float).tiny:
        raise RuntimeError("Signal de référence pondéré nul sur la ligne ; choisir une fenêtre plus longue.")
    line_ratio = filtered_weighted/reference_weighted
    figures(output, design, times, ref, filtered, frame_times,
            np.stack((ref_movie, filtered_movie)), source, sensors)
    plot_line(output, line_y, weights, line_reference_energy, line_filtered_energy)
    np.savez_compressed(output/"verification_3d.npz", times_s=times, reference_m=ref,
                        filtered_m=filtered, focus_reference_m=focus_ref,
                        focus_filtered_m=focus_filtered, frame_times_s=frame_times,
                        surface_reference_m=ref_movie, surface_filtered_m=filtered_movie,
                        line_y_m=line_y, line_x_m=line_x, line_weights=weights,
                        line_reference_m=line_ref, line_filtered_m=line_filtered,
                        line_reference_energy_m2_s=line_reference_energy,
                        line_filtered_energy_m2_s=line_filtered_energy,
                        sensor_positions_m=np.array(sensors), source_position_m=source)
    report = {"mode_source": cfg.SOURCE_MODE, "dt_s": dt, "fenetre_foyer_s": cfg.FOCUS_WINDOW,
              "rapport_energie_uz_foyer": ratio, "source_m": source,
              "rapport_energie_uz_ligne_ponderee": line_ratio,
              "ligne_focale_x_m": line_x, "ligne_focale_sigma_m": cfg.FOCAL_SIGMA,
              "foyer_m": positions[n], "capteurs_m": sensors}
    (output/"verification_3d.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    focus_text = f"{ratio:.3f}" if ratio is not None else "indisponible"
    print(f"Rapport intégré u_z² pondéré sur la ligne (avec/sans) : {line_ratio:.3f} ; "
          f"au foyer : {focus_text}. Résultats : {output}")


if __name__ == "__main__":
    main()

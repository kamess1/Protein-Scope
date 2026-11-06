"""Build every static figure of a report from an AnalysisResult."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D

from ..constants import KYTE_DOOLITTLE, vdw_radius
from ..druggability import WEIGHTS
from ..interactions import PROBES, ligand_bonds
from ..pipeline import AnalysisResult
from ..structure import canonical_name
from .render import add_outlines, composite_xray, diverging_colors, look_along, render_spheres
from .style import (
    CATEGORICAL,
    DIVERGING_MID,
    DIVERGING_NEGATIVE,
    DIVERGING_POSITIVE,
    GRID,
    HYDROPHILIC,
    HYDROPHOBIC,
    INK,
    INK_SECONDARY,
    OTHER,
    apply_style,
)

POTENTIAL_LIMIT = 5.0
_KIND_COLORS = {
    "hydrogen bond": CATEGORICAL[0],
    "salt bridge (candidate)": CATEGORICAL[1],
    "pi-stack (parallel)": CATEGORICAL[6],
    "pi-stack (T-shaped)": CATEGORICAL[4],
    "metal coordination": CATEGORICAL[5],
    "halogen contact": CATEGORICAL[2],
}


def _view(result: AnalysisResult) -> tuple[np.ndarray, np.ndarray, float]:
    coords = result.structure.protein.coords
    center = coords.mean(axis=0)
    half_width = float(np.linalg.norm(coords - center, axis=1).max()) + 4.0
    if result.pockets:
        direction = result.pockets[0].center - center
        if np.linalg.norm(direction) > 3.0:
            return direction, center, half_width
    _, _, axes = np.linalg.svd(coords - center, full_matrices=False)
    return axes[2], center, half_width


def _save_render(image: np.ndarray, path: Path, title: str, legend=None, annotate=None, colorbar=None) -> None:
    figure, axes = plt.subplots(figsize=(6.2, 6.2 if not colorbar else 6.7), dpi=170)
    axes.imshow(image, interpolation="lanczos")
    axes.set_axis_off()
    axes.set_title(title, pad=8)
    if annotate:
        annotate(axes)
    if legend:
        axes.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncols=min(len(legend), 5))
    if colorbar:
        cmap, limit, label, low_text, high_text = colorbar
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-limit, limit))
        bar = figure.colorbar(mappable, ax=axes, orientation="horizontal", fraction=0.035, pad=0.02)
        bar.set_label(f"{label}   ({low_text} to {high_text})")
        bar.outline.set_visible(False)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _chain_colors(result: AnalysisResult) -> tuple[np.ndarray, dict[str, str]]:
    table = result.structure.protein
    chain_hex = {
        chain: CATEGORICAL[i] if i < len(CATEGORICAL) else OTHER
        for i, chain in enumerate(result.structure.chains)
    }
    # A pale tint keeps the pocket overlay the most saturated element of the picture.
    tint = {c: 0.45 * np.array(to_rgb(h)) + 0.55 * np.ones(3) for c, h in chain_hex.items()}
    return np.array([tint[key.chain] for key in table.residues]), chain_hex


def render_overview(result: AnalysisResult, path: Path) -> None:
    table = result.structure.protein
    direction, center, half_width = _view(result)
    camera = look_along(direction, center, half_width)
    radii = np.array([vdw_radius(e) for e in table.elements])
    colors, chain_hex = _chain_colors(result)
    base = render_spheres(table.coords, radii, colors, camera)

    pockets = result.pockets[:5]
    annotations = []
    if pockets:
        points = np.vstack([p.points for p in pockets])
        point_colors = np.vstack([
            np.tile(np.array(to_rgb("#c2185b")), (len(p.points), 1)) for p in pockets
        ])
        overlay = render_spheres(points, np.full(len(points), 1.1), point_colors, camera)
        image = composite_xray(base, overlay)
        scale = base.rgb.shape[0] / (2 * half_width)
        for pocket in pockets:
            local = (pocket.center - center) @ camera.rotation.T
            annotations.append((
                (local[0] + half_width) * scale, (half_width - local[1]) * scale, str(pocket.pocket_id),
            ))
    else:
        image = add_outlines(base)

    def annotate(axes) -> None:
        for x, y, text in annotations:
            axes.annotate(
                text, (x, y), ha="center", va="center", color="white", fontsize=8, weight="bold",
                bbox={"boxstyle": "circle,pad=0.25", "fc": INK, "ec": "white", "lw": 0.8},
            )

    legend = [
        Line2D([], [], marker="o", ls="", ms=7, mfc=hex_value, mec="none", label=f"Chain {chain}")
        for chain, hex_value in list(chain_hex.items())[:8]
    ]
    if pockets:
        legend.append(Line2D([], [], marker="o", ls="", ms=7, mfc="#c2185b", mec="none", label="Pockets"))
    _save_render(image, path, f"{result.identifier}: structure and top pockets (by rank)", legend, annotate)


def render_property(result: AnalysisResult, path: Path, mode: str) -> None:
    table = result.structure.protein
    direction, center, half_width = _view(result)
    camera = look_along(direction, center, half_width)
    radii = np.array([vdw_radius(e) for e in table.elements])
    if mode == "electrostatics":
        values = result.atom_surface_potential
        limit = POTENTIAL_LIMIT
        low, high = DIVERGING_NEGATIVE, DIVERGING_POSITIVE
        title = f"{result.identifier}: surface electrostatics (screened Coulomb estimate)"
        label = "Potential, kT/e"
        ends = ("negative", "positive")
    else:
        values = np.array([KYTE_DOOLITTLE.get(canonical_name(k.name), 0.0) for k in table.residues])
        limit = 4.5
        low, high = HYDROPHILIC, HYDROPHOBIC
        title = f"{result.identifier}: surface hydropathy (Kyte-Doolittle)"
        label = "Hydropathy index"
        ends = ("hydrophilic", "hydrophobic")
    colors = diverging_colors(values, limit, low, DIVERGING_MID, high)
    layer = render_spheres(table.coords, radii, colors, camera)
    cmap = LinearSegmentedColormap.from_list("scale", [low, DIVERGING_MID, high])
    _save_render(add_outlines(layer, strength=0.35), path, title, colorbar=(cmap, limit, label, *ends))


def render_pocket_closeup(result: AnalysisResult, pocket_id: int, path: Path) -> None:
    """Cut the protein open at the pocket center and show the pocket and any bound ligand."""
    table = result.structure.protein
    pocket = result.pocket(pocket_id)
    center_protein = table.coords.mean(axis=0)
    direction = pocket.center - center_protein
    if np.linalg.norm(direction) < 3.0:
        direction = np.linalg.svd(table.coords - center_protein, full_matrices=False)[2][2]
    extent = float(np.linalg.norm(pocket.points - pocket.center, axis=1).max())
    half_width = max(extent + 10.0, 15.0)
    camera = look_along(direction, pocket.center, half_width)

    radii = np.array([vdw_radius(e) for e in table.elements])
    apolar = np.isin(table.elements, ["C", "S"])
    colors = np.where(apolar[:, None], np.array(to_rgb("#d9d6cd")), np.array(to_rgb("#7ea7c9")))
    lining = set(pocket.lining_keys)
    near = np.array([key in lining for key in table.residues])
    colors = np.where(
        (near & apolar)[:, None], np.array(to_rgb("#e9b26f")),
        np.where((near & ~apolar)[:, None], np.array(to_rgb("#3f8fd0")), colors),
    )
    base = render_spheres(table.coords, radii, colors, camera, clip_front=2.0)

    ligand_labels = [
        label for label, entries in result.ligand_pockets.items()
        if any(entry["pocket_id"] == pocket_id for entry in entries)
    ]
    ligand_labels_shown = ligand_labels[:2]
    volume = render_spheres(
        pocket.points, np.full(len(pocket.points), 0.5), np.tile(np.array(to_rgb("#f2a8c4")), (len(pocket.points), 1)),
        camera,
    )
    image = composite_xray(base, volume, hidden_alpha=0.75)
    if ligand_labels_shown:
        # The ligand is drawn last and opaque, so the pocket grid never hides the bound pose.
        coords = np.vstack([result.ligand(label).atoms.coords for label in ligand_labels_shown])
        ligand_layer = render_spheres(
            coords, np.full(len(coords), 1.15), np.tile(np.array(to_rgb("#0f8f5f")), (len(coords), 1)), camera,
        )
        outlined = add_outlines(ligand_layer, strength=0.45)
        image[ligand_layer.mask] = outlined[ligand_layer.mask]

    legend = [
        Line2D([], [], marker="o", ls="", ms=7, mfc="#e9b26f", mec="none", label="Lining: C/S"),
        Line2D([], [], marker="o", ls="", ms=7, mfc="#3f8fd0", mec="none", label="Lining: N/O"),
        Line2D([], [], marker="o", ls="", ms=7, mfc="#f2a8c4", mec="none", label="Pocket volume"),
    ]
    if ligand_labels_shown:
        legend.append(Line2D([], [], marker="o", ls="", ms=7, mfc="#1baf7a", mec="none",
                             label="Ligand " + ", ".join(ligand_labels_shown)))
    title = f"Pocket {pocket_id}: cut-away view, {pocket.volume:.0f} A3, druggability {pocket.druggability:.2f}"
    _save_render(image, path, title, legend)


def plot_pocket_ranking(result: AnalysisResult, path: Path) -> None:
    pockets = result.pockets[:10]
    figure, axes = plt.subplots(figsize=(7.2, 0.42 * len(pockets) + 1.5), dpi=170)
    positions = np.arange(len(pockets))[::-1]
    names = list(WEIGHTS)
    left = np.zeros(len(pockets))
    for slot, name in enumerate(names):
        values = np.array([WEIGHTS[name] * p.score_components[name] for p in pockets])
        axes.barh(
            positions, values, left=left, height=0.56, color=CATEGORICAL[slot],
            edgecolor=axes.get_facecolor(), linewidth=1.2,
            label=f"{name.capitalize()} (max {WEIGHTS[name]:.2f})",
        )
        left += values
    for position, pocket, total in zip(positions, pockets, left):
        axes.text(
            total + 0.012, position, f"{total:.2f}  ·  {pocket.volume:.0f} A3",
            va="center", color=INK_SECONDARY, fontsize=8,
        )
    axes.set_yticks(positions, [f"Pocket {p.pocket_id}" for p in pockets])
    axes.set_xlim(0, 1.2)
    axes.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axes.set_xlabel("Heuristic druggability score, split by weighted component")
    axes.grid(axis="x")
    axes.set_axisbelow(True)
    axes.spines["left"].set_visible(False)
    axes.tick_params(axis="y", length=0)
    axes.set_title(f"{result.identifier}: pocket ranking")
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2 - 0.25 / max(len(pockets), 1)), ncols=3)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def plot_residue_profile(result: AnalysisResult, path: Path) -> None:
    """Three aligned panels on separate axes: burial, hydropathy and surface potential."""
    residues = result.surface.residues
    index = np.arange(len(residues))
    relative = np.array([r.relative_sasa for r in residues])
    window = 9
    hydropathy = np.convolve(
        np.pad([r.hydropathy for r in residues], window // 2, mode="edge"),
        np.ones(window) / window, mode="valid",
    )
    potential = np.array([result.residue_potential.get(r.key, 0.0) for r in residues])

    figure, panels = plt.subplots(3, 1, figsize=(7.4, 5.2), dpi=170, sharex=True)
    panels[0].fill_between(index, relative, step="mid", color=CATEGORICAL[0], linewidth=0, alpha=0.85)
    panels[0].set_ylabel("Relative SASA")
    panels[0].set_ylim(0, 1)
    panels[0].set_title(f"{result.identifier}: per-residue profile", pad=16)

    panels[1].plot(index, hydropathy, color=CATEGORICAL[0], linewidth=1.2)
    panels[1].axhline(0, color=GRID, linewidth=0.8, zorder=0)
    panels[1].set_ylabel(f"Hydropathy ({window}-residue mean)")

    panels[2].plot(index, potential, color=CATEGORICAL[0], linewidth=0.9)
    panels[2].axhline(0, color=GRID, linewidth=0.8, zorder=0)
    panels[2].set_ylabel("Potential, kT/e")
    panels[2].set_xlabel("Residue, in chain order")

    position = {r.key: i for i, r in enumerate(residues)}
    top = result.pockets[0] if result.pockets else None
    if top:
        marks = [position[key] for key in top.lining_keys if key in position]
        for panel in panels:
            bottom, _ = panel.get_ylim()
            panel.plot(
                marks, np.full(len(marks), bottom), ls="", marker="|", ms=7, mew=1.1,
                color=CATEGORICAL[1], clip_on=False, zorder=4,
            )
        panels[0].legend(
            handles=[Line2D([], [], ls="", marker="|", ms=7, mew=1.1, color=CATEGORICAL[1],
                            label="Lines pocket 1")],
            loc="upper right",
        )

    boundaries, chain_labels, start = [], [], 0
    for i in range(1, len(residues) + 1):
        if i == len(residues) or residues[i].key.chain != residues[start].key.chain:
            chain_labels.append(((start + i) / 2, residues[start].key.chain))
            if i < len(residues):
                boundaries.append(i)
            start = i
    for panel in panels:
        for boundary in boundaries:
            panel.axvline(boundary, color=INK_SECONDARY, linewidth=0.6)
    if len(chain_labels) > 1:
        for x, chain in chain_labels:
            panels[0].annotate(
                f"Chain {chain}", (x, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=7, color=INK_SECONDARY,
            )
    figure.align_ylabels(panels)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def plot_probe_map(result: AnalysisResult, path: Path) -> bool:
    computed = sorted(result.probes)
    if not computed:
        return False
    probes = list(PROBES)
    matrix = np.array([
        [next(r.best_energy for r in result.probes[pid] if r.probe == probe) for probe in probes]
        for pid in computed
    ])
    figure, axes = plt.subplots(figsize=(7.2, 0.5 * len(computed) + 1.7), dpi=170)
    cmap = LinearSegmentedColormap.from_list("favorable", ["#eaf2fb", "#2a78d6", "#0f3f7a"])
    limit = max(float(-matrix.min()), 1.0)
    image = axes.imshow(np.clip(-matrix, 0, None), cmap=cmap, vmin=0, vmax=limit, aspect="auto")
    axes.set_xticks(np.arange(len(probes)), [p.replace(" (", "\n(") for p in probes], fontsize=7.5)
    axes.set_yticks(np.arange(len(computed)), [f"Pocket {pid}" for pid in computed])
    axes.tick_params(length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.set_xticks(np.arange(-0.5, len(probes)), minor=True)
    axes.set_yticks(np.arange(-0.5, len(computed)), minor=True)
    axes.grid(which="minor", color=axes.get_facecolor(), linewidth=2)
    axes.tick_params(which="minor", length=0)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            shade = max(-matrix[row, column], 0) / limit
            axes.text(
                column, row, f"{matrix[row, column]:.1f}", ha="center", va="center", fontsize=8,
                color="white" if shade > 0.5 else INK,
            )
    axes.set_title(f"{result.identifier}: best probe energy per pocket, kcal/mol (rigid-grid estimate)")
    bar = figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    bar.set_label("Favorability (-E), kcal/mol")
    bar.outline.set_visible(False)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return True


def plot_interaction_diagram(result: AnalysisResult, ligand_label: str, path: Path) -> None:
    """2D interaction map: the ligand flattened by PCA, with residues around it in a ring."""
    ligand = result.ligand(ligand_label)
    profile = result.ligand_profiles[ligand_label]
    atoms = ligand.atoms
    centered = atoms.coords - atoms.coords.mean(axis=0)
    axes_3d = np.linalg.svd(centered, full_matrices=False)[2]
    flat = centered @ axes_3d[:2].T
    span = max(float(np.linalg.norm(flat, axis=1).max()), 1.0)
    flat = flat / span
    bonds = ligand_bonds(atoms)
    index_by_name = {str(name): i for i, name in enumerate(atoms.names)}

    specific = [c for c in profile.contacts if c.kind != "hydrophobic"]
    best: dict[tuple[str, str, str], float] = {}
    for contact in specific:
        key = (contact.residue, contact.kind, contact.ligand_atom)
        best[key] = min(best.get(key, 99.0), contact.distance)
    polar_residues = sorted({key[0] for key in best})
    hydrophobic = sorted({c.residue for c in profile.contacts if c.kind == "hydrophobic"} - set(polar_residues))

    def mean_angle(residue: str) -> float:
        anchors = [
            flat[index_by_name[key[2]]] for key in best
            if key[0] == residue and key[2] in index_by_name
        ]
        anchor = np.mean(anchors, axis=0) if anchors else np.array([1.0, 0.0])
        return float(np.arctan2(anchor[1], anchor[0]))

    polar_residues.sort(key=mean_angle)
    count = len(polar_residues)
    # Keep the ring in the angular order of the anchors, spread evenly so labels do not collide.
    positions = {}
    if count:
        offset = mean_angle(polar_residues[0]) if count == 1 else -np.pi + np.pi / count
        for i, residue in enumerate(polar_residues):
            angle = offset + 2 * np.pi * i / count
            positions[residue] = 1.75 * np.array([np.cos(angle), 0.9 * np.sin(angle)])

    figure, axes = plt.subplots(figsize=(7.2, 7.0), dpi=170)
    for i, j in bonds:
        axes.plot(flat[[i, j], 0], flat[[i, j], 1], color=INK, linewidth=1.5, zorder=3, solid_capstyle="round")
    element_colors = {"N": "#2a78d6", "O": "#e34948", "S": "#c98500", "P": "#eb6834", "F": "#008300",
                      "CL": "#008300", "BR": "#8a4b20", "I": "#6250d6"}
    for point, element in zip(flat, atoms.elements):
        color = element_colors.get(str(element))
        if color:
            axes.plot(*point, marker="o", ms=6.5, color=color, mec="white", mew=0.8, zorder=4)

    used = set()
    for (residue, kind, ligand_atom), distance in best.items():
        target = positions[residue]
        anchor = flat[index_by_name[ligand_atom]] if ligand_atom in index_by_name else np.array([0.0, 0.0])
        color = _KIND_COLORS.get(kind, OTHER)
        used.add(kind)
        axes.plot(
            [anchor[0], target[0]], [anchor[1], target[1]], color=color,
            linewidth=1.1, linestyle=(0, (3, 2)), zorder=2,
        )
        middle = anchor * 0.45 + target * 0.55
        axes.text(
            *middle, f"{distance:.1f}", fontsize=6.3, color=INK_SECONDARY, ha="center", va="center", zorder=5,
            bbox={"boxstyle": "round,pad=0.12", "fc": axes.get_facecolor(), "ec": "none"},
        )
    for residue, position in positions.items():
        axes.text(
            *position, residue, ha="center", va="center", fontsize=8, weight="bold", zorder=6,
            bbox={"boxstyle": "round,pad=0.32", "fc": "#eef3f9", "ec": GRID},
        )

    ring = len(hydrophobic)
    for i, residue in enumerate(hydrophobic):
        angle = np.pi / ring * 0.5 + 2 * np.pi * i / max(ring, 1)
        position = 2.5 * np.array([np.cos(angle), 0.94 * np.sin(angle)])
        axes.text(
            *position, residue, ha="center", va="center", fontsize=7, color=INK_SECONDARY, zorder=6,
            bbox={"boxstyle": "round,pad=0.26", "fc": "#fbf1e3", "ec": GRID},
        )

    handles = [
        Line2D([], [], color=_KIND_COLORS.get(kind, OTHER), linestyle=(0, (3, 2)), linewidth=1.3, label=kind)
        for kind in _KIND_COLORS if kind in used
    ]
    handles.append(Line2D([], [], ls="", marker="s", ms=7, mfc="#fbf1e3", mec=GRID, label="Hydrophobic contact only"))
    axes.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncols=3)
    axes.set_xlim(-3.0, 3.0)
    axes.set_ylim(-2.75, 2.75)
    axes.set_aspect("equal")
    axes.set_axis_off()
    axes.set_title(f"{result.identifier}: contacts of ligand {ligand_label} (distances in A, projection not to scale)")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def generate_figures(result: AnalysisResult, out_dir: Path, closeups: int = 3) -> dict[str, Path]:
    """Create all figures below out_dir/figures; returns a name-to-path mapping."""
    apply_style()
    directory = Path(out_dir) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Path] = {}

    def target(name: str) -> Path:
        figures[name] = directory / f"{name}.png"
        return figures[name]

    render_overview(result, target("overview"))
    render_property(result, target("surface_hydropathy"), "hydropathy")
    render_property(result, target("surface_electrostatics"), "electrostatics")
    plot_residue_profile(result, target("residue_profile"))
    if result.pockets:
        plot_pocket_ranking(result, target("pocket_ranking"))
        for pocket in result.pockets[:closeups]:
            render_pocket_closeup(result, pocket.pocket_id, target(f"pocket_{pocket.pocket_id}"))
    if plot_probe_map(result, directory / "probe_map.png"):
        figures["probe_map"] = directory / "probe_map.png"
    # Symmetry-related copies of a ligand look alike, so one diagram per ligand type is enough.
    seen = set()
    for label in result.ligand_profiles:
        name = result.ligand(label).key.name
        if name in seen:
            continue
        seen.add(name)
        safe = label.replace(":", "_")
        plot_interaction_diagram(result, label, target(f"ligand_{safe}"))
    return figures



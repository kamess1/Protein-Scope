from __future__ import annotations

import numpy as np
import pytest

from proteinscope.constants import MAX_ASA
from proteinscope.druggability import score_pocket
from proteinscope.electrostatics import build_charge_model, potential_at
from proteinscope.interactions import find_planar_rings, ligand_bonds
from proteinscope.pockets import Pocket, PocketParams, build_occupancy, psp_events
from proteinscope.surface import analyze_surface, max_diameter


def test_sasa_is_bounded_by_the_reference_scale(toy_structure) -> None:
    surface = analyze_surface(toy_structure)
    assert surface.summary.total_sasa > 0
    assert 0.0 <= surface.summary.apolar_fraction <= 1.0
    for residue in surface.residues:
        assert residue.sasa <= MAX_ASA[residue.key.name] * 1.5
        assert 0.0 <= residue.relative_sasa <= 1.0


def test_max_diameter_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    points = rng.normal(size=(300, 3)) * 10
    expected = max(
        float(np.linalg.norm(a - b)) for i, a in enumerate(points) for b in points[i + 1:]
    )
    assert max_diameter(points) == pytest.approx(expected)


def test_charges_match_the_sequence(toy_structure) -> None:
    model = build_charge_model(toy_structure)
    # 15 aspartates at -1 each, plus the +1 N terminus (the model has no OXT to charge).
    assert model.net_charge == pytest.approx(-14.0)


def test_potential_falls_off_and_follows_the_sign(toy_structure) -> None:
    model = build_charge_model(toy_structure)
    center = toy_structure.protein.coords.mean(axis=0)
    near, far = potential_at(np.array([center + [12, 0, 0], center + [40, 0, 0]]), model)
    assert near < 0  # The toy protein is negatively charged overall.
    assert abs(far) < abs(near)


def test_buried_fraction_is_a_fraction(toy_analysis) -> None:
    for profile in toy_analysis.ligand_profiles.values():
        assert 0.0 <= profile.buried_fraction <= 1.0


def test_occupancy_grid_marks_atom_centers() -> None:
    coords = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    radii = np.array([1.7, 1.7])
    grid, occupied = build_occupancy(coords, radii, PocketParams())
    for position in coords:
        index = np.round((position - grid.origin) / grid.spacing).astype(int)
        assert occupied[tuple(index)]
    assert not occupied[tuple(np.round((np.array([3.0, 0, 0]) - grid.origin)).astype(int))]


def test_psp_events_find_the_gap_between_two_walls() -> None:
    # Two parallel slabs with free space between them: the gap is buried along x only.
    coords = np.array([
        [x, y, z] for x in (0.0, 8.0) for y in np.arange(-6, 7, 2.0) for z in np.arange(-6, 7, 2.0)
    ])
    params = PocketParams(grid_spacing=1.0, probe_radius=1.0)
    grid, occupied = build_occupancy(coords, np.full(len(coords), 1.7), params)
    events = psp_events(occupied, params)
    middle = tuple(np.round((np.array([4.0, 0.0, 0.0]) - grid.origin)).astype(int))
    outside = tuple(np.round((np.array([-9.0, 0.0, 0.0]) - grid.origin)).astype(int))
    assert events[middle] >= 1
    assert events[outside] == 0


def test_pockets_are_ranked_by_score(toy_analysis) -> None:
    scores = [pocket.druggability for pocket in toy_analysis.pockets]
    assert scores == sorted(scores, reverse=True)
    assert [p.pocket_id for p in toy_analysis.pockets] == list(range(1, len(scores) + 1))


def _pocket(**kwargs) -> Pocket:
    defaults = dict(
        pocket_id=0, volume=500.0, center=np.zeros(3), points=np.zeros((1, 3)), enclosure=0.9,
        depth=4.0, lining_residues=["LEU1:A"] * 20, lining_keys=[], apolar_fraction=0.7,
        mean_hydropathy=1.0, net_charge=0.0, charged_residues=1, donor_acceptor_atoms=10,
        mean_potential=0.0,
    )
    return Pocket(**{**defaults, **kwargs})


def test_druggability_rewards_enclosed_apolar_pockets() -> None:
    good = score_pocket(_pocket())
    shallow = score_pocket(_pocket(enclosure=0.6, depth=1.5, apolar_fraction=0.3, charged_residues=9))
    assert good.druggability > shallow.druggability
    assert 0.0 <= shallow.druggability <= good.druggability <= 1.0


def test_druggability_penalizes_oversized_cavities() -> None:
    assert score_pocket(_pocket(volume=600.0)).score_components["size"] > (
        score_pocket(_pocket(volume=6000.0)).score_components["size"]
    )


def test_ring_detection_finds_a_benzene() -> None:
    from proteinscope.structure import AtomTable, ResidueKey

    angles = np.arange(6) * np.pi / 3
    coords = np.column_stack([1.39 * np.cos(angles), 1.39 * np.sin(angles), np.zeros(6)])
    key = ResidueKey("A", 1, " ", "LIG")
    atoms = AtomTable(
        coords=coords, elements=np.array(["C"] * 6), names=np.array([f"C{i}" for i in range(6)]),
        residues=[key] * 6, bfactors=np.zeros(6), occupancies=np.ones(6),
    )
    assert len(find_planar_rings(atoms, ligand_bonds(atoms))) == 1

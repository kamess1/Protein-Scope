"""Tests that talk to RCSB. Run with PROTEINSCOPE_NETWORK_TESTS=1."""

from __future__ import annotations

import pytest

from proteinscope import rcsb
from proteinscope.pipeline import run_analysis
from proteinscope.pockets import ligand_pocket_overlap

pytestmark = pytest.mark.network


def test_download_is_cached(tmp_path) -> None:
    first = rcsb.download_structure("1HSG", tmp_path)
    assert first.is_file() and first.stat().st_size > 10_000
    assert rcsb.download_structure("1HSG", tmp_path) == first


def test_search_filters_by_resolution() -> None:
    hits = rcsb.search("HIV-1 protease", max_results=5, max_resolution=2.0, require_ligand=True)
    assert hits
    assert all(hit.get("resolution_A", 0) <= 2.0 for hit in hits if hit.get("resolution_A"))


@pytest.mark.parametrize(
    ("pdb_id", "ligand"),
    [("1HSG", "MK1"), ("2HYY", "STI"), ("3ERT", "OHT"), ("1HCK", "ATP"), ("2RH1", "CAU")],
)
def test_known_ligand_sits_in_the_top_pocket(pdb_id: str, ligand: str) -> None:
    """The detector must put each of these ligands in its highest-ranked pocket."""
    result = run_analysis(pdb_id, probe_pockets=0)
    instances = [item for item in result.structure.ligands if item.key.name == ligand]
    assert instances, f"{ligand} not parsed from {pdb_id}"
    top = result.pockets[0]
    assert max(ligand_pocket_overlap(item.atoms.coords, top) for item in instances) >= 0.5

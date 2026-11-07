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


def test_alphafold_model_is_analysed_with_plddt(tmp_path) -> None:
    result = run_analysis("AF-P69905", probe_pockets=0, cache_dir=tmp_path)  # Haemoglobin alpha.
    assert result.is_model and result.audit.is_predicted_model
    assert result.audit.mean_plddt and result.audit.mean_plddt > 50
    assert result.context["plddt"]["pockets"]


def test_target_context_and_ligand_chemistry_for_1hsg(tmp_path) -> None:
    result = run_analysis("1HSG", probe_pockets=0, cache_dir=tmp_path)
    chemistry = result.context["ligand_chemistry"]["MK1902:B"]
    assert chemistry["comp_id"] == "MK1" and chemistry["smiles"]
    assert result.context["targets"][0]["accession"] == "P03367"

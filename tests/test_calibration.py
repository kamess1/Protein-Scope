"""Offline tests for the fitted pocket score and the UniProt functional annotation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from proteinscope import druggability, uniprot
from proteinscope.enrich import annotation_summary
from proteinscope.pockets import Pocket
from proteinscope.report.html import _annotation_detail, _function_cell


def _pocket(**kwargs: Any) -> Pocket:
    defaults = dict(
        pocket_id=1, volume=700.0, center=np.zeros(3), points=np.zeros((1, 3)), enclosure=0.88,
        depth=4.0, lining_residues=["LEU1:A"] * 25, lining_keys=[], apolar_fraction=0.62,
        mean_hydropathy=1.0, net_charge=0.0, charged_residues=3, donor_acceptor_atoms=30,
        mean_potential=0.0,
    )
    return Pocket(**{**defaults, **kwargs})


# --- the fitted score ----------------------------------------------------------------


def test_the_fitted_model_ships_with_the_package() -> None:
    model = druggability.load_model()
    assert model is not None, "data/ligandability.json is missing; run scripts/pocket_model.py"
    assert list(model["features"]) == list(druggability.FEATURES)
    assert len(model["coefficients"]) == len(druggability.FEATURES)
    assert model["training"]["entries"] >= 50 and model["training"]["positive_pockets"] >= 50


def test_the_fitted_model_beats_the_heuristic_it_replaces() -> None:
    performance = druggability.load_model()["performance"]
    baseline = performance["baseline_heuristic"]
    assert performance["grouped_by"] == "pdb entry"  # Never cross-validate pockets of one entry.
    assert performance["roc_auc"] > baseline["roc_auc"]
    assert performance["top1_accuracy"] > baseline["top1_accuracy"]


def test_scoring_reports_a_probability_and_names_its_model() -> None:
    pocket = druggability.score_pocket(_pocket())
    assert 0.0 <= pocket.druggability <= 1.0
    assert pocket.score_model.startswith("fitted logistic")
    assert 0.0 <= pocket.heuristic_score <= 1.0  # The heuristic stays available for comparison.
    assert pocket.druggability_class in {"low", "moderate", "high"}
    assert set(pocket.summary()) >= {"druggability", "heuristic_score", "score_model"}


def test_the_heuristic_is_used_when_no_model_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(druggability, "load_model", lambda: None)
    pocket = druggability.score_pocket(_pocket())
    assert pocket.score_model == "heuristic"
    assert pocket.druggability == pytest.approx(pocket.heuristic_score)


def test_a_bigger_more_enclosed_pocket_scores_higher() -> None:
    small = druggability.ligandability(_pocket(volume=90.0))
    large = druggability.ligandability(_pocket(volume=900.0))
    assert small < large


def test_feature_vector_matches_the_shipped_feature_order() -> None:
    values = druggability.feature_vector(_pocket(volume=1000.0, lining_residues=["A"] * 20))
    assert list(values) == list(druggability.FEATURES)
    assert values["log_volume"] == pytest.approx(3.0)
    assert values["charge_density"] == pytest.approx(3 / 20)


# --- UniProt annotation --------------------------------------------------------------

_ENTRY = {
    "primaryAccession": "P00000",
    "uniProtkbId": "TEST_HUMAN",
    "proteinDescription": {"recommendedName": {"fullName": {"value": "Test kinase"}}},
    "genes": [{"geneName": {"value": "TSTK"}}],
    "sequence": {"value": "MGSKLAWDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQR", "length": 40},
    "uniProtKBCrossReferences": [
        {"database": "Pfam", "id": "PF00069", "properties": [{"value": "Pkinase"}]},
        {"database": "PDB", "id": "1ABC"},
    ],
    "features": [
        {"type": "Active site", "location": {"start": {"value": 9}, "end": {"value": 9}},
         "description": "Proton acceptor"},
        {"type": "Binding site", "location": {"start": {"value": 7}, "end": {"value": 8}},
         "description": "", "ligand": {"name": "ATP"}},
        {"type": "Natural variant", "location": {"start": {"value": 10}, "end": {"value": 10}},
         "description": "found in cancer", "featureId": "VAR_1"},
        {"type": "Disulfide bond", "location": {"start": {"value": 3}, "end": {"value": 20}}},
        {"type": "Domain", "location": {"start": {"value": 5}, "end": {"value": 40}},
         "description": "Protein kinase"},
        # A long interval describes a region, not a residue, and must not tag every pocket.
        {"type": "Natural variant", "location": {"start": {"value": 1}, "end": {"value": 40}},
         "description": "large deletion"},
        {"type": "Chain", "location": {"start": {"value": 1}, "end": {"value": 40}}},
    ],
}


def test_features_keep_residue_level_annotation_only() -> None:
    found = {f["type"]: f for f in uniprot.features(_ENTRY)}
    assert set(found) == {"Active site", "Binding site", "Natural variant", "Disulfide bond"}
    assert found["Binding site"]["positions"] == [7, 8]
    assert found["Binding site"]["description"] == "binds ATP"  # Taken from the ligand field.
    assert found["Disulfide bond"]["positions"] == [3, 20]  # Two partners, not an interval.
    assert found["Natural variant"]["positions"] == [10]  # The 40-residue deletion is dropped.
    assert "Chain" not in found and "Domain" not in found


def test_regions_and_description_are_read_separately() -> None:
    assert uniprot.regions(_ENTRY) == [
        {"type": "Domain", "start": 5, "end": 40, "description": "Protein kinase"}
    ]
    described = uniprot.description(_ENTRY)
    assert described["protein_name"] == "Test kinase" and described["genes"] == ["TSTK"]
    assert described["pfam"] == [{"id": "PF00069", "name": "Pkinase"}]


def test_alignment_maps_a_truncated_and_mutated_construct() -> None:
    reference = uniprot.sequence(_ENTRY)
    # The construct starts at reference position 7, carries a point mutation and misses a loop.
    observed = "WDEFGAIK" + "STVWY"
    mapping, identity = uniprot.map_to_reference(observed, reference)
    assert mapping[0] == 7  # First observed residue is reference 7, not 1.
    assert mapping[5] == 12  # H12A: the mutated position still maps.
    assert mapping[8] == 21  # After the missing loop the numbering picks up again.
    assert identity == pytest.approx(12 / 13)  # One mismatch out of thirteen aligned pairs.


def test_a_different_protein_is_rejected_instead_of_mismapped() -> None:
    _, identity = uniprot.map_to_reference("PPPPPPPPPPPP", uniprot.sequence(_ENTRY))
    assert identity < uniprot.MIN_IDENTITY


# --- how the annotation is presented -------------------------------------------------

_ROW = {
    "residues": [
        {"residue": "ASP10:A", "uniprot_position": 9,
         "features": [{"type": "Active site", "category": "catalytic",
                       "description": "Proton acceptor", "ligand": None}]},
        {"residue": "LEU8:A", "uniprot_position": 7,
         "features": [{"type": "Binding site", "category": "binding",
                       "description": "binds ATP", "ligand": "ATP"}]},
    ],
    "categories": {"catalytic": 1, "binding": 1, "variant": 4},
    "regions": ["Protein kinase"], "accessions": ["P00000"],
    "variants": 4, "cofactors": ["ATP"], "is_functional_site": True,
}


def test_annotation_summary_leads_with_the_cofactor() -> None:
    assert annotation_summary(_ROW).startswith("binds ATP")
    assert "4 variants" in annotation_summary(_ROW)
    assert annotation_summary(None) == "-"


def test_the_function_cell_truncates_and_escapes() -> None:
    cell = _function_cell(_ROW, limit=2)
    assert "binds ATP" in cell and "+2" in cell
    assert _function_cell(None) == '<span class="muted">-</span>'
    hostile = {**_ROW, "cofactors": ["<script>"], "categories": {}, "variants": 0}
    assert "<script>" not in _function_cell(hostile)


def test_the_detail_table_puts_function_before_variants() -> None:
    html = _annotation_detail(_ROW)
    assert html.index("Active site") < html.index("Binding site")
    assert "Protein kinase" in html and "ASP10:A" in html
    assert _annotation_detail(None) == ""

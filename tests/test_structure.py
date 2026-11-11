from __future__ import annotations

import pytest
import requests

from proteinscope import rcsb
from proteinscope.constants import MAX_ASA, vdw_radius
from proteinscope.rcsb import normalize_pdb_id
from proteinscope.structure import canonical_name, load_structure


def test_parses_protein_and_ligand(toy_structure) -> None:
    assert toy_structure.chains == ["A"]
    assert len(toy_structure.protein_residues) == 60
    assert [item.key.name for item in toy_structure.ligands] == ["LIG"]
    assert len(toy_structure.protein) == 300


def test_rejects_unknown_format(tmp_path) -> None:
    path = tmp_path / "structure.xyz"
    path.write_text("nothing", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_structure(path)


def test_rejects_structure_without_protein(tmp_path) -> None:
    path = tmp_path / "ligand.pdb"
    path.write_text(
        "HETATM    1  C1  LIG A 900       0.000   0.000   0.000  1.00  0.00           C\nEND\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no protein"):
        load_structure(path)


@pytest.mark.parametrize("value", ["1hsg", " 1HSG ", "4HHB"])
def test_normalizes_ids(value: str) -> None:
    assert normalize_pdb_id(value) == value.strip().upper()


@pytest.mark.parametrize("value", ["HSG", "HSGX", "1HS", "../etc/passwd"])
def test_rejects_bad_ids(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_pdb_id(value)


def test_download_reports_a_missing_entry(tmp_path, monkeypatch) -> None:
    class _Response:
        status_code = 404
        ok = False

        def iter_content(self, size: int):  # The download streams, to bound its total time.
            return iter(())

    monkeypatch.setattr("proteinscope.rcsb.requests.get", lambda *a, **k: _Response())
    with pytest.raises(rcsb.RCSBError, match="does not exist"):
        rcsb.download_structure("1ABC", tmp_path)


def test_download_reports_an_unreachable_server(tmp_path, monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr("proteinscope.rcsb.requests.get", fail)
    with pytest.raises(rcsb.RCSBError, match="failed"):
        rcsb.download_structure("1ABC", tmp_path)


def test_modified_residues_map_to_parent() -> None:
    assert canonical_name("MSE") == "MET"
    assert canonical_name("LEU") == "LEU"
    assert set(MAX_ASA) == {canonical_name(name) for name in MAX_ASA}


def test_vdw_radii_cover_unknown_elements() -> None:
    assert vdw_radius("C") == pytest.approx(1.70)
    assert vdw_radius("xx") > 0

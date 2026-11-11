"""Offline tests for the scoring code of scripts/benchmark_coach420.py.

The benchmark compares methods only as fairly as this code counts, so it is tested on small
hand-made inputs whose answers are known.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_coach420.py"
_spec = importlib.util.spec_from_file_location("benchmark_coach420", _SCRIPT)
bench = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bench  # Dataclasses look their module up while it is executing.
_spec.loader.exec_module(bench)


def _hetatm(serial: int, name: str, resname: str, chain: str, resseq: int, xyz, element: str) -> str:
    return (f"HETATM{serial:5d} {name:<4} {resname:>3} {chain}{resseq:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 20.00          {element:>2}")


def test_ligands_are_grouped_per_residue_and_filtered_by_code() -> None:
    text = "\n".join([
        _hetatm(1, "C1", "NAD", "A", 401, (0, 0, 0), "C"),
        _hetatm(2, "H1", "NAD", "A", 401, (0, 0, 1), "H"),       # Hydrogens are not ligand atoms.
        _hetatm(3, "C2", "NAD", "A", 401, (1, 0, 0), "C"),
        _hetatm(4, "C1", "NAD", "B", 401, (9, 9, 9), "C"),       # Second copy: a second ligand.
        _hetatm(5, "O", "HOH", "A", 501, (5, 5, 5), "O"),        # Not a relevant code.
        _hetatm(6, "C1", "OAA", "A", 402, (3, 0, 0), "C"),
        "ATOM      7  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
    ])
    found = bench.ligand_residues(text, ("NAD", "OAA"))
    assert sorted(len(atoms) for atoms in found) == [1, 1, 2]


def _protein_atom(serial: int, xyz) -> str:
    return (f"ATOM  {serial:5d}  CA  ALA A{serial:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 20.00           C")


def test_covalently_joined_residues_count_as_one_ligand() -> None:
    sugar_chain = [_hetatm(i + 1, "C1", code, "A", 600 + i, (1.5 * i, 3.0, 0), "C")
                   for i, code in enumerate(["NAG", "NAG", "BMA"])]          # 1.5 Å apart: bonded.
    separate = _hetatm(9, "C1", "NAG", "A", 700, (0.0, 2.0, 2.5), "C")    # 2.7 Å from the chain: its own.
    text = "\n".join([_protein_atom(1, (0, 0, 0)), *sugar_chain, separate])
    assert sorted(len(lig) for lig in bench.ligands(text, ("NAG", "BMA"))) == [1, 3]


def test_a_ligand_that_does_not_touch_the_chain_is_not_counted() -> None:
    text = "\n".join([
        _protein_atom(1, (0, 0, 0)),
        _hetatm(2, "C1", "ATP", "A", 500, (3.9, 0, 0), "C"),   # 3.9 Å: in contact.
        _hetatm(3, "C1", "ATP", "B", 500, (30, 0, 0), "C"),    # Bound to a chain that is gone.
    ])
    assert [len(lig) for lig in bench.ligands(text, ("ATP",))] == [1]


def test_fpocket_centres_follow_pocket_numbers() -> None:
    text = "\n".join([
        _hetatm(1, "STP", "STP", "C", 2, (10, 0, 0), "C"),
        _hetatm(2, "STP", "STP", "C", 1, (0, 0, 0), "C"),
        _hetatm(3, "STP", "STP", "C", 1, (2, 0, 0), "C"),
        _hetatm(4, "C1", "LIG", "A", 1, (99, 99, 99), "C"),      # Not a sphere.
    ])
    centers = bench.fpocket_centers(text)
    assert len(centers) == 2
    assert np.allclose(centers[0], [1, 0, 0]) and np.allclose(centers[1], [10, 0, 0])


def test_p2rank_centres_follow_rank_and_tolerate_padding() -> None:
    text = ("name     ,  rank,   score, center_x, center_y, center_z\n"
            "pocket2  ,     2,    3.1,     5.0,      0.0,      0.0\n"
            "pocket1  ,     1,    9.9,     1.0,      2.0,      3.0\n")
    centers = bench.p2rank_centers(text)
    assert np.allclose(centers[0], [1, 2, 3]) and np.allclose(centers[1], [5, 0, 0])


def test_dca_counts_the_nearest_ligand_atom_against_the_cutoff() -> None:
    ligand = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    assert bench.found(ligand, [np.array([9.9, 0.0, 0.0])])        # 3.9 Å from the second atom.
    assert not bench.found(ligand, [np.array([10.1, 0.0, 0.0])])   # 4.1 Å.
    assert bench.found(ligand, [np.array([10.0, 0.0, 0.0])])       # Exactly 4 Å counts.


def test_top_n_and_top_n_plus_two_are_counted_per_ligand() -> None:
    near_a, near_b = np.array([[0.0, 0.0, 0.0]]), np.array([[50.0, 0.0, 0.0]])
    far = np.array([100.0, 100.0, 100.0])
    centers = [np.array([0.5, 0, 0]), far, far, np.array([50.5, 0, 0])]
    # n = 2: the top two pockets find ligand A only; the top four find both.
    assert bench.score_case([near_a, near_b], centers) == (1, 2, 2)
    assert bench.score_case([near_a, near_b], []) == (0, 0, 0)
    # A site found only far down the list counts for "any" but not for the ranked cutoffs.
    late = [far, far, far, far, np.array([0.5, 0, 0])]
    assert bench.score_case([near_a], late) == (0, 0, 1)


def test_list_rows_skip_comments_headers_and_parameters() -> None:
    text = "# comment\nPARAM.PREDICTION_METHOD=fpocket\n\nHEADER: protein ligand_codes\ncoach420/1a26A.pdb  CNA\n"
    assert bench._rows(text) == [["coach420/1a26A.pdb", "CNA"]]


@pytest.mark.parametrize("name", ["P2Rank", "fpocket"])
def test_the_published_reference_numbers_are_the_papers_table_3(name: str) -> None:
    # Krivák & Hoksza 2018, Table 3, COACH420, DCA 4 Å.
    expected = {"P2Rank": (72.0, 78.3), "fpocket": (56.4, 68.9)}[name]
    assert (bench.PUBLISHED[name]["top_n"], bench.PUBLISHED[name]["top_n2"]) == expected

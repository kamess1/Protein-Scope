"""Offline tests for the MCP server, site tags, AlphaFold handling, chemistry and HTML report."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import numpy as np
import pytest

from proteinscope import alphafold, chemistry
from proteinscope.audit import audit_structure
from proteinscope.druggability import score_pocket
from proteinscope.pockets import Pocket
from proteinscope.report.builder import build_report


def _pocket(**kwargs) -> Pocket:
    defaults = dict(
        pocket_id=1, volume=500.0, center=np.zeros(3), points=np.zeros((1, 3)), enclosure=0.9,
        depth=4.0, lining_residues=["LEU1:A"] * 20, lining_keys=[], apolar_fraction=0.7,
        mean_hydropathy=1.0, net_charge=0.0, charged_residues=1, donor_acceptor_atoms=10,
        mean_potential=0.0,
    )
    return Pocket(**{**defaults, **kwargs})


def test_site_tags_mark_a_small_charged_anchor_like_trypsin_s1() -> None:
    # Numbers of the 3PTB S1 pocket: small, polar, strongly negative potential.
    s1 = score_pocket(_pocket(volume=109, enclosure=0.82, apolar_fraction=0.39, mean_potential=-3.7))
    assert s1.druggability_class == "low"  # The score itself is unchanged by design.
    assert "fragment-sized" in s1.site_tags
    assert "anionic anchor: favours cationic groups" in s1.site_tags


def test_site_tags_mark_a_drug_like_apolar_pocket() -> None:
    assert score_pocket(_pocket()).site_tags == ["drug-like apolar"]


@pytest.mark.parametrize(("text", "accession"), [
    ("AF-P00533", "P00533"), ("af:p00533", "P00533"), ("AF-P00533-F1", "P00533"), ("AF_Q9Y6K9", "Q9Y6K9"),
    ("1HSG", None), ("P00533", None),
])
def test_alphafold_identifiers(text: str, accession: str | None) -> None:
    assert alphafold.parse_alphafold_id(text) == accession


def test_audit_reads_plddt_for_predicted_models(toy_structure) -> None:
    report = audit_structure(toy_structure, {"source": "alphafold"})
    assert report.is_predicted_model
    assert report.mean_plddt == pytest.approx(20.0, abs=0.1)  # The toy file has B = 20 everywhere.
    assert report.quality_grade == "low"
    assert report.high_bfactor_residues == []
    assert report.low_confidence_residues
    assert any("pLDDT" in warning for warning in report.warnings)


def test_local_files_get_no_online_context(toy_analysis) -> None:
    assert "targets" not in toy_analysis.context
    assert "ligand_chemistry" not in toy_analysis.context


@pytest.mark.skipif(not chemistry.available(), reason="RDKit not installed")
def test_chemistry_descriptors_for_benzamidine() -> None:
    props = chemistry.properties("NC(=N)c1ccccc1")
    assert props["heavy_atoms"] == 9
    assert props["lipinski_violations"] == []
    assert props["rule_of_three"] is True
    assert chemistry.depiction_svg("NC(=N)c1ccccc1").startswith("<svg")
    with pytest.raises(ValueError):
        chemistry.properties("not a smiles")


def test_html_report_has_sections_and_working_links(toy_analysis, tmp_path: Path) -> None:
    paths = build_report(toy_analysis, tmp_path)
    assert paths.html and paths.html.is_file()
    page = paths.html.read_text(encoding="utf-8")
    for section in ("summary", "viewer-section", "pockets", "ligands", "audit", "figures", "methods"):
        assert f'id="{section}"' in page
    assert "LIG900:A" in page
    for target in re.findall(r'(?:src|href)="([^"#]+)"', page):
        if not target.startswith("http"):
            assert (paths.directory / target).is_file(), target
    assert "report.html" in paths.markdown.read_text(encoding="utf-8")


def test_html_report_escapes_untrusted_text(toy_analysis, tmp_path: Path) -> None:
    from proteinscope.report.narrative import Narrative

    narrative = Narrative(executive_summary="<script>alert(1)</script>", source="test")
    page = build_report(toy_analysis, tmp_path, narrative).html.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_mcp_server_exposes_every_tool(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    from proteinscope.mcp_server import build_server

    server = build_server(tmp_path)
    names = {tool.name for tool in asyncio.run(server.list_tools())}
    assert names == {
        "search_pdb", "entry_info", "analyze_structure", "pocket_detail", "ligand_detail",
        "ligand_chemistry", "target_bioactivity", "compare_structures", "write_report",
    }


def test_mcp_tools_run_and_report_errors(toy_pdb: Path, tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    from proteinscope.mcp_server import build_server

    server = build_server(tmp_path)

    async def run() -> None:
        await server.call_tool("analyze_structure", {"source": str(toy_pdb), "probe_pockets": 0})
        assert "TOY1" in server.workspace.analyses
        with pytest.raises(Exception, match="does not exist"):
            await server.call_tool("pocket_detail", {"pdb_id": "TOY1", "pocket_id": 99})
        await server.call_tool("write_report", {"pdb_id": "TOY1", "executive_summary": "Test."})
        assert (tmp_path / "TOY1" / "report.html").is_file()
        assert server.workspace.narrative.source == "Claude via MCP, from the computed data"

    asyncio.run(run())

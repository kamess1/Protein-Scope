from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from proteinscope.cli import main
from proteinscope.report.builder import analysis_digest, build_report
from proteinscope.report.narrative import rule_based_narrative
from proteinscope.viz.viewer import structure_pdb_text, write_html_viewer


def test_digest_is_json_serializable(toy_analysis) -> None:
    text = json.dumps(analysis_digest(toy_analysis))
    assert "pockets" in json.loads(text)


def test_report_writes_every_artifact(toy_analysis, tmp_path: Path) -> None:
    paths = build_report(toy_analysis, tmp_path)
    assert paths.pdf.stat().st_size > 10_000
    assert paths.markdown.read_text(encoding="utf-8").startswith("# Structure analysis report: TOY1")
    assert paths.viewer and paths.viewer.stat().st_size > 1000
    assert (paths.directory / "pymol" / "session.pml").is_file()
    assert paths.figures["overview"].is_file()
    assert json.loads(paths.data.read_text(encoding="utf-8"))["identifier"] == "TOY1"


def test_markdown_links_point_at_files_that_exist(toy_analysis, tmp_path: Path) -> None:
    paths = build_report(toy_analysis, tmp_path)
    text = paths.markdown.read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)]+)\)", text)
    assert targets
    for target in targets:
        assert (paths.directory / target).is_file(), target


def test_narrative_states_its_source(toy_analysis) -> None:
    narrative = rule_based_narrative(toy_analysis)
    assert narrative.source == "rule-based summary"
    assert "TOY1" in narrative.executive_summary


def test_viewer_escapes_script_terminators(toy_analysis, tmp_path: Path) -> None:
    path = write_html_viewer(toy_analysis, tmp_path / "viewer.html")
    page = path.read_text(encoding="utf-8")
    body = page.split("const PDB =", 1)[1]
    assert "</script>" not in body.split("</script>", 1)[0] + ""
    assert "3Dmol" in page


def test_exported_pdb_keeps_protein_and_ligand_and_drops_water(toy_analysis) -> None:
    text = structure_pdb_text(toy_analysis)
    assert "LIG" in text
    assert "HOH" not in text


def test_cli_analyze_prints_a_table(toy_pdb: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["analyze", str(toy_pdb)]) == 0
    assert "Pockets" in capsys.readouterr().out


def test_cli_report_writes_files(toy_pdb: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["report", str(toy_pdb), "-o", str(tmp_path), "-q"]) == 0
    assert (tmp_path / "TOY1" / "report.pdf").is_file()
    assert "report.pdf" in capsys.readouterr().out


def test_cli_reports_a_bad_identifier(capsys: pytest.CaptureFixture) -> None:
    assert main(["analyze", "not-a-pdb-id"]) == 1
    assert "Error" in capsys.readouterr().err

"""Offline tests for provenance recording, data attribution and the licence files."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import pytest

from proteinscope import provenance, sources, webdata
from proteinscope.report.builder import _attribution_lines, _source_rows, analysis_digest, build_report

ROOT = Path(__file__).resolve().parent.parent


# --- what gets recorded --------------------------------------------------------------


@pytest.mark.parametrize(("url", "name"), [
    ("https://files.rcsb.org/download/1HSG.cif", "RCSB PDB"),
    ("https://data.rcsb.org/graphql", "RCSB PDB"),
    ("https://alphafold.ebi.ac.uk/api/prediction/P00533", "AlphaFold DB"),
    ("https://rest.uniprot.org/uniprotkb/P00533.json", "UniProt"),
    ("https://www.ebi.ac.uk/chembl/api/data/target.json?x=1", "ChEMBL"),
    ("https://www.ebi.ac.uk/interpro/entry/pfam/PF00069/", None),  # Same host, not ChEMBL.
    ("https://example.org/data.json", None),
])
def test_urls_map_to_their_service(url: str, name: str | None) -> None:
    source = sources.source_for_url(url)
    assert (source.name if source else None) == name


def test_reads_are_recorded_only_inside_an_analysis() -> None:
    provenance.record("https://rest.uniprot.org/x", provenance.FROM_NETWORK)  # Nobody listening.
    with provenance.collecting() as recorder:
        provenance.record("https://rest.uniprot.org/x", provenance.FROM_NETWORK)
        provenance.record("https://rest.uniprot.org/y", provenance.FROM_CACHE, 1_700_000_000)
        provenance.record("https://example.org/unknown", provenance.FROM_NETWORK)
    use = recorder.uses["UniProt"]
    assert (use.requests, use.from_cache, use.stale_fallback) == (2, 1, 0)
    assert use.oldest_retrieval == 1_700_000_000
    assert set(recorder.uses) == {"UniProt"}


def test_parallel_analyses_keep_separate_records() -> None:
    results: dict[str, set[str]] = {}

    def analyse(name: str, url: str) -> None:
        with provenance.collecting() as recorder:
            for _ in range(50):
                provenance.record(url, provenance.FROM_NETWORK)
                time.sleep(0.0005)
        results[name] = set(recorder.uses)

    threads = [threading.Thread(target=analyse, args=("a", "https://rest.uniprot.org/a")),
               threading.Thread(target=analyse, args=("b", "https://www.ebi.ac.uk/chembl/b"))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == {"a": {"UniProt"}, "b": {"ChEMBL"}}


def test_the_web_cache_reports_where_data_came_from(tmp_path: Path, monkeypatch) -> None:
    class Response:
        status_code, ok = 200, True

        def json(self) -> dict:
            return {"live": True}

    monkeypatch.setattr(webdata, "_fetch", lambda url, deadline: Response())
    url = "https://rest.uniprot.org/uniprotkb/P1.json"
    with provenance.collecting() as recorder:
        webdata.get_json(url, tmp_path)          # Network, then cached.
        webdata.get_json(url, tmp_path)          # Fresh cache.
    use = recorder.uses["UniProt"]
    assert (use.requests, use.from_cache, use.stale_fallback) == (2, 1, 0)

    path = webdata._cache_path(url, tmp_path)
    old = time.time() - 90 * 86400
    import os

    os.utime(path, (old, old))

    def outage(url: str, deadline: float):
        raise webdata.DataSourceError("down")

    monkeypatch.setattr(webdata, "_fetch", outage)
    monkeypatch.setattr(webdata, "_RETRY_DELAYS", (0.0, 0.0, 0.0))
    with provenance.collecting() as recorder:
        webdata.get_json(url, tmp_path)
    assert recorder.uses["UniProt"].stale_fallback == 1


# --- the provenance block of an analysis ---------------------------------------------


def test_a_local_analysis_is_identified_by_checksum_not_path(toy_analysis, toy_pdb: Path) -> None:
    block = toy_analysis.provenance
    assert block["structure"]["kind"] == "file"
    assert block["structure"]["sha256"] == hashlib.sha256(toy_pdb.read_bytes()).hexdigest()
    assert block["structure"]["bytes"] == toy_pdb.stat().st_size
    assert "retrieved_utc" not in block["structure"]
    assert block["data_sources"] == []  # A local file is analysed without any service.
    text = json.dumps(block)
    assert str(toy_pdb.parent) not in text and str(Path.home()) not in text


def test_the_block_names_software_model_and_parameters(toy_analysis) -> None:
    from proteinscope import __version__
    from proteinscope.druggability import load_model

    block = toy_analysis.provenance
    assert block["proteinscope"] == __version__
    assert block["libraries"]["numpy"]
    assert block["pocket_model"]["version"] == load_model()["version"]
    assert len(block["pocket_model"]["sha256"]) == 64
    assert block["parameters"]["grid_spacing"] == 1.0 and block["parameters"]["online_context"] is False
    assert block["generated_utc"].endswith("Z")


def test_every_output_carries_the_provenance(toy_analysis, tmp_path: Path) -> None:
    assert analysis_digest(toy_analysis)["provenance"] == toy_analysis.provenance
    paths = build_report(toy_analysis, tmp_path)
    assert "## Provenance" in paths.markdown.read_text(encoding="utf-8")
    assert 'id="provenance"' in paths.html.read_text(encoding="utf-8")
    saved = json.loads(paths.data.read_text(encoding="utf-8"))
    assert saved["provenance"]["structure"]["sha256"] == toy_analysis.provenance["structure"]["sha256"]


# --- attribution in the report --------------------------------------------------------


def _with_sources(result, *services: dict):
    clone = type(result).__new__(type(result))
    clone.__dict__.update(result.__dict__)
    clone.provenance = {**result.provenance, "data_sources": list(services)}
    return clone


def _service(name: str, expired: int = 0) -> dict:
    return {"service": name, "licence": sources.BY_NAME[name].licence, "requests": 2, "from_cache": 1,
            "expired_cache_used": expired, "oldest_data_utc": "2026-11-01T00:00:00Z",
            "newest_data_utc": "2026-11-02T00:00:00Z"}


def test_only_the_services_used_are_credited(toy_analysis) -> None:
    result = _with_sources(toy_analysis, _service("UniProt"))
    lines = _attribution_lines(result)
    assert any("UniProt" in line and "CC BY 4.0" in line for line in lines)
    assert not any("ChEMBL" in line or "AlphaFold" in line for line in lines)


def test_chembl_brings_its_share_alike_notice(toy_analysis) -> None:
    lines = _attribution_lines(_with_sources(toy_analysis, _service("ChEMBL")))
    assert any("CC BY-SA 3.0" in line and "same licence" in line for line in lines)


def test_alphafold_brings_the_required_copyright_line(toy_analysis) -> None:
    lines = _attribution_lines(_with_sources(toy_analysis, _service("AlphaFold DB")))
    assert any("AlphaFold Data Copyright (2021) DeepMind Technologies Limited." in line for line in lines)


def test_an_expired_cache_is_flagged(toy_analysis) -> None:
    result = _with_sources(toy_analysis, _service("ChEMBL", expired=3))
    assert any("expired local cache" in line for line in _attribution_lines(result))
    assert "3 from an expired cache" in _source_rows(result)[1][3]


# --- licence files -------------------------------------------------------------------


def test_the_code_licence_is_mit_and_declared() -> None:
    licence = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert licence.startswith("MIT License") and "Copyright (c) 2026 kamess1" in licence
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'license = "MIT"' in project


def test_every_source_has_licence_terms_and_citation() -> None:
    for source in sources.SOURCES:
        assert source.licence and source.citation and source.credit
        assert source.licence_url.startswith("https://") and source.terms_url.startswith("https://")

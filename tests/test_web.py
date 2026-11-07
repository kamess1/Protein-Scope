"""Offline tests for `proteinscope serve`: the overview page and the static server."""

from __future__ import annotations

import re
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from proteinscope import web
from proteinscope.report.builder import build_report


@pytest.fixture
def served(toy_analysis, tmp_path: Path):
    """A running server over an output directory that holds one real report."""
    build_report(toy_analysis, tmp_path)
    server = web.build_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{web.HOST}:{server.server_address[1]}", tmp_path
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310 - our own loopback server.
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def test_an_empty_directory_explains_what_to_do(tmp_path: Path) -> None:
    assert web.collect_entries(tmp_path) == []
    page = web.render_index([], tmp_path)
    assert "No reports yet" in page and "proteinscope report 1HSG" in page


def test_a_directory_without_an_analysis_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "notes").mkdir()
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "analysis.json").write_text("{ not json", encoding="utf-8")
    assert web.collect_entries(tmp_path) == []


def test_an_entry_carries_what_the_card_shows(toy_analysis, tmp_path: Path) -> None:
    build_report(toy_analysis, tmp_path)
    entry = web.collect_entries(tmp_path)[0]
    assert entry.identifier == "TOY1"
    assert entry.pocket_count == len(toy_analysis.pockets)
    assert entry.top_score == pytest.approx(toy_analysis.pockets[0].druggability, abs=1e-3)
    assert entry.href == "TOY1/report.html"
    assert set(entry.files) >= {"HTML", "PDF", "Markdown", "JSON"}
    assert entry.thumbnail == "TOY1/figures/overview.png"


def test_the_newest_report_comes_first(toy_analysis, tmp_path: Path) -> None:
    build_report(toy_analysis, tmp_path)
    older = tmp_path / "OLD1"
    older.mkdir()
    (older / "analysis.json").write_text('{"identifier": "OLD1", "pockets": []}', encoding="utf-8")
    import os

    os.utime(older / "analysis.json", (0, 0))
    assert [entry.identifier for entry in web.collect_entries(tmp_path)] == ["TOY1", "OLD1"]


def test_the_index_links_resolve_and_text_is_escaped(toy_analysis, tmp_path: Path) -> None:
    build_report(toy_analysis, tmp_path)
    entries = web.collect_entries(tmp_path)
    entries[0].title = '<script>alert("x")</script>'
    page = web.render_index(entries, tmp_path)
    assert "<script>alert" not in page.split("</head>")[1]
    for target in re.findall(r'(?:src|href)="([^"#]+)"', page):
        assert (tmp_path / target).is_file(), target


def test_the_server_delivers_the_index_and_the_report(served) -> None:
    base, _ = served
    status, body = _get(f"{base}/")
    assert status == 200 and b"TOY1" in body
    assert _get(f"{base}/TOY1/report.html")[0] == 200
    assert _get(f"{base}/TOY1/figures/overview.png")[0] == 200


def test_the_server_refuses_to_leave_the_output_directory(served, tmp_path: Path) -> None:
    base, out_dir = served
    secret = tmp_path.parent / "secret.txt"
    secret.write_text("private", encoding="utf-8")
    for attempt in ("/../secret.txt", "/TOY1/../../secret.txt", "/%2e%2e/secret.txt"):
        status, body = _get(base + attempt)
        assert status == 404, attempt
        assert b"private" not in body, attempt


def test_the_index_is_rebuilt_for_every_request(served, toy_analysis) -> None:
    base, out_dir = served
    assert b"SECOND" not in _get(f"{base}/")[1]
    second = out_dir / "SECOND"
    second.mkdir()
    (second / "analysis.json").write_text(
        '{"identifier": "SECOND", "pockets": [], "audit": {"title": "A later run"}}', encoding="utf-8"
    )
    body = _get(f"{base}/")[1]
    assert b"SECOND" in body and b"A later run" in body

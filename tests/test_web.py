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


def test_a_symlink_out_of_the_output_directory_is_refused(served, tmp_path: Path) -> None:
    base, out_dir = served
    secret = tmp_path.parent / "linked-secret.txt"
    secret.write_text("private", encoding="utf-8")
    (out_dir / "escape.txt").symlink_to(secret)
    status, body = _get(f"{base}/escape.txt")
    assert status == 404 and b"private" not in body


def test_directories_are_not_listed(served) -> None:
    base, _ = served
    assert _get(f"{base}/TOY1/")[0] == 404
    assert _get(f"{base}/TOY1/figures/")[0] == 404
    assert _get(f"{base}/TOY1/report.html")[0] == 200  # The files themselves still work.


def test_the_overview_never_shows_an_absolute_path(served) -> None:
    base, out_dir = served
    page = _get(f"{base}/")[1].decode("utf-8")
    assert str(Path.home()) not in page
    assert str(out_dir) not in page


def test_display_path_keeps_the_home_directory_out_of_sight() -> None:
    # Outside the working directory only the folder name survives, never the path to it.
    assert web.display_path(Path.home() / "private" / "reports") == "reports"
    assert web.display_path(Path.cwd() / "output") == "./output"
    assert web.display_path(Path.cwd()) == "."
    assert str(Path.home()) not in web.display_path(Path.home())


def test_responses_do_not_leak_the_local_url_to_a_cdn(served) -> None:
    base, _ = served
    with urllib.request.urlopen(f"{base}/TOY1/report.html", timeout=10) as response:  # noqa: S310
        headers = response.headers
    assert headers["Referrer-Policy"] == "no-referrer"
    assert headers["X-Content-Type-Options"] == "nosniff"


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


def test_written_reports_contain_no_local_paths(toy_analysis, tmp_path: Path) -> None:
    """Reports get shared. Nothing in them may name the account or the disk layout."""
    paths = build_report(toy_analysis, tmp_path)
    home = str(Path.home())
    written = [paths.markdown, paths.data, paths.pdf, paths.html, paths.viewer, paths.pymol]
    for path in [p for p in written if p] + sorted(paths.figures.values()):
        blob = path.read_bytes()
        assert home.encode() not in blob, f"{path.name} names the home directory"
        assert str(tmp_path).encode() not in blob, f"{path.name} names its own absolute path"


def test_the_committed_mcp_config_has_no_absolute_paths() -> None:
    """.mcp.json is committed, so it must not carry anyone's directory layout."""
    import json as _json

    config = Path(__file__).resolve().parent.parent / ".mcp.json"
    if not config.is_file():
        pytest.skip("no .mcp.json in this checkout")
    server = _json.loads(config.read_text(encoding="utf-8"))["mcpServers"]["proteinscope"]
    for value in [server["command"], *server.get("args", [])]:
        assert not value.startswith(("/", "~")), f"{value!r} is an absolute path"
        assert "Users" not in value and "home" not in value

"""Offline tests for analyses started from the browser: catalogue, job queue and the API gates.

The analysis itself is replaced by a fake runner, so these tests need no network; what they
check is who may start an analysis, with what input, and what the browser gets back.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from proteinscope import catalog, jobs, pipeline, web


# --- catalogue and identifiers -------------------------------------------------------


def test_the_catalogue_loads_and_is_normalised() -> None:
    areas, entries = catalog.load_catalog()
    assert len(entries) >= 20
    assert len({e.id for e in entries}) == len(entries)  # No duplicates.
    for entry in entries:
        assert catalog.validate_identifier(entry.id) == entry.id
        assert entry.area in areas and entry.name and entry.why


@pytest.mark.parametrize(("text", "expected"), [
    ("1hsg", "1HSG"), ("  6oim ", "6OIM"), ("af-p00533", "AF-P00533"), ("AF_Q9Y6K9", "AF-Q9Y6K9"),
])
def test_identifiers_are_normalised(text: str, expected: str) -> None:
    assert catalog.validate_identifier(text) == expected


@pytest.mark.parametrize("text", [
    "/etc/passwd", "../1HSG", "1HSG.pdb", "~/1HSG", "1HSG/../x", "C:\\1HSG", "file:///etc/hosts",
    "https://example.org/1HSG", "", "   ", "1HS", "12345", "HSG1", "x" * 200, "AF-../../x", None, 1234,
])
def test_anything_but_an_identifier_is_refused(text: object) -> None:
    with pytest.raises(ValueError):
        catalog.validate_identifier(text)  # type: ignore[arg-type]


def test_the_web_path_never_reads_a_local_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file named like a PDB ID must not be picked up when files are not allowed."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "1ABC").write_text("not a structure", encoding="utf-8")
    monkeypatch.setattr(pipeline.rcsb, "download_structure", lambda pdb_id, cache: Path("downloaded.cif"))
    assert pipeline.resolve_source("1ABC")[2] == "file"  # The CLI keeps accepting local files.
    path, identifier, kind = pipeline.resolve_source("1ABC", allow_files=False)
    assert (path, identifier, kind) == (Path("downloaded.cif"), "1ABC", "pdb")


# --- the job queue -------------------------------------------------------------------


def _fake_runner(calls: list[str] | None = None, fail: Exception | None = None):
    def run(identifier: str, out_dir: Path, progress) -> None:
        if calls is not None:
            calls.append(identifier)
        progress("Searching pockets", 0.4)
        progress("Going backwards", 0.2)  # The bar must never move back.
        if fail:
            raise fail
        (out_dir / identifier).mkdir(parents=True, exist_ok=True)
        (out_dir / identifier / "report.html").write_text("<h1>report</h1>", encoding="utf-8")
    return run


def test_a_job_runs_to_a_report(tmp_path: Path) -> None:
    queue = jobs.JobQueue(tmp_path, _fake_runner())
    queue.submit("1HSG")
    assert queue.wait_idle(10)
    job = queue.get("1HSG")
    assert job.state == jobs.DONE and job.fraction == 1.0
    assert job.to_dict()["report"] == "1HSG/report.html"
    assert (tmp_path / "1HSG" / "report.html").is_file()


def test_progress_never_runs_backwards(tmp_path: Path) -> None:
    seen: list[float] = []
    gate = threading.Event()

    def run(identifier: str, out_dir: Path, progress) -> None:
        progress("a", 0.5)
        seen.append(queue.get(identifier).fraction)
        progress("b", 0.3)
        seen.append(queue.get(identifier).fraction)
        progress("c", 5.0)  # Out of range values are clamped below completion.
        seen.append(queue.get(identifier).fraction)
        gate.set()

    queue = jobs.JobQueue(tmp_path, run)
    queue.submit("1HSG")
    assert gate.wait(10) and queue.wait_idle(10)
    assert seen == [0.5, 0.5, 0.99]


def test_a_failure_ends_the_job_with_a_safe_message(tmp_path: Path) -> None:
    secret = Path.home() / "secret" / "structures" / "1HSG.cif"
    queue = jobs.JobQueue(tmp_path, _fake_runner(fail=ValueError(f"could not parse {secret}\nTraceback ...")))
    queue.submit("1HSG")
    assert queue.wait_idle(10)
    job = queue.get("1HSG")
    assert job.state == jobs.FAILED
    assert str(Path.home()) not in job.error and "Traceback" not in job.error
    assert "~" in job.error

    queue.runner = _fake_runner()  # The worker survived the failure and takes the next job.
    queue.submit("1HSG")
    assert queue.wait_idle(10) and queue.get("1HSG").state == jobs.DONE


def test_internal_errors_are_not_described(tmp_path: Path) -> None:
    assert jobs.public_message(KeyError(str(Path.home() / "x"))) == (
        "The analysis stopped with an internal error (KeyError)."
    )


def test_resubmitting_a_running_job_returns_it(tmp_path: Path) -> None:
    release = threading.Event()
    calls: list[str] = []

    def run(identifier: str, out_dir: Path, progress) -> None:
        calls.append(identifier)
        release.wait(10)

    queue = jobs.JobQueue(tmp_path, run)
    first = queue.submit("1HSG")
    assert queue.submit("1HSG") is first
    release.set()
    assert queue.wait_idle(10) and calls == ["1HSG"]


def test_the_queue_has_a_limit(tmp_path: Path) -> None:
    release = threading.Event()
    queue = jobs.JobQueue(tmp_path, lambda i, o, p: release.wait(10), max_pending=2)
    queue.submit("1HSG")
    queue.submit("2HYY")
    with pytest.raises(jobs.QueueFull):
        queue.submit("3ERT")
    release.set()
    assert queue.wait_idle(10)


# --- the HTTP API --------------------------------------------------------------------


@pytest.fixture
def server(tmp_path: Path):
    calls: list[str] = []
    instance = web.build_server(tmp_path, port=0, runner=_fake_runner(calls))
    instance.calls = calls  # type: ignore[attr-defined]
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield instance
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=5)


def _request(server, path: str, body: object = None, headers: dict[str, str] | None = None,
             token: bool = True, raw: bytes | None = None) -> tuple[int, dict]:
    sent = {"Content-Type": "application/json"} if body is not None or raw is not None else {}
    if token:
        sent[web.TOKEN_HEADER] = server.token
    sent.update(headers or {})
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    request = urllib.request.Request(server.url.rstrip("/") + path, data=data, headers=sent)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - loopback.
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        payload = error.read()
        try:
            return error.code, json.loads(payload)
        except ValueError:
            return error.code, {}


def test_a_valid_request_starts_an_analysis(server) -> None:
    status, payload = _request(server, "/api/analyze", {"id": " 6oim "})
    assert status == 202 and payload["job"]["id"] == "6OIM"
    assert server.jobs.wait_idle(10)
    status, payload = _request(server, "/api/jobs")
    assert status == 200 and payload["jobs"][0]["state"] == "done"
    assert server.calls == ["6OIM"]


@pytest.mark.parametrize(("headers", "token", "expected"), [
    ({}, False, 403),                                           # No token.
    ({web.TOKEN_HEADER: "guess"}, False, 403),                  # Wrong token.
    ({"Origin": "https://evil.example"}, True, 403),            # Another site.
    ({"Sec-Fetch-Site": "cross-site"}, True, 403),              # Another site, as browsers say it.
    ({"Host": "rebound.example:8765"}, True, 421),              # DNS rebinding.
])
def test_foreign_requests_are_refused(server, headers, token, expected) -> None:
    status, _ = _request(server, "/api/analyze", {"id": "1HSG"}, headers=headers, token=token)
    assert status == expected
    assert server.calls == [] and server.jobs.snapshot() == []


def test_only_json_with_an_identifier_is_accepted(server) -> None:
    assert _request(server, "/api/analyze", raw=b"id=1HSG",
                    headers={"Content-Type": "application/x-www-form-urlencoded"})[0] == 415
    assert _request(server, "/api/analyze", raw=b"x" * (web.MAX_BODY + 1))[0] == 413
    assert _request(server, "/api/analyze", raw=b"{not json")[0] == 400
    assert _request(server, "/api/analyze", {"identifier": "1HSG"})[0] == 400
    status, payload = _request(server, "/api/analyze", {"id": "../../etc/passwd"})
    assert status == 400 and "PDB ID" in payload["error"]
    assert server.calls == []


def test_reading_jobs_needs_the_token(server) -> None:
    assert _request(server, "/api/jobs", token=False)[0] == 403
    assert _request(server, "/api/nothing")[0] == 404


def test_the_page_carries_this_servers_token_only(server, tmp_path: Path) -> None:
    with urllib.request.urlopen(server.url, timeout=10) as response:  # noqa: S310
        page = response.read().decode()
        assert response.headers["X-Frame-Options"] == "SAMEORIGIN"
    assert f'content="{server.token}"' in page
    other = web.build_server(tmp_path / "other", port=0)
    try:
        assert other.token != server.token and len(server.token) >= 40
    finally:
        other.server_close()


def test_a_read_only_server_starts_nothing(tmp_path: Path) -> None:
    instance = web.build_server(tmp_path, port=0, read_only=True)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        assert instance.jobs is None
        assert _request(instance, "/api/analyze", {"id": "1HSG"})[0] == 405
        with urllib.request.urlopen(instance.url, timeout=10) as response:  # noqa: S310
            page = response.read().decode()
        assert 'id="lookup"' not in page and "data-analyse" not in page
        assert "proteinscope report 1M17" in page  # Read-only cards say how to get the report.
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=5)


def test_catalogue_cards_know_which_reports_exist(toy_analysis, tmp_path: Path) -> None:
    (tmp_path / "1M17").mkdir()
    (tmp_path / "1M17" / "analysis.json").write_text('{"identifier": "1M17", "pockets": []}', encoding="utf-8")
    page = web.render_index(web.collect_entries(tmp_path), tmp_path, token="t", read_only=False)
    card = page[page.index('data-id="1M17"'):]
    card = card[:card.index("</article>")]
    assert "Report ready" in card and 'href="1M17/report.html"' in card
    other = page[page.index('data-id="6OIM"'):]
    other = other[:other.index("</article>")]
    assert "Not analysed yet" in other and 'data-analyse="6OIM"' in other

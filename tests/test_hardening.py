"""Offline tests for the script integrity pin, the model texts and the web cache expiry."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pytest

from proteinscope import druggability, webdata
from proteinscope.viz import viewer


# --- 3Dmol.js integrity ----------------------------------------------------------------


def test_the_viewer_pins_its_cdn_script(toy_analysis, tmp_path: Path) -> None:
    page = viewer.write_html_viewer(toy_analysis, tmp_path / "viewer.html").read_text(encoding="utf-8")
    tags = re.findall(r"<script src=[^>]+>", page)
    assert len(tags) == 1, "every external script needs its own integrity pin"
    tag = tags[0]
    assert f'src="{viewer.VIEWER_SCRIPT}"' in tag
    assert f'integrity="{viewer.VIEWER_SCRIPT_INTEGRITY}"' in tag
    assert 'crossorigin="anonymous"' in tag  # Without it the browser ignores integrity.
    assert re.fullmatch(r"sha(384|512)-[A-Za-z0-9+/]+=*", viewer.VIEWER_SCRIPT_INTEGRITY)


def test_inserted_text_cannot_smuggle_in_another_placeholder(toy_analysis, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(toy_analysis.audit, "title", "Title with __PDB__ and __SCRIPT__ inside")
    page = viewer.write_html_viewer(toy_analysis, tmp_path / "viewer.html").read_text(encoding="utf-8")
    assert "Title with __PDB__ and __SCRIPT__ inside" in page
    assert page.count(viewer.VIEWER_SCRIPT) == 1


# --- texts about the model -----------------------------------------------------------


def test_the_score_description_comes_from_the_shipped_model() -> None:
    model = druggability.load_model()
    text = druggability.score_description()
    assert f"{model['training']['entries']} PDB entries" in text
    assert f"ROC-AUC {model['performance']['roc_auc']:.2f}" in text
    assert "not a measure of how tractable" in text


def test_the_description_follows_a_retrained_model() -> None:
    model = json.loads(json.dumps(druggability.load_model()))
    model["training"]["entries"], model["performance"]["roc_auc"] = 140, 0.95
    text = druggability.score_description(model)
    assert "140 PDB entries" in text and "ROC-AUC 0.95" in text


def test_without_a_model_the_heuristic_is_described(monkeypatch) -> None:
    monkeypatch.setattr(druggability, "load_model", lambda: None)
    text = druggability.score_description()
    assert "hand-weighted heuristic" in text and "not a probability" in text


def test_the_llm_instructions_match_the_model() -> None:
    entries = druggability.load_model()["training"]["entries"]
    pytest.importorskip("mcp")
    from proteinscope.mcp_server import INSTRUCTIONS

    texts = [INSTRUCTIONS]
    try:
        from proteinscope.agents.crew import SHARED_RULES
    except ImportError:  # The API client is optional; the MCP path must not need it.
        pass
    else:
        texts.append(SHARED_RULES)
    for text in texts:
        assert f"{entries} PDB entries" in text
        assert "78 complexes" not in text
        assert "hand-weighted heuristic that ranks" not in text
        assert druggability.SCORE_WEAKNESS in text


# --- web cache expiry ----------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: object, status: int = 200) -> None:
        self.status_code, self._payload = status, payload

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def json(self) -> object:
        return self._payload


@pytest.fixture
def fetches(monkeypatch):
    """Record every network fetch; tests set `.answer` to a response or an exception."""
    class Recorder:
        answer: object = _FakeResponse({"fresh": True})
        count = 0

    def fake_fetch(url: str, deadline: float):
        Recorder.count += 1
        if isinstance(Recorder.answer, Exception):
            raise Recorder.answer
        return Recorder.answer

    monkeypatch.setattr(webdata, "_fetch", fake_fetch)
    monkeypatch.setattr(webdata, "_RETRY_DELAYS", (0.0, 0.0, 0.0))  # Same retries, no waiting.
    monkeypatch.delenv("PROTEINSCOPE_CACHE_DAYS", raising=False)
    return Recorder


def _cache(tmp_path: Path, url: str, payload: object, age_days: float) -> Path:
    path = webdata._cache_path(url, tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    stamp = time.time() - age_days * 86400
    os.utime(path, (stamp, stamp))
    return path


def test_a_fresh_answer_comes_from_the_cache(tmp_path: Path, fetches) -> None:
    _cache(tmp_path, "https://x/a", {"cached": True}, age_days=3)
    assert webdata.get_json("https://x/a", tmp_path) == {"cached": True}
    assert fetches.count == 0


def test_an_expired_answer_is_fetched_again(tmp_path: Path, fetches) -> None:
    path = _cache(tmp_path, "https://x/a", {"cached": True}, age_days=45)
    assert webdata.get_json("https://x/a", tmp_path) == {"fresh": True}
    assert fetches.count == 1
    assert json.loads(path.read_text()) == {"fresh": True}  # The cache was renewed.
    assert time.time() - path.stat().st_mtime < 60


def test_an_expired_answer_beats_an_outage(tmp_path: Path, fetches) -> None:
    _cache(tmp_path, "https://x/a", {"cached": True}, age_days=45)
    fetches.answer = webdata.DataSourceError("ChEMBL is down")
    assert webdata.get_json("https://x/a", tmp_path) == {"cached": True}


def test_without_any_copy_an_outage_is_reported(tmp_path: Path, fetches) -> None:
    fetches.answer = webdata.DataSourceError("ChEMBL is down")
    with pytest.raises(webdata.DataSourceError):
        webdata.get_json("https://x/a", tmp_path)


def test_the_cache_age_can_be_configured(tmp_path: Path, fetches, monkeypatch) -> None:
    _cache(tmp_path, "https://x/a", {"cached": True}, age_days=3)
    monkeypatch.setenv("PROTEINSCOPE_CACHE_DAYS", "1")
    assert webdata.get_json("https://x/a", tmp_path) == {"fresh": True}
    monkeypatch.setenv("PROTEINSCOPE_CACHE_DAYS", "not a number")  # Falls back to 30 days.
    assert webdata.cache_max_age() == 30 * 86400


def test_a_corrupt_cache_file_is_replaced(tmp_path: Path, fetches) -> None:
    path = _cache(tmp_path, "https://x/a", {}, age_days=1)
    path.write_text("{ broken", encoding="utf-8")
    assert webdata.get_json("https://x/a", tmp_path) == {"fresh": True}


# --- retries -------------------------------------------------------------------------


def _sequence(monkeypatch, *answers):
    """Make _fetch answer with these responses/exceptions in turn; return the call log."""
    calls = []
    pending = list(answers)

    def fake_fetch(url: str, deadline: float):
        calls.append(url)
        answer = pending.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(webdata, "_fetch", fake_fetch)
    monkeypatch.setattr(webdata, "_RETRY_DELAYS", (0.0, 0.0, 0.0))
    return calls


def test_a_transient_server_error_is_retried(tmp_path: Path, monkeypatch) -> None:
    calls = _sequence(monkeypatch, _FakeResponse({}, 500), webdata.DataSourceError("reset"),
                      _FakeResponse({}, 503), _FakeResponse({"ok": True}))
    assert webdata.get_json("https://x/r", tmp_path) == {"ok": True}
    assert len(calls) == 4


def test_a_persistent_server_error_is_reported(tmp_path: Path, monkeypatch) -> None:
    calls = _sequence(monkeypatch, *[_FakeResponse({}, 500)] * 4)
    with pytest.raises(webdata.DataSourceError, match="HTTP 500"):
        webdata.get_json("https://x/r", tmp_path)
    assert len(calls) == 1 + len(webdata._RETRY_DELAYS)


def test_a_missing_record_is_not_retried(tmp_path: Path, monkeypatch) -> None:
    calls = _sequence(monkeypatch, _FakeResponse({}, 404))
    with pytest.raises(webdata.DataSourceError, match="Not found"):
        webdata.get_json("https://x/r", tmp_path)
    assert len(calls) == 1

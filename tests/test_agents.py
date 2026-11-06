"""Agent tests. They exercise the tool layer and the loop against a stub, never the API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from proteinscope.agents.runner import Agent, AgentError, _validate
from proteinscope.agents.tools import Workspace, build_tools, execute


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(out_dir=tmp_path)


def test_tool_definitions_are_well_formed(workspace: Workspace) -> None:
    for tool in build_tools(workspace).values():
        schema = tool.input_schema
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) <= set(schema["properties"])
        assert tool.description.strip()


def test_unknown_tool_is_reported_not_raised(workspace: Workspace) -> None:
    payload, failed = execute(build_tools(workspace), "nope", {})
    assert failed and "Unknown tool" in json.loads(payload)["error"]


def test_tool_errors_become_results(workspace: Workspace) -> None:
    payload, failed = execute(build_tools(workspace), "pocket_detail", {"pdb_id": "1ABC", "pocket_id": 1})
    assert failed and "not been analysed" in json.loads(payload)["error"]


def test_analysis_flows_into_the_workspace(workspace: Workspace, toy_pdb: Path) -> None:
    tools = build_tools(workspace)
    digest = tools["analyze_structure"].run(str(toy_pdb))
    assert digest["identifier"] == "TOY1"
    assert "TOY1" in workspace.analyses
    detail = tools["pocket_detail"].run("toy1", 1)
    assert detail["pocket_id"] == 1 and detail["probes"]


def test_write_report_records_the_author(workspace: Workspace, toy_pdb: Path) -> None:
    tools = build_tools(workspace)
    tools["analyze_structure"].run(str(toy_pdb))
    result = tools["write_report"].run("TOY1", "A summary.", recommendations=["Do the thing."])
    assert Path(result["pdf"]).is_file()
    assert workspace.narrative.source == "Claude, from the computed data"
    assert "A summary." in Path(result["markdown"]).read_text(encoding="utf-8")


def test_validate_catches_truncated_tool_input(workspace: Workspace) -> None:
    tool = build_tools(workspace)["analyze_structure"]
    _validate(tool, {"source": "1HSG", "probe_pockets": 3, "max_pockets": 12})
    with pytest.raises(ValueError, match="missing required"):
        _validate(tool, {"source": "1HSG"})
    with pytest.raises(ValueError, match="expected"):
        _validate(tool, {"source": 5, "probe_pockets": 3, "max_pockets": 12})


class _Block:
    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


class _Response:
    def __init__(self, content: list[Any], stop_reason: str) -> None:
        self.content = content
        self.stop_reason = stop_reason
        self.stop_details = None
        self.usage = _Block(input_tokens=10, output_tokens=5, cache_read_input_tokens=0)


class _StubAgent(Agent):
    """Agent with the API call replaced by a scripted list of responses."""

    def __init__(self, responses: list[_Response], **kwargs: Any) -> None:
        super().__init__(client=_Block(), **kwargs)
        self.responses = responses

    def _request(self) -> Any:
        return self.responses.pop(0)


def test_loop_executes_tools_then_stops(workspace: Workspace, toy_pdb: Path) -> None:
    tools = build_tools(workspace)
    call = _Block(type="tool_use", id="toolu_1", name="analyze_structure",
                  input={"source": str(toy_pdb), "probe_pockets": 1, "max_pockets": 5})
    agent = _StubAgent(
        [_Response([call], "tool_use"), _Response([_Block(type="text", text="Done.")], "end_turn")],
        name="test", system="s", tools={"analyze_structure": tools["analyze_structure"]},
    )
    turn = agent.run("analyse it")
    assert turn.text == "Done."
    assert turn.tool_calls == ["analyze_structure"]
    assert "TOY1" in workspace.analyses
    results = agent.messages[2]["content"]
    assert results[0]["is_error"] is False


def test_loop_returns_invalid_input_to_the_model(workspace: Workspace) -> None:
    tools = build_tools(workspace)
    call = _Block(type="tool_use", id="toolu_1", name="analyze_structure", input={"probe_pockets": 1})
    agent = _StubAgent(
        [_Response([call], "tool_use"), _Response([_Block(type="text", text="ok")], "end_turn")],
        name="test", system="s", tools={"analyze_structure": tools["analyze_structure"]},
    )
    agent.run("go")
    result = agent.messages[2]["content"][0]
    assert result["is_error"] and "INVALID_INPUT" in result["content"]


def test_refusal_raises_before_content_is_read(workspace: Workspace) -> None:
    response = _Response([], "refusal")
    response.stop_details = _Block(type="refusal", category="cyber")
    agent = _StubAgent([response], name="test", system="s", tools={})
    with pytest.raises(AgentError, match="declined"):
        agent.run("go")


def test_step_limit_is_enforced(workspace: Workspace, toy_pdb: Path) -> None:
    call = _Block(type="tool_use", id="t", name="entry_info", input={"pdb_id": "1HSG"})
    agent = _StubAgent(
        [_Response([call], "tool_use") for _ in range(3)],
        name="test", system="s", tools={},
    )
    with pytest.raises(AgentError, match="stopped after"):
        agent.run("go", max_steps=2)

"""Minimal agent runtime on the Anthropic Messages API: one tool loop per agent."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
_MAX_TOKENS = 64000
_MAX_MALFORMED_RETRIES = 2
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models that accept fallbacks="default". Other models simply run without it.
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}


class AgentError(RuntimeError):
    """An agent could not finish its task."""


class AgentRefusal(AgentError):
    """The model declined the request, after any configured fallback also declined."""

    def __init__(self, agent: str, category: str | None) -> None:
        self.agent = agent
        self.category = category
        super().__init__(f"Agent {agent!r} was declined by the model (category: {category or 'unspecified'}).")


@dataclass
class Tool:
    """A client-side tool: validated pydantic input, handler returning JSON-serialisable data."""

    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[BaseModel], Any]

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
            # Inputs stream as generated, so run() validates them itself before any handler call.
            "eager_input_streaming": True,
        }


@dataclass
class RunLog:
    """Trace and token usage shared by all agents in one run."""

    events: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    on_event: Callable[[dict[str, Any]], None] | None = None

    def record(self, **event: Any) -> None:
        self.events.append(event)
        if self.on_event:
            self.on_event(event)

    def add_usage(self, usage: Any) -> None:
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0


@dataclass
class AgentConfig:
    model: str = field(default_factory=lambda: os.environ.get("PROTEINSCOPE_MODEL", DEFAULT_MODEL))
    effort: str = DEFAULT_EFFORT
    max_turns: int = 15


class Agent:
    def __init__(
        self,
        name: str,
        system: str,
        tools: list[Tool],
        client: anthropic.Anthropic,
        config: AgentConfig,
        log: RunLog,
    ) -> None:
        self.name = name
        self.system = system
        self.tools = {tool.name: tool for tool in tools}
        self.client = client
        self.config = config
        self.log = log

    def _request(self, messages: list[dict[str, Any]]) -> Any:
        """Send one streamed request and return the final message."""
        params: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": _MAX_TOKENS,
            "system": self.system,
            "messages": messages,
            "tools": [tool.definition() for tool in self.tools.values()],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.config.effort},
            "cache_control": {"type": "ephemeral"},
        }
        if self.config.model in _FALLBACK_MODELS:
            params["betas"] = [_FALLBACK_BETA]
            params["fallbacks"] = "default"

        for attempt in range(_MAX_MALFORMED_RETRIES + 1):
            try:
                with self.client.beta.messages.stream(**params) as stream:
                    return stream.get_final_message()
            except ValueError as exc:  # The SDK's error for tool JSON it could not parse at all.
                if attempt == _MAX_MALFORMED_RETRIES:
                    raise AgentError(f"Agent {self.name!r} kept producing malformed tool input") from exc
                logger.warning("Malformed tool input from %s, retrying: %s", self.name, exc)
        raise AssertionError("unreachable")

    def _execute(self, block: Any) -> dict[str, Any]:
        result: dict[str, Any] = {"type": "tool_result", "tool_use_id": block.id}
        tool = self.tools.get(block.name)
        if tool is None:
            self.log.record(agent=self.name, type="tool_error", tool=block.name, error="unknown tool")
            return {**result, "is_error": True, "content": f"Unknown tool {block.name!r}"}
        try:
            parsed = tool.input_model.model_validate(block.input)
        except ValidationError as exc:
            # Do not run the tool on malformed input; Claude sees the problem and can retry.
            self.log.record(agent=self.name, type="tool_error", tool=block.name, error="invalid input")
            return {
                **result,
                "is_error": True,
                "content": json.dumps({"INVALID_JSON": block.input, "errors": exc.errors(include_url=False, include_context=False)}, default=str),
            }
        self.log.record(agent=self.name, type="tool_call", tool=block.name, input=parsed.model_dump())
        try:
            output = tool.handler(parsed)
        except AgentRefusal as exc:
            return {**result, "is_error": True, "content": str(exc)}
        except (ValueError, KeyError, RuntimeError, OSError) as exc:
            # RCSB outages, bad IDs and the like are for the agent to work around.
            self.log.record(agent=self.name, type="tool_error", tool=block.name, error=str(exc))
            return {**result, "is_error": True, "content": str(exc)}
        content = output if isinstance(output, str) else json.dumps(output, default=str)
        return {**result, "content": content}

    def run(self, task: str) -> str:
        """Run the tool loop on `task` until the model finishes; returns its final text."""
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        self.log.record(agent=self.name, type="start", task=task)
        for turn in range(self.config.max_turns):
            response = self._request(messages)
            self.log.add_usage(response.usage)
            stop = response.stop_reason

            if stop == "refusal":
                details = getattr(response, "stop_details", None)
                category = getattr(details, "category", None)
                self.log.record(agent=self.name, type="refusal", category=category)
                raise AgentRefusal(self.name, category)
            if stop == "max_tokens":
                # A truncated tool input can still parse, so this turn's tools must not run.
                raise AgentError(f"Agent {self.name!r} hit the output limit of {_MAX_TOKENS} tokens")

            # The whole content goes back, thinking and fallback blocks included.
            messages.append({"role": "assistant", "content": response.content})
            if stop == "pause_turn":
                continue
            calls = [block for block in response.content if block.type == "tool_use"]
            if stop != "tool_use" or not calls:
                text = "\n".join(b.text for b in response.content if b.type == "text").strip()
                self.log.record(agent=self.name, type="finish", turns=turn + 1, text=text)
                return text
            # All results of one turn go back in a single user message.
            messages.append({"role": "user", "content": [self._execute(block) for block in calls]})
        raise AgentError(f"Agent {self.name!r} did not finish within {self.config.max_turns} turns")

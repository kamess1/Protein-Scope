"""Agent loop over the Claude Messages API.

One Agent is a role: a system prompt plus the tools it may call. It streams its turn,
executes tool calls and loops until Claude stops calling tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import anthropic

from .tools import Tool, execute

MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOKENS = 32000
MAX_STEPS = 24


class AgentError(RuntimeError):
    """The agent could not complete its turn."""


@dataclass
class Event:
    """Something worth showing a user while the agent works."""

    kind: str  # "text" | "tool" | "tool_result" | "status"
    text: str
    data: dict[str, Any] = field(default_factory=dict)


Listener = Callable[[Event], None]


@dataclass
class Turn:
    text: str
    messages: list[dict[str, Any]]
    tool_calls: list[str]
    stop_reason: str | None
    usage: dict[str, int]


def _validate(tool: Tool, arguments: Any) -> None:
    """Check a tool input against its schema. Eager streaming means the API does not."""
    schema = tool.input_schema
    if not isinstance(arguments, dict):
        raise ValueError(f"input is {type(arguments).__name__}, expected an object")
    missing = [key for key in schema["required"] if key not in arguments]
    if missing:
        raise ValueError(f"missing required field(s): {', '.join(missing)}")
    kinds = {
        "string": str, "integer": int, "number": (int, float), "boolean": bool,
        "array": list, "object": dict,
    }
    for key, value in arguments.items():
        declared = schema["properties"].get(key, {}).get("type")
        allowed = [declared] if isinstance(declared, str) else declared
        if not allowed or value is None and "null" in allowed:
            continue
        expected = tuple(kinds[name] for name in allowed if name in kinds)
        if expected and not isinstance(value, expected):
            raise ValueError(f"field {key!r} is {type(value).__name__}, expected {'/'.join(allowed)}")


class Agent:
    """A single Claude role with its own system prompt, tools and conversation."""

    def __init__(
        self,
        name: str,
        system: str,
        tools: dict[str, Tool],
        client: anthropic.Anthropic | None = None,
        model: str = MODEL,
        effort: str = "high",
        listener: Listener | None = None,
    ) -> None:
        self.name = name
        self.system = system
        self.tools = tools
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.listener = listener
        self.messages: list[dict[str, Any]] = []
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}

    def _emit(self, kind: str, text: str, **data: Any) -> None:
        if self.listener:
            self.listener(Event(kind, text, data))

    def _tool_params(self) -> list[dict[str, Any]]:
        definitions = []
        for tool in self.tools.values():
            definition = tool.definition()
            if tool.name == "write_report":
                # Its arguments are long prose, so stream them instead of waiting for the
                # server to buffer the whole thing. Eager input is validated in this file.
                definition.pop("strict", None)
                definition["eager_input_streaming"] = True
            definitions.append(definition)
        return definitions

    def _request(self) -> Any:
        system = [{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}]
        try:
            with self.client.beta.messages.stream(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=self.messages,
                tools=self._tool_params(),
                thinking={"type": "adaptive", "display": "summarized"},
                output_config={"effort": self.effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            ) as stream:
                for event in stream:
                    if event.type == "content_block_stop" and getattr(event, "content_block", None) is not None:
                        block = event.content_block
                        if block.type == "thinking" and getattr(block, "thinking", ""):
                            self._emit("status", block.thinking)
                        elif block.type == "text" and block.text.strip():
                            self._emit("text", block.text)
                return stream.get_final_message()
        except ValueError as exc:  # Unparseable eagerly streamed tool input.
            raise AgentError(f"{self.name}: the model sent a malformed tool call ({exc})") from exc

    def run(self, prompt: str, max_steps: int = MAX_STEPS) -> Turn:
        self.messages.append({"role": "user", "content": prompt})
        called: list[str] = []
        response = None

        for _ in range(max_steps):
            response = self._request()
            for key in self.usage:
                self.usage[key] += getattr(response.usage, key, 0) or 0

            if response.stop_reason == "refusal":
                detail = getattr(response, "stop_details", None)
                category = getattr(detail, "category", None) or "unspecified"
                raise AgentError(f"{self.name}: the request was declined (category: {category}).")
            if response.stop_reason == "max_tokens":
                raise AgentError(f"{self.name}: the response hit the token limit before finishing.")

            self.messages.append({"role": "assistant", "content": response.content})
            calls = [block for block in response.content if block.type == "tool_use"]
            if not calls:
                break

            results = []
            for call in calls:
                called.append(call.name)
                self._emit("tool", f"{call.name}", tool=call.name, input=call.input)
                tool = self.tools.get(call.name)
                if tool is not None:
                    try:
                        _validate(tool, call.input)
                    except ValueError as exc:
                        results.append({
                            "type": "tool_result", "tool_use_id": call.id, "is_error": True,
                            "content": json.dumps({"INVALID_INPUT": str(exc)}),
                        })
                        continue
                payload, failed = execute(self.tools, call.name, call.input)
                self._emit("tool_result", f"{call.name}: {'error' if failed else 'ok'}", tool=call.name)
                results.append({
                    "type": "tool_result", "tool_use_id": call.id, "is_error": failed, "content": payload,
                })
            self.messages.append({"role": "user", "content": results})
        else:
            raise AgentError(f"{self.name}: stopped after {max_steps} steps without finishing.")

        text = "\n\n".join(block.text for block in response.content if block.type == "text")
        return Turn(text, self.messages, called, response.stop_reason, dict(self.usage))


def subset(tools: dict[str, Tool], names: Iterable[str]) -> dict[str, Tool]:
    return {name: tools[name] for name in names if name in tools}

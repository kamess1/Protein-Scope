"""MCP server: exposes the ProteinScope tools to an MCP client such as Claude Code.

The client's model becomes the orchestrator, so no Anthropic API key is needed here. Run it
with `proteinscope mcp` (stdio transport); `.mcp.json` in the repository registers it with
Claude Code.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Callable

from .agents.tools import ToolError, Workspace, build_tools

INSTRUCTIONS = """\
ProteinScope analyses protein structures from the PDB, AlphaFold DB or local files.
Typical flow: search_pdb (or entry_info) -> analyze_structure -> pocket_detail /
ligand_detail / ligand_chemistry / target_bioactivity -> write_report.
Every number comes from the tools; never invent values. Pocket druggability is a
hand-weighted heuristic that ranks pockets within one structure, not a probability.
Identifiers: 4-character PDB IDs (1HSG), AlphaFold models as AF-<UniProt> (AF-P00533),
or a path to a .pdb/.cif file."""


def _wrap(run: Callable[..., Any]) -> Callable[..., Any]:
    """Turn the package's ToolError into the MCP SDK's, so the client sees a tool error."""
    from mcp.server.mcpserver.exceptions import ToolError as MCPToolError

    @functools.wraps(run)
    def call(*args: Any, **kwargs: Any) -> Any:
        try:
            return run(*args, **{k: v for k, v in kwargs.items() if v is not None})
        except ToolError as exc:
            raise MCPToolError(str(exc)) from exc

    return call


def build_server(out_dir: Path | str | None = None) -> Any:
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError('The MCP SDK is not installed. Run: pip install "proteinscope[mcp]"') from exc

    out = Path(out_dir or os.environ.get("PROTEINSCOPE_OUT", "output")).expanduser().resolve()
    workspace = Workspace(out_dir=out)
    tools = build_tools(workspace, narrative_source="Claude via MCP, from the computed data")
    server = MCPServer("proteinscope", instructions=INSTRUCTIONS, version="0.2.0")
    for tool in tools.values():
        server.add_tool(_wrap(tool.run), name=tool.name, description=tool.description)
    server.workspace = workspace  # Exposed for tests and debugging.
    return server


def main(out_dir: str | None = None) -> None:
    build_server(out_dir).run("stdio")


if __name__ == "__main__":
    main()

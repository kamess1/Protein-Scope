"""The multi-agent workflow: Claude orchestrates specialised roles over one workspace."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

from .runner import Agent, AgentError, Event, Listener, Turn, subset
from .tools import Tool, Workspace, build_tools

SHARED_RULES = """
You work on experimentally determined protein structures. Rules that override any instinct
to be helpful:
- Every number you state must come from a tool result in this conversation. Never estimate,
  recall or interpolate a value, and never name a residue you have not seen in a tool result.
- The druggability score is a hand-weighted heuristic that ranks pockets within one structure.
  It is not a probability and not comparable across proteins. Say so whenever you use it.
- Electrostatic potentials are screened-Coulomb estimates, hydrogen bonds and salt bridges are
  geometric candidates without hydrogens, and probe energies are rigid-grid estimates, not
  binding free energies. Carry these limits into what you write.
- A crystal structure is one conformation. Absent loops, crystallisation additives and missing
  biological assembly partners can create or hide pockets. When the audit warns about these,
  the warning belongs in your conclusions.
- If the data do not support a conclusion, say what is missing instead of hedging.
""".strip()

SCOUT_PROMPT = f"""
You are the structure scout. You find the right PDB entries for a research question.

{SHARED_RULES}

Work like this: search with the terms the user gave, then narrow with filters (resolution,
bound ligand, organism) rather than judging by title alone. Check promising entries with
entry_info. Prefer high resolution, a low R-free and a bound ligand when the goal is a binding
site. Note when the biological assembly is larger than the deposited model.

Answer with a short ranked list: PDB ID, one line on what it is, and the reason it fits or
does not. Name your first choice explicitly. Do not analyse structures; that is another agent's
job.
""".strip()

ANALYST_PROMPT = f"""
You are the structure analyst. You run the computational analysis and interpret it.

{SHARED_RULES}

Work like this: call analyze_structure once per structure. Read the audit before the pockets,
because the audit decides how much the pockets can be trusted. Use pocket_detail on the pockets
that matter to see the full lining and the probe energies, and ligand_detail on bound ligands.
When a ligand sits in a pocket, compare what the pocket predicts with what the ligand actually
contacts: agreement supports the detection, disagreement is a finding worth reporting.

Read the probe results as chemistry: which probe is most favourable where, and what that
suggests a ligand needs at that position. A cation probe scoring well on an acidic pocket is
expected, not a discovery.

Answer with your findings in prose, with the numbers that support them. State clearly which
pockets you consider real binding sites and which look like surface grooves or crystallisation
artefacts.
""".strip()

REPORTER_PROMPT = f"""
You are the report writer. You turn the analysis into the final document.

{SHARED_RULES}

You receive the analyst's findings. Pull any number you need with pocket_detail and
ligand_detail, then call write_report exactly once.

Write for a structural biologist who has not seen this analysis:
- executive_summary: 3-5 sentences. What the structure is, what was found, what it means.
- pocket_interpretation: which sites are real and why, what the score components say, what the
  audit warnings do to your confidence.
- ligand_interpretation: how bound ligands engage the site, which contacts drive it.
- recommendations: concrete next steps. Name the structure, the site or the method; no generic
  advice such as "run further studies".

Prose, no Markdown headings or bullet syntax: the report template adds the structure. After
write_report returns, reply with one sentence naming the files written.
""".strip()

ORCHESTRATOR_PROMPT = f"""
You orchestrate a team that analyses protein structures and writes reports.

{SHARED_RULES}

Your team, reached with the delegate tool:
- scout: finds PDB entries for a question. Use it when the user names a protein or a goal
  rather than a specific entry.
- analyst: runs and interprets the analysis of a named structure or file.
- reporter: writes the final report files. It can only report on structures the analyst has
  already analysed.

Delegate with full context: a teammate sees only the instruction you send, not this
conversation. Say which structure, which question, and what the previous teammate found.

Plan first, in one or two sentences. Then delegate. Send the reporter last, and only when the
user wants report files. For a question that needs no new analysis, answer it yourself from
what the teammates reported.

Finish with a short answer for the user: what was done, the key findings with their numbers,
and where the files are. Do not repeat the whole report.
""".strip()


@dataclass
class CrewResult:
    answer: str
    workspace: Workspace
    transcript: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


class Crew:
    """Claude as orchestrator over the scout, analyst and reporter roles."""

    def __init__(
        self,
        out_dir: Path | str = "output",
        client: anthropic.Anthropic | None = None,
        listener: Listener | None = None,
        model: str = "claude-opus-5-5",
    ) -> None:
        self.workspace = Workspace(out_dir=Path(out_dir))
        self.client = client or anthropic.Anthropic()
        self.listener = listener
        self.model = model
        self.tools = build_tools(self.workspace)
        self.members: dict[str, Agent] = {}

    def _member(self, role: str) -> Agent:
        if role not in self.members:
            config = {
                "scout": (SCOUT_PROMPT, ["search_pdb", "entry_info"], "medium"),
                "analyst": (ANALYST_PROMPT, [
                    "analyze_structure", "pocket_detail", "ligand_detail", "compare_structures", "entry_info",
                ], "high"),
                "reporter": (REPORTER_PROMPT, ["pocket_detail", "ligand_detail", "write_report"], "high"),
            }[role]
            prompt, names, effort = config
            self.members[role] = Agent(
                name=role, system=prompt, tools=subset(self.tools, names),
                client=self.client, model=self.model, effort=effort,
                listener=self._nested_listener(role),
            )
        return self.members[role]

    def _nested_listener(self, role: str) -> Listener:
        def listen(event: Event) -> None:
            if self.listener:
                self.listener(Event(event.kind, event.text, {**event.data, "agent": role}))
        return listen

    def _delegate_tool(self) -> Tool:
        def delegate(agent: str, instruction: str) -> dict[str, Any]:
            role = agent.strip().lower()
            if role not in {"scout", "analyst", "reporter"}:
                return {"error": f"Unknown teammate {agent!r}; choose scout, analyst or reporter."}
            if self.listener:
                self.listener(Event("status", f"Delegating to {role}", {"agent": "orchestrator"}))
            try:
                turn: Turn = self._member(role).run(instruction)
            except AgentError as exc:
                return {"agent": role, "error": str(exc)}
            return {
                "agent": role,
                "report": turn.text,
                "tools_used": turn.tool_calls,
                "analysed_structures": sorted(self.workspace.analyses),
            }

        return Tool(
            name="delegate",
            description=(
                "Give a task to a teammate and receive their written report. 'scout' finds PDB "
                "entries, 'analyst' analyses a structure, 'reporter' writes the report files. "
                "The teammate sees only your instruction, so include all context it needs."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "agent": {"type": "string", "enum": ["scout", "analyst", "reporter"]},
                    "instruction": {"type": "string", "description": "The full task, with context"},
                },
                "required": ["agent", "instruction"],
                "additionalProperties": False,
            },
            run=delegate,
        )

    def run(self, request: str) -> CrewResult:
        lead = Agent(
            name="orchestrator",
            system=ORCHESTRATOR_PROMPT,
            tools={"delegate": self._delegate_tool()},
            client=self.client,
            model=self.model,
            effort="high",
            listener=self._nested_listener("orchestrator"),
        )
        turn = lead.run(request)
        usage = dict(lead.usage)
        for member in self.members.values():
            for key, value in member.usage.items():
                usage[key] = usage.get(key, 0) + value
        transcript = [
            {"agent": name, "messages": json.loads(json.dumps(member.messages, default=str))}
            for name, member in [("orchestrator", lead), *self.members.items()]
        ]
        return CrewResult(turn.text, self.workspace, transcript, usage)

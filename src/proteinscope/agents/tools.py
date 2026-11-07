"""Tool layer the agents call. Every tool returns JSON-ready data, never prose.

Tools work on a shared Workspace, so one agent's analysis is visible to the next.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .. import rcsb
from ..pipeline import AnalysisResult, run_analysis
from ..pockets import PocketParams
from ..report.builder import ReportPaths, analysis_digest, build_report
from ..report.narrative import Narrative
from ..webdata import DataSourceError

MAX_SEARCH_RESULTS = 25


class ToolError(RuntimeError):
    """A tool could not do what was asked; the message goes back to the model."""


@dataclass
class Workspace:
    """Holds analyses and outputs for one run, keyed by structure identifier."""

    out_dir: Path
    analyses: dict[str, AnalysisResult] = field(default_factory=dict)
    reports: dict[str, ReportPaths] = field(default_factory=dict)
    narrative: Narrative | None = None
    log: list[dict[str, Any]] = field(default_factory=list)

    def require(self, identifier: str) -> AnalysisResult:
        key = identifier.strip().upper()
        if key not in self.analyses:
            known = sorted(self.analyses) or ["none"]
            raise ToolError(f"{key} has not been analysed yet. Analysed so far: {', '.join(known)}")
        return self.analyses[key]


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    run: Callable[..., Any]

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "strict": True,
        }


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _context_summary(context: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"notes": context.get("notes", [])}
    if "plddt" in context:
        summary["plddt"] = context["plddt"]
    if "ligand_chemistry" in context:
        summary["ligands_with_chemistry"] = sorted(context["ligand_chemistry"])
    if "annotations" in context:
        summary["functional_annotation"] = {
            "proteins": [
                {"accession": p.get("accession"), "protein_name": p.get("protein_name"),
                 "genes": p.get("genes"), "mapped_residues": p.get("mapped_residues")}
                for p in context["annotations"]["proteins"]
            ],
            "pockets": {
                pocket_id: {"categories": row["categories"], "cofactors": row["cofactors"],
                            "is_functional_site": row["is_functional_site"], "regions": row["regions"]}
                for pocket_id, row in context["annotations"]["pockets"].items()
            },
        }
    if "targets" in context:
        summary["targets"] = [
            {
                "accession": row["accession"],
                "chembl_id": (row.get("chembl") or {}).get("target_chembl_id"),
                "measured_activities": row.get("measured_activities", 0),
                "approved_drugs": row.get("approved_drugs", 0),
            }
            for row in context["targets"]
        ]
    return summary


def build_tools(
    workspace: Workspace, narrative_source: str = "Claude, from the computed data"
) -> dict[str, Tool]:
    def search_pdb(
        query: str,
        max_results: int = 5,
        max_resolution: float | None = None,
        require_ligand: bool = False,
        organism: str | None = None,
    ) -> dict[str, Any]:
        try:
            hits = rcsb.search(
                query, min(max_results, MAX_SEARCH_RESULTS), max_resolution, require_ligand, organism
            )
        except rcsb.RCSBError as exc:
            raise ToolError(str(exc)) from exc
        return {"query": query, "count": len(hits), "results": hits}

    def entry_info(pdb_id: str) -> dict[str, Any]:
        try:
            return rcsb.entry_metadata(pdb_id)
        except (rcsb.RCSBError, ValueError) as exc:
            raise ToolError(str(exc)) from exc

    def analyze_structure(source: str, probe_pockets: int = 3, max_pockets: int = 12) -> dict[str, Any]:
        try:
            result = run_analysis(
                source, params=PocketParams(max_pockets=max(1, min(max_pockets, 20))),
                probe_pockets=max(0, min(probe_pockets, 6)),
            )
        except (ValueError, FileNotFoundError, rcsb.RCSBError, DataSourceError) as exc:
            raise ToolError(str(exc)) from exc
        workspace.analyses[result.identifier] = result
        digest = analysis_digest(result)
        # Full lining lists and contact tables are large; the report keeps them, the model
        # asks for them per pocket when it needs them.
        for pocket in digest["pockets"]:
            pocket["lining_residues"] = pocket["lining_residues"][:12]
        for ligand in digest["ligands"].values():
            ligand.pop("top_residue_vdw_kcal", None)
        # Target and chemistry context have their own tools; keep the headline here.
        context = digest.pop("context", {})
        digest["context_summary"] = _context_summary(context)
        return digest

    def pocket_detail(pdb_id: str, pocket_id: int) -> dict[str, Any]:
        result = workspace.require(pdb_id)
        try:
            pocket = result.pocket(pocket_id)
        except KeyError as exc:
            raise ToolError(str(exc)) from exc
        ligands = [
            label for label, entries in result.ligand_pockets.items()
            if any(entry["pocket_id"] == pocket_id for entry in entries)
        ]
        return {
            **pocket.summary(),
            "ligands_in_pocket": ligands,
            "probes": [probe.to_dict() for probe in result.probes_for(pocket_id)],
        }

    def ligand_detail(pdb_id: str, ligand: str) -> dict[str, Any]:
        result = workspace.require(pdb_id)
        profile = result.ligand_profiles.get(ligand)
        if profile is None:
            raise ToolError(
                f"{ligand!r} is not a ligand of {pdb_id}. Available: {sorted(result.ligand_profiles)}"
            )
        return {**profile.summary(), "pockets": result.ligand_pockets.get(ligand, [])}

    def ligand_chemistry(pdb_id: str) -> dict[str, Any]:
        result = workspace.require(pdb_id)
        chemistry = result.context.get("ligand_chemistry")
        if chemistry is None:
            raise ToolError(
                "No ligand chemistry for this structure. " + " ".join(result.context.get("notes", []))
            )
        return {label: {k: v for k, v in entry.items() if k != "svg"} for label, entry in chemistry.items()}

    def target_bioactivity(pdb_id: str) -> dict[str, Any]:
        result = workspace.require(pdb_id)
        rows = result.context.get("targets")
        if rows is None:
            raise ToolError(
                "No target context for this structure. " + " ".join(result.context.get("notes", []))
            )
        return {"targets": rows}

    def functional_sites(pdb_id: str, pocket_id: int | None = None) -> dict[str, Any]:
        result = workspace.require(pdb_id)
        annotations = result.context.get("annotations")
        if not annotations:
            raise ToolError(
                "No UniProt annotation for this structure. " + " ".join(result.context.get("notes", []))
            )
        pockets = annotations["pockets"]
        if pocket_id is None:
            return {"proteins": annotations["proteins"], "pockets": pockets}
        if pocket_id not in pockets:
            annotated = sorted(pockets) or ["none"]
            raise ToolError(
                f"Pocket {pocket_id} of {pdb_id} has no annotated residue. "
                f"Annotated pockets: {', '.join(str(p) for p in annotated)}"
            )
        return {"proteins": annotations["proteins"], "pocket": pockets[pocket_id]}

    def compare_structures(pdb_ids: list[str]) -> dict[str, Any]:
        rows = []
        for identifier in pdb_ids:
            result = workspace.require(identifier)
            top = result.pockets[0] if result.pockets else None
            rows.append({
                "pdb_id": result.identifier,
                "resolution_A": result.audit.resolution_A,
                "quality_grade": result.audit.quality_grade,
                "residues": result.audit.residue_count,
                "ligands": result.audit.ligands,
                "pocket_count": len(result.pockets),
                "top_pocket": top.summary() | {"lining_residues": top.lining_residues[:12]} if top else None,
            })
        return {"structures": rows}

    def write_report(
        pdb_id: str,
        executive_summary: str,
        pocket_interpretation: str = "",
        ligand_interpretation: str = "",
        recommendations: list[str] | None = None,
    ) -> dict[str, Any]:
        result = workspace.require(pdb_id)
        narrative = Narrative(
            executive_summary=executive_summary,
            pocket_interpretation=pocket_interpretation,
            ligand_interpretation=ligand_interpretation,
            recommendations=recommendations or [],
            source=narrative_source,
        )
        workspace.narrative = narrative
        paths = build_report(result, workspace.out_dir, narrative)
        workspace.reports[result.identifier] = paths
        return {
            "markdown": str(paths.markdown), "pdf": str(paths.pdf), "data": str(paths.data),
            "viewer": str(paths.viewer) if paths.viewer else None,
            "figures": sorted(paths.figures), "warnings": paths.warnings,
        }

    tools = [
        Tool(
            "search_pdb",
            "Search the RCSB PDB by free text, optionally filtered by resolution, bound ligand and "
            "organism. Returns entry IDs with title, method, resolution and ligands.",
            _schema({
                "query": {"type": "string", "description": "Free-text query, e.g. 'KRAS G12C inhibitor'"},
                "max_results": {"type": "integer", "description": "1-25, default 5"},
                "max_resolution": {"type": ["number", "null"], "description": "Upper bound in Angstrom"},
                "require_ligand": {"type": "boolean", "description": "Only entries with a bound non-polymer"},
                "organism": {"type": ["string", "null"], "description": "Exact source organism name"},
            }, ["query", "max_results", "max_resolution", "require_ligand", "organism"]),
            search_pdb,
        ),
        Tool(
            "entry_info",
            "Fetch metadata for one PDB entry: title, method, resolution, R factors, citation and the "
            "chain count of biological assembly 1. Does not download coordinates.",
            _schema({"pdb_id": {"type": "string"}}, ["pdb_id"]),
            entry_info,
        ),
        Tool(
            "analyze_structure",
            "Download and analyse a structure, given a PDB ID, an AlphaFold DB ID or a local file path. "
            "Runs the quality audit, surface and electrostatics, pocket detection with druggability "
            "ranking and site tags, ligand contact profiling and probe simulation, and fetches ligand "
            "chemistry and ChEMBL target data. Returns the numeric digest.",
            _schema({
                "source": {"type": "string", "description": "A PDB ID (1HSG), an AlphaFold ID (AF-P00533) or a path to a .pdb/.cif file"},
                "probe_pockets": {"type": "integer", "description": "How many top pockets to probe, 0-6, default 3"},
                "max_pockets": {"type": "integer", "description": "Maximum pockets to keep, 1-20, default 12"},
            }, ["source", "probe_pockets", "max_pockets"]),
            analyze_structure,
        ),
        Tool(
            "pocket_detail",
            "Full detail for one pocket of an already analysed structure: all lining residues, score "
            "components, probe energies and any ligand that sits in it.",
            _schema({"pdb_id": {"type": "string"}, "pocket_id": {"type": "integer"}}, ["pdb_id", "pocket_id"]),
            pocket_detail,
        ),
        Tool(
            "ligand_detail",
            "Full contact profile of one bound ligand, including every polar contact with distances "
            "and the per-residue van der Waals energies.",
            _schema({
                "pdb_id": {"type": "string"},
                "ligand": {"type": "string", "description": "Ligand label as reported, e.g. 'MK1902:B'"},
            }, ["pdb_id", "ligand"]),
            ligand_detail,
        ),
        Tool(
            "ligand_chemistry",
            "Drug-likeness of the bound ligands of an analysed structure: SMILES from the PDB "
            "Chemical Component Dictionary and RDKit descriptors (MW, cLogP, HBD/HBA, TPSA, QED, "
            "Lipinski and Veber violations, rule of three).",
            _schema({"pdb_id": {"type": "string"}}, ["pdb_id"]),
            ligand_chemistry,
        ),
        Tool(
            "target_bioactivity",
            "What ChEMBL knows about the protein of an analysed structure: UniProt accession, ChEMBL "
            "target, number of measured potencies, the most potent compounds and drugs with a known "
            "mechanism. Use it to put the geometric druggability score into context.",
            _schema({"pdb_id": {"type": "string"}}, ["pdb_id"]),
            target_bioactivity,
        ),
        Tool(
            "functional_sites",
            "UniProt annotation of an analysed structure, mapped onto its residues by sequence "
            "alignment: active and cofactor binding sites, modifications, disulfides, disease "
            "variants and domains, grouped per pocket. Use it to tell whether a well-shaped pocket "
            "is the functional site, and which pockets carry disease mutations.",
            _schema({
                "pdb_id": {"type": "string"},
                "pocket_id": {"type": ["integer", "null"], "description": "One pocket, or null for all"},
            }, ["pdb_id", "pocket_id"]),
            functional_sites,
        ),
        Tool(
            "compare_structures",
            "Side-by-side comparison of structures that have already been analysed.",
            _schema({"pdb_ids": {"type": "array", "items": {"type": "string"}}}, ["pdb_ids"]),
            compare_structures,
        ),
        Tool(
            "write_report",
            "Write the final report as Markdown and PDF, with figures, an interactive 3D viewer and a "
            "JSON data file. Your text is inserted; all numbers and figures come from the analysis. "
            "Call this once the analysis is complete.",
            _schema({
                "pdb_id": {"type": "string"},
                "executive_summary": {"type": "string", "description": "2-5 sentences for a reader who has not seen the data"},
                "pocket_interpretation": {"type": "string", "description": "What the pocket results mean; may be empty"},
                "ligand_interpretation": {"type": "string", "description": "What the ligand contacts mean; may be empty"},
                "recommendations": {"type": "array", "items": {"type": "string"}},
            }, ["pdb_id", "executive_summary", "pocket_interpretation", "ligand_interpretation", "recommendations"]),
            write_report,
        ),
    ]
    return {tool.name: tool for tool in tools}


def execute(tools: dict[str, Tool], name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
    """Run a tool by name; returns (JSON text for the model, is_error)."""
    tool = tools.get(name)
    if tool is None:
        return json.dumps({"error": f"Unknown tool {name!r}"}), True
    try:
        # The model may send null for optional arguments; the Python defaults should win.
        cleaned = {key: value for key, value in arguments.items() if value is not None}
        return json.dumps(tool.run(**cleaned), default=str), False
    except ToolError as exc:
        return json.dumps({"error": str(exc)}), True
    except TypeError as exc:
        return json.dumps({"error": f"Invalid arguments for {name}: {exc}"}), True

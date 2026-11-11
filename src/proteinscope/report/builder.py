"""Assemble Markdown and PDF reports. Both are built from the same content model."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from Bio.PDB.PDBExceptions import PDBIOException
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    CondPageBreak,
    KeepTogether,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .. import __version__, chemistry
from ..druggability import SCORE_WEAKNESS, score_description
from ..sources import BY_NAME as SOURCE_BY_NAME
from ..enrich import CATEGORY_LABELS, annotation_summary
from ..pipeline import AnalysisResult
from ..viz.figures import generate_figures
from ..viz.viewer import write_html_viewer, write_pymol_session
from .html import write_html_report
from .narrative import Narrative, rule_based_narrative

METHOD_NOTES = [
    "Pockets: LIGSITE-style grid scan (1 A grid, 7 scan lines, 14 A range) on the protein only, so bound ligands do not influence detection.",
    score_description(),
    SCORE_WEAKNESS,
    "Electrostatics: screened Coulomb sum with formal pH 7 charges and a distance-dependent dielectric. It indicates sign and rough size only; use APBS for quantitative potentials.",
    "Ligand contacts: heavy-atom geometry without hydrogens or ligand protonation states, so hydrogen bonds and salt bridges are candidates.",
    "Probe simulation: rigid protein, Lennard-Jones plus Coulomb plus an H-bond well, without desolvation or entropy. Values compare probes and pockets and are not binding free energies.",
    "Site tags: rule-based labels (size, enclosure, polarity, mean potential beyond +/-2.5 kT/e) that flag fragment-sized or charged anchor sites the drug-likeness score ranks low.",
    "Ligand chemistry: SMILES from the wwPDB Chemical Component Dictionary, descriptors from RDKit (Crippen cLogP, QED). Rule-of-five and Veber checks are filters, not verdicts.",
    "Target context: UniProt mapping from RCSB and bioactivity from ChEMBL. Known potent ligands and approved drugs are stronger evidence of druggability than any geometric score.",
    "Functional annotation: UniProtKB features (active and binding sites, modifications, disulfides, disease variants, mutagenesis) mapped onto the observed residues by BLOSUM62 alignment of each chain against the reference sequence, so renumbered or truncated constructs stay correct. Features are curated evidence, not measurements, and absence of annotation is not absence of function.",
    "AlphaFold models: the B-factor column holds pLDDT. Pockets lined by residues below pLDDT 70 are unreliable, and side chains of apo predictions may close induced-fit pockets.",
]


@dataclass
class ReportPaths:
    directory: Path
    markdown: Path
    pdf: Path
    data: Path
    viewer: Path | None
    pymol: Path | None
    figures: dict[str, Path]
    warnings: list[str]
    html: Path | None = None


def analysis_digest(result: AnalysisResult) -> dict[str, Any]:
    """All computed values as JSON-ready data; also what agents read."""
    return {
        "identifier": result.identifier,
        "metadata": result.metadata,
        "audit": result.audit.to_dict(),
        "surface": result.surface.summary.to_dict(),
        "net_charge_estimate": round(result.charge_model.net_charge, 2),
        "pockets": [pocket.summary() for pocket in result.pockets],
        "ligand_pockets": result.ligand_pockets,
        "ligands": {label: profile.summary() for label, profile in result.ligand_profiles.items()},
        "probes": {
            str(pocket_id): [probe.to_dict() for probe in probes]
            for pocket_id, probes in sorted(result.probes.items())
        },
        "context": _json_context(result.context),
        "provenance": result.provenance,
    }


def _json_context(context: dict[str, Any]) -> dict[str, Any]:
    """The context without inline SVG drawings, which the HTML report renders itself."""
    clean = dict(context)
    if "ligand_chemistry" in clean:
        clean["ligand_chemistry"] = {
            label: {k: v for k, v in entry.items() if k != "svg"}
            for label, entry in clean["ligand_chemistry"].items()
        }
    return clean


def _fmt(value: Any, digits: int = 2, unit: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}{unit}"
    return f"{value}{unit}"


def _summary_rows(result: AnalysisResult) -> list[list[str]]:
    audit = result.audit
    surface = result.surface.summary
    if audit.is_predicted_model:
        confidence = [
            ["Mean pLDDT", _fmt(audit.mean_plddt, 1)],
            ["Residues with pLDDT < 50", str(len(audit.low_confidence_residues))],
        ]
    else:
        confidence = [
            ["Resolution", _fmt(audit.resolution_A, 2, " A")],
            ["R-free / R-work", f"{_fmt(audit.r_free, 3)} / {_fmt(audit.r_work, 3)}"],
        ]
    return [
        ["Method", _fmt(audit.method)],
        *confidence,
        ["Quality grade", audit.quality_grade],
        ["Chains (residues)", ", ".join(f"{c} ({n})" for c, n in audit.chains.items())],
        ["Heavy atoms", str(audit.heavy_atom_count)],
        ["Unobserved residues / chain breaks", f"{audit.unobserved_residue_count} / {len(audit.chain_breaks)}"],
        ["Bound ligands", ", ".join(audit.ligands) or "none"],
        ["Ions / additives", f"{len(audit.ions)} / {len(audit.additives)}"],
        ["Maximum diameter", _fmt(surface.max_diameter, 1, " A")],
        ["Radius of gyration", _fmt(surface.radius_of_gyration, 1, " A")],
        ["Total SASA (apolar fraction)", f"{surface.total_sasa:.0f} A2 ({surface.apolar_fraction:.0%})"],
        ["Exposed residues", f"{surface.exposed_residue_fraction:.0%}"],
        ["Net charge estimate, pH 7", f"{result.charge_model.net_charge:+.1f} e"],
        ["Hydrophobic surface patches", str(len(surface.patches))],
    ]


def _pocket_rows(result: AnalysisResult) -> list[list[str]]:
    plddt = (result.context.get("plddt") or {}).get("pockets", {})
    annotations = ((result.context.get("annotations") or {}).get("pockets") or {})
    head = ["#", "Volume (A3)", "Enclosure", "Depth (A)", "Apolar", "Potential (kT/e)", "Score", "Class"]
    rows = [head + (["pLDDT"] if plddt else []) + ["Site"] + (["Function"] if annotations else [])]
    for pocket in result.pockets:
        row = [
            str(pocket.pocket_id), f"{pocket.volume:.0f}", f"{pocket.enclosure:.2f}", f"{pocket.depth:.1f}",
            f"{pocket.apolar_fraction:.2f}", f"{pocket.mean_potential:+.1f}", f"{pocket.druggability:.2f}",
            pocket.druggability_class,
        ]
        if plddt:
            row.append(_fmt((plddt.get(pocket.pocket_id) or {}).get("mean_plddt"), 0))
        row.append("; ".join(pocket.site_tags) or "-")
        if annotations:
            row.append(annotation_summary(annotations.get(pocket.pocket_id)))
        rows.append(row)
    return rows


def _annotation_rows(result: AnalysisResult, limit: int = 8) -> list[list[str]]:
    """Which annotated residues line which pocket, most informative pockets first."""
    annotations = ((result.context.get("annotations") or {}).get("pockets") or {})
    rows = [["Pocket", "Residue", "UniProt", "Annotation"]]
    for pocket_id, row in sorted(annotations.items()):
        # Functional features first; variants alone rarely explain what a pocket does.
        residues = sorted(
            row["residues"],
            key=lambda r: min(list(CATEGORY_LABELS).index(f["category"])
                              if f["category"] in CATEGORY_LABELS else 9 for f in r["features"]),
        )
        for residue in residues[:limit]:
            feature = residue["features"][0]
            text = feature["description"] or feature["type"]
            rows.append([
                str(pocket_id), residue["residue"], str(residue["uniprot_position"]),
                f"{feature['type']}: {text}"[:90] if feature["description"] else feature["type"],
            ])
        if len(residues) > limit:
            rows.append([str(pocket_id), f"+{len(residues) - limit} more", "", ""])
    return rows if len(rows) > 1 else []


def _protein_rows(result: AnalysisResult) -> list[list[str]]:
    rows = [["UniProt", "Protein", "Gene", "Length", "Mapped residues", "Pfam"]]
    for protein in ((result.context.get("annotations") or {}).get("proteins") or []):
        rows.append([
            protein.get("accession") or "-", (protein.get("protein_name") or "")[:46],
            ", ".join(protein.get("genes") or []) or "-", _fmt(protein.get("length"), 0),
            str(protein.get("mapped_residues", 0)),
            ", ".join(f"{p['id']} {p['name']}" for p in (protein.get("pfam") or [])[:3]) or "-",
        ])
    return rows if len(rows) > 1 else []


def _chemistry_rows(result: AnalysisResult) -> list[list[str]]:
    chemistry = result.context.get("ligand_chemistry") or {}
    rows = [["Ligand", "MW", "cLogP", "HBD/HBA", "TPSA", "RotB", "QED", "Rule-of-five", "Name"]]
    seen = set()
    for label, entry in chemistry.items():
        props = entry.get("properties")
        if not props or entry["comp_id"] in seen:
            continue
        seen.add(entry["comp_id"])
        violations = props["lipinski_violations"]
        rows.append([
            label, f"{props['molecular_weight']:.0f}", f"{props['clogp']:.1f}", f"{props['hbd']}/{props['hba']}",
            f"{props['tpsa']:.0f}", str(props["rotatable_bonds"]), f"{props['qed']:.2f}",
            "pass" if not violations else ", ".join(violations), (entry.get("name") or "")[:60],
        ])
    return rows if len(rows) > 1 else []


def _target_rows(result: AnalysisResult) -> list[list[str]]:
    rows = [["UniProt", "Protein", "ChEMBL target", "Measured potencies", "Drugs (approved)"]]
    for row in result.context.get("targets") or []:
        chembl = row.get("chembl") or {}
        drugs = row.get("drugs") or []
        rows.append([
            row["accession"], (row.get("description") or "")[:50],
            f"{chembl.get('target_chembl_id', '-')} {chembl.get('pref_name') or ''}".strip(),
            str(row.get("measured_activities", 0)), f"{len(drugs)} ({row.get('approved_drugs', 0)})",
        ])
    return rows if len(rows) > 1 else []


def _top_compound_rows(result: AnalysisResult, limit: int = 6) -> list[list[str]]:
    rows = [["Compound", "Name", "pChEMBL", "Assay type"]]
    for row in result.context.get("targets") or []:
        for compound in (row.get("top_compounds") or [])[:limit]:
            rows.append([
                compound["molecule_chembl_id"], compound.get("name") or "-",
                _fmt(compound.get("pchembl"), 1), compound.get("type") or "-",
            ])
    return rows if len(rows) > 1 else []


def _ligand_rows(result: AnalysisResult) -> list[list[str]]:
    rows = [["Ligand", "Atoms", "Buried", "Pocket", "H-bond", "Salt", "Hydrophobic", "Stack", "Metal"]]
    for label, profile in result.ligand_profiles.items():
        counts = profile.counts()
        stacks = sum(v for k, v in counts.items() if k.startswith("pi-stack"))
        hits = result.ligand_pockets.get(label, [])
        rows.append([
            label, str(profile.atom_count), f"{profile.buried_fraction:.0%}",
            str(hits[0]["pocket_id"]) if hits else "-",
            str(counts.get("hydrogen bond", 0)), str(counts.get("salt bridge (candidate)", 0)),
            str(counts.get("hydrophobic", 0)), str(stacks), str(counts.get("metal coordination", 0)),
        ])
    return rows


def _probe_rows(result: AnalysisResult, pocket_id: int) -> list[list[str]]:
    rows = [["Probe", "Best E (kcal/mol)", "Favorable points", "Residues at best position"]]
    for probe in result.probes_for(pocket_id):
        rows.append([
            probe.probe, f"{probe.best_energy:.1f}", f"{probe.favorable_fraction:.0%}",
            ", ".join(probe.hotspot_residues[:5]),
        ])
    return rows


def _provenance_rows(result: AnalysisResult) -> list[list[str]]:
    """Who computed this, with what, from which input. Values only, never local paths."""
    prov = result.provenance
    if not prov:
        return []
    model = prov.get("pocket_model") or {}
    structure = prov.get("structure") or {}
    libraries = ", ".join(f"{name} {version}" for name, version in (prov.get("libraries") or {}).items() if version)
    params = prov.get("parameters") or {}
    rows = [
        ["Generated (UTC)", prov.get("generated_utc") or "n/a"],
        ["Software", f"ProteinScope {prov.get('proteinscope')} (MIT License), Python {prov.get('python')}"],
        ["Libraries", libraries or "n/a"],
        ["Pocket model", (
            f"{model.get('kind')} {model.get('version', '')}, {model.get('training_entries', '?')} training "
            f"entries, sha256 {str(model.get('sha256', ''))[:16]}" if model.get("version") else str(model.get("kind"))
        )],
        ["Input structure", (
            f"{structure.get('source', 'local file')} {structure.get('identifier')}, "
            f"{structure.get('bytes', 0):,} bytes, sha256 {str(structure.get('sha256', ''))[:16]}"
            + (f", retrieved {structure['retrieved_utc']}" if structure.get("retrieved_utc") else "")
        )],
        ["Parameters", (
            f"grid {params.get('grid_spacing')} A, buried lines >= {params.get('psp_threshold')}, "
            f"min volume {params.get('min_volume')} A3, probes on {params.get('probe_pockets')} pockets, "
            f"online context {'on' if params.get('online_context') else 'off'} (all values in analysis.json)"
        )],
    ]
    return rows


def _source_rows(result: AnalysisResult) -> list[list[str]]:
    services = (result.provenance or {}).get("data_sources") or []
    rows = [["Service", "Licence", "Reads (cached)", "Data retrieved (UTC)"]]
    for service in services:
        span = service.get("oldest_data_utc") or "n/a"
        if service.get("newest_data_utc") and service["newest_data_utc"] != span:
            span = f"{span} to {service['newest_data_utc']}"
        if service.get("expired_cache_used"):
            span += f" ({service['expired_cache_used']} from an expired cache)"
        rows.append([service["service"], service["licence"],
                     f"{service['requests']} ({service['from_cache']})", span])
    return rows if len(rows) > 1 else []


def _attribution_lines(result: AnalysisResult) -> list[str]:
    """Credits and references the data providers ask for, for the services this report used."""
    used = [s["service"] for s in (result.provenance or {}).get("data_sources") or []]
    lines = []
    for name in used:
        source = SOURCE_BY_NAME[name]
        lines.append(f"{source.name} ({source.licence}, {source.licence_url}): {source.credit} "
                     f"Cite: {source.citation} Terms: {source.terms_url}")
    meta = result.metadata or {}
    if meta.get("citation_title"):
        doi = f", doi:{meta['citation_doi']}" if meta.get("citation_doi") else ""
        title = " ".join(str(meta["citation_title"]).split())  # RCSB titles carry line breaks.
        lines.append(f"Primary citation of {result.identifier}: {title} "
                     f"({meta.get('citation_year') or 'n.d.'}{doi}).")
    if "ChEMBL" in used:
        lines.append("The target context taken from ChEMBL (potencies, compounds, drugs) is shared under "
                     "CC BY-SA 3.0; anything built on that part must carry the same licence.")
    if any(s.get("expired_cache_used") for s in (result.provenance or {}).get("data_sources") or []):
        lines.append("Some external data came from an expired local cache because the service could not "
                     "be reached; it may be out of date.")
    return lines


def _md_table(rows: list[list[str]], header: bool = True) -> str:
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(str(cell).replace("|", "\\|") for cell in cells) + " |"

    if header:
        head, body = rows[0], rows[1:]
    else:
        head, body = ["Property", "Value"], rows
    return "\n".join([line(head), line(["---"] * len(head)), *[line(row) for row in body]])


def write_markdown(
    result: AnalysisResult,
    narrative: Narrative,
    figures: dict[str, Path],
    path: Path,
    viewer: Path | None,
    pymol: Path | None,
) -> Path:
    def figure(name: str, caption: str) -> str:
        if name not in figures:
            return ""
        relative = os.path.relpath(figures[name], path.parent)
        return f"![{caption}]({relative})\n\n*{caption}*\n\n"

    audit = result.audit
    parts = [f"# Structure analysis report: {result.identifier}\n"]
    parts.append(f"{audit.title or ''}\n\nGenerated {date.today().isoformat()} by ProteinScope {__version__}.\n")
    parts.append(f"## Summary\n\n*Written by: {narrative.source}*\n\n{narrative.executive_summary}\n")
    if narrative.recommendations:
        parts.append("### Recommendations\n\n" + "\n".join(f"- {r}" for r in narrative.recommendations) + "\n")
    parts.append("## Structure audit\n\n" + _md_table(_summary_rows(result), header=False) + "\n")
    if audit.warnings:
        parts.append("### Warnings\n\n" + "\n".join(f"- {w}" for w in audit.warnings) + "\n")
    parts.append(figure("overview", "Structure with the top-ranked pockets"))
    parts.append("## Surface properties\n")
    parts.append(figure("surface_hydropathy", "Surface colored by Kyte-Doolittle hydropathy"))
    parts.append(figure("surface_electrostatics", "Surface colored by estimated electrostatic potential"))
    parts.append(figure("residue_profile", "Per-residue burial, hydropathy and potential"))
    patches = result.surface.summary.patches[:5]
    if patches:
        rows = [["Patch", "Apolar area (A2)", "Residues"]]
        rows += [[str(p.rank), f"{p.apolar_area:.0f}", ", ".join(p.residues[:10])] for p in patches]
        parts.append("### Hydrophobic patches\n\n" + _md_table(rows) + "\n")

    parts.append("## Binding pockets\n")
    if result.pockets:
        parts.append(_md_table(_pocket_rows(result)) + "\n")
        parts.append(figure("pocket_ranking", "Druggability score by component"))
        if narrative.pocket_interpretation:
            parts.append(f"*Written by: {narrative.source}*\n\n{narrative.pocket_interpretation}\n")
        for pocket in result.pockets[:3]:
            parts.append(f"### Pocket {pocket.pocket_id}\n")
            parts.append(f"**Lining residues:** {', '.join(pocket.lining_residues)}\n")
            parts.append(figure(f"pocket_{pocket.pocket_id}", f"Cut-away view of pocket {pocket.pocket_id}"))
            parts.append(_md_table(_probe_rows(result, pocket.pocket_id)) + "\n")
        parts.append(figure("probe_map", "Best probe energies across pockets"))
    else:
        parts.append("No pockets passed the detection thresholds.\n")

    if result.ligand_profiles:
        parts.append("## Bound ligands\n\n" + _md_table(_ligand_rows(result)) + "\n")
        if narrative.ligand_interpretation:
            parts.append(f"*Written by: {narrative.source}*\n\n{narrative.ligand_interpretation}\n")
        for name, file_path in figures.items():
            if name.startswith("ligand_"):
                parts.append(figure(name, f"Contact map for {name.removeprefix('ligand_').replace('_', ':')}"))
        chemistry = _chemistry_rows(result)
        if chemistry:
            parts.append("### Ligand chemistry\n\n" + _md_table(chemistry) + "\n")
            for name in figures:
                if name.startswith("chem_"):
                    parts.append(figure(name, f"2D structure of {name.removeprefix('chem_')}"))

    protein_rows = _protein_rows(result)
    annotation_rows = _annotation_rows(result)
    if protein_rows or annotation_rows:
        parts.append("## Functional annotation (UniProt)\n")
        if protein_rows:
            parts.append(_md_table(protein_rows) + "\n")
        if annotation_rows:
            parts.append("### Annotated residues lining the pockets\n\n" + _md_table(annotation_rows) + "\n")

    target_rows = _target_rows(result)
    if target_rows:
        parts.append("## Target context (ChEMBL)\n\n" + _md_table(target_rows) + "\n")
        compounds = _top_compound_rows(result)
        if compounds:
            parts.append("### Most potent measured compounds\n\n" + _md_table(compounds) + "\n")
    if result.context.get("notes"):
        parts.append("### Context notes\n\n" + "\n".join(f"- {n}" for n in result.context["notes"]) + "\n")

    links = ["- Interactive HTML report: [report.html](report.html)"]
    if viewer:
        links.append(f"- Interactive 3D viewer: [{viewer.name}]({os.path.relpath(viewer, path.parent)})")
    if pymol:
        links.append(f"- PyMOL session script: [{pymol.name}]({os.path.relpath(pymol, path.parent)})")
    if links:
        parts.append("## 3D files\n\n" + "\n".join(links) + "\n")
    provenance_rows = _provenance_rows(result)
    if provenance_rows:
        parts.append("## Provenance\n\n" + _md_table(provenance_rows, header=False) + "\n")
        source_rows = _source_rows(result)
        if source_rows:
            parts.append("### Data sources\n\n" + _md_table(source_rows) + "\n")
        attribution = _attribution_lines(result)
        if attribution:
            parts.append("### Licences and attribution\n\n" + "\n".join(f"- {line}" for line in attribution) + "\n")
    parts.append("## Methods and limits\n\n" + "\n".join(f"- {note}" for note in METHOD_NOTES) + "\n")
    path.write_text("\n".join(part for part in parts if part), encoding="utf-8")
    return path


class _Pdf:
    """Small wrapper around platypus that holds the styles and the growing story."""

    def __init__(self) -> None:
        base = getSampleStyleSheet()
        ink = colors.HexColor("#0b0b0b")
        muted = colors.HexColor("#52514e")
        self.body = ParagraphStyle("B", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13.5, textColor=ink, alignment=TA_LEFT, spaceAfter=6)
        self.small = ParagraphStyle("S", parent=self.body, fontSize=8, leading=10.5, textColor=muted)
        self.cell = ParagraphStyle("C", parent=self.body, fontSize=8, leading=10, spaceAfter=0)
        self.head_cell = ParagraphStyle("HC", parent=self.cell, fontName="Helvetica-Bold")
        self.h1 = ParagraphStyle("H1", parent=base["Title"], fontName="Helvetica-Bold", fontSize=19, leading=23, alignment=TA_LEFT, spaceAfter=4, textColor=ink)
        self.h2 = ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16, spaceBefore=12, spaceAfter=5, textColor=ink)
        self.h3 = ParagraphStyle("H3", parent=base["Heading3"], fontName="Helvetica-Bold", fontSize=10.5, leading=13, spaceBefore=8, spaceAfter=3, textColor=ink)
        self.caption = ParagraphStyle("Cap", parent=self.small, spaceBefore=2, spaceAfter=9)
        self.story: list[Any] = []
        self.width = A4[0] - 4 * cm

    def text(self, value: str, style: ParagraphStyle | None = None) -> None:
        for block in value.split("\n\n"):
            if block.strip():
                self.story.append(Paragraph(escape(block.strip()).replace("\n", "<br/>"), style or self.body))

    def heading(self, value: str, style: ParagraphStyle) -> None:
        self.story.append(Paragraph(escape(value), style))

    def bullets(self, values: list[str]) -> None:
        self.story.append(ListFlowable(
            [ListItem(Paragraph(escape(v), self.body), leftIndent=12) for v in values],
            bulletType="bullet", start="•", leftIndent=12,
        ))

    def table(self, rows: list[list[str]], widths: list[float] | None = None, header: bool = True) -> None:
        cells = [
            [Paragraph(escape(str(c)), self.head_cell if header and r == 0 else self.cell) for c in row]
            for r, row in enumerate(rows)
        ]
        table = Table(cells, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
        style = [
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#dcdbd5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ]
        if header:
            style.append(("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#0b0b0b")))
        table.setStyle(TableStyle(style))
        self.story += [table, Spacer(1, 8)]

    def image(self, path: Path, caption: str, max_width: float | None = None, max_height: float = 15 * cm) -> None:
        width, height = ImageReader(str(path)).getSize()
        scale = min((max_width or self.width) / width, max_height / height)
        self.story.append(KeepTogether([
            Image(str(path), width=width * scale, height=height * scale, hAlign="LEFT"),
            Paragraph(escape(caption), self.caption),
        ]))


def write_pdf(result: AnalysisResult, narrative: Narrative, figures: dict[str, Path], path: Path) -> Path:
    pdf = _Pdf()
    audit = result.audit
    pdf.heading(f"Structure analysis report: {result.identifier}", pdf.h1)
    if audit.title:
        pdf.text(audit.title, pdf.small)
    pdf.text(f"Generated {date.today().isoformat()} by ProteinScope {__version__}.", pdf.small)

    pdf.heading("Summary", pdf.h2)
    pdf.text(f"Written by: {narrative.source}", pdf.small)
    pdf.text(narrative.executive_summary)
    if narrative.recommendations:
        pdf.heading("Recommendations", pdf.h3)
        pdf.bullets(narrative.recommendations)

    pdf.heading("Structure audit", pdf.h2)
    pdf.table(_summary_rows(result), widths=[6.5 * cm, pdf.width - 6.5 * cm], header=False)
    if audit.warnings:
        pdf.heading("Warnings", pdf.h3)
        pdf.bullets(audit.warnings)
    if "overview" in figures:
        pdf.image(figures["overview"], "Structure with the top-ranked pockets. Pockets behind the protein show through.", max_height=12.5 * cm)

    pdf.story.append(CondPageBreak(10 * cm))
    pdf.heading("Surface properties", pdf.h2)
    side = (pdf.width - 0.4 * cm) / 2
    pair = [name for name in ("surface_hydropathy", "surface_electrostatics") if name in figures]
    if len(pair) == 2:
        images = []
        for name in pair:
            width, height = ImageReader(str(figures[name])).getSize()
            images.append(Image(str(figures[name]), width=side, height=height * side / width))
        pdf.story.append(Table([images], colWidths=[side + 0.2 * cm] * 2, hAlign="LEFT", style=[
            ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))
        pdf.text("Same view of the surface colored by hydropathy (left) and estimated electrostatic potential (right).", pdf.caption)
    if "residue_profile" in figures:
        pdf.image(figures["residue_profile"], "Per-residue burial, hydropathy and surface potential.", max_height=10.5 * cm)
    patches = result.surface.summary.patches[:5]
    if patches:
        pdf.heading("Hydrophobic patches", pdf.h3)
        rows = [["Patch", "Apolar area (A2)", "Residues"]]
        rows += [[str(p.rank), f"{p.apolar_area:.0f}", ", ".join(p.residues[:10])] for p in patches]
        pdf.table(rows, widths=[1.5 * cm, 2.8 * cm, pdf.width - 4.3 * cm])

    pdf.story.append(CondPageBreak(10 * cm))
    pdf.heading("Binding pockets", pdf.h2)
    if result.pockets:
        pdf.table(_pocket_rows(result))
        if "pocket_ranking" in figures:
            pdf.image(figures["pocket_ranking"], "Druggability score, split by weighted component.", max_height=9 * cm)
        if narrative.pocket_interpretation:
            pdf.text(f"Written by: {narrative.source}", pdf.small)
            pdf.text(narrative.pocket_interpretation)
        for pocket in result.pockets[:3]:
            pdf.story.append(CondPageBreak(8 * cm))
            pdf.heading(f"Pocket {pocket.pocket_id}", pdf.h3)
            pdf.text("Lining residues: " + ", ".join(pocket.lining_residues), pdf.small)
            name = f"pocket_{pocket.pocket_id}"
            if name in figures:
                pdf.image(figures[name], f"Cut-away view of pocket {pocket.pocket_id}.", max_height=9.5 * cm)
            pdf.table(_probe_rows(result, pocket.pocket_id), widths=[3.8 * cm, 2.5 * cm, 2.4 * cm, pdf.width - 8.7 * cm])
        if "probe_map" in figures:
            pdf.image(figures["probe_map"], "Best probe energy per pocket from the rigid-grid simulation.", max_height=7 * cm)
    else:
        pdf.text("No pockets passed the detection thresholds.")

    if result.ligand_profiles:
        pdf.story.append(CondPageBreak(10 * cm))
        pdf.heading("Bound ligands", pdf.h2)
        pdf.table(_ligand_rows(result))
        if narrative.ligand_interpretation:
            pdf.text(f"Written by: {narrative.source}", pdf.small)
            pdf.text(narrative.ligand_interpretation)
        for name, file_path in figures.items():
            if name.startswith("ligand_"):
                pdf.image(file_path, f"Contact map for {name.removeprefix('ligand_').replace('_', ':')}.", max_height=14 * cm)
        chemistry = _chemistry_rows(result)
        if chemistry:
            pdf.heading("Ligand chemistry", pdf.h3)
            pdf.table(chemistry)
            for name, file_path in figures.items():
                if name.startswith("chem_"):
                    pdf.image(file_path, f"2D structure of {name.removeprefix('chem_')}.", max_width=8 * cm, max_height=6 * cm)

    protein_rows = _protein_rows(result)
    annotation_rows = _annotation_rows(result)
    if protein_rows or annotation_rows:
        pdf.story.append(CondPageBreak(6 * cm))
        pdf.heading("Functional annotation (UniProt)", pdf.h2)
        if protein_rows:
            pdf.table(protein_rows)
        if annotation_rows:
            pdf.heading("Annotated residues lining the pockets", pdf.h3)
            pdf.table(annotation_rows, widths=[1.4 * cm, 2.6 * cm, 1.8 * cm, pdf.width - 5.8 * cm])

    target_rows = _target_rows(result)
    if target_rows:
        pdf.story.append(CondPageBreak(6 * cm))
        pdf.heading("Target context (ChEMBL)", pdf.h2)
        pdf.table(target_rows)
        compounds = _top_compound_rows(result)
        if compounds:
            pdf.heading("Most potent measured compounds", pdf.h3)
            pdf.table(compounds)

    provenance_rows = _provenance_rows(result)
    if provenance_rows:
        pdf.story.append(CondPageBreak(7 * cm))
        pdf.heading("Provenance", pdf.h2)
        pdf.table(provenance_rows, widths=[3.6 * cm, pdf.width - 3.6 * cm], header=False)
        source_rows = _source_rows(result)
        if source_rows:
            pdf.heading("Data sources", pdf.h3)
            pdf.table(source_rows)
        attribution = _attribution_lines(result)
        if attribution:
            pdf.heading("Licences and attribution", pdf.h3)
            pdf.bullets(attribution)

    pdf.story.append(CondPageBreak(7 * cm))
    pdf.heading("Methods and limits", pdf.h2)
    pdf.bullets(METHOD_NOTES)

    def footer(canvas: Any, document: Any) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#52514e"))
        canvas.drawString(2 * cm, 1.1 * cm, f"ProteinScope report: {result.identifier}")
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {document.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title=f"Structure analysis report: {result.identifier}", author=f"ProteinScope {__version__}",
    )
    document.build(pdf.story, onFirstPage=footer, onLaterPages=footer)
    return path


def build_report(result: AnalysisResult, out_dir: Path | str, narrative: Narrative | None = None) -> ReportPaths:
    """Write figures, 3D files, JSON data, Markdown and PDF below out_dir/<identifier>."""
    directory = Path(out_dir) / result.identifier
    directory.mkdir(parents=True, exist_ok=True)
    narrative = narrative or rule_based_narrative(result)
    figures = generate_figures(result, directory)
    figures.update(_chemistry_figures(result, directory / "figures"))

    warnings: list[str] = []
    viewer = pymol = None
    try:
        viewer = write_html_viewer(result, directory / "viewer.html")
        pymol = write_pymol_session(result, directory)
    except PDBIOException as exc:  # The PDB format cannot hold the long chain IDs of very large entries.
        warnings.append(f"3D exports skipped: {exc}")

    data = directory / "analysis.json"
    data.write_text(json.dumps(analysis_digest(result), indent=2, default=str), encoding="utf-8")
    markdown = write_markdown(result, narrative, figures, directory / "report.md", viewer, pymol)
    pdf = write_pdf(result, narrative, figures, directory / "report.pdf")
    html_report = write_html_report(result, narrative, figures, directory / "report.html", viewer, pymol)
    return ReportPaths(directory, markdown, pdf, data, viewer, pymol, figures, warnings, html_report)


def _chemistry_figures(result: AnalysisResult, directory: Path) -> dict[str, Path]:
    """2D depictions of each ligand type, for the PDF and Markdown (the HTML inlines SVG)."""
    if not chemistry.available():
        return {}
    figures = {}
    for entry in (result.context.get("ligand_chemistry") or {}).values():
        name = f"chem_{entry['comp_id']}"
        if entry.get("smiles") and entry.get("properties") and name not in figures:
            try:
                figures[name] = chemistry.depiction_png(entry["smiles"], directory / f"{name}.png")
            except ValueError:
                continue
    return figures

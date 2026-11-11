"""Interactive HTML report: one page with the narrative, sortable tables, the 3D viewer,
ligand chemistry, target context and every figure.

The page is rendered here in Python, so it reads fine without JavaScript and prints cleanly.
Scripts only add sorting, expandable pocket rows, a figure lightbox, a theme switch and the
link from the pocket table to the embedded 3D viewer.
"""

from __future__ import annotations

import os
from datetime import date
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import __version__
from ..druggability import WEIGHTS
from ..enrich import CATEGORY_LABELS, annotation_summary

if TYPE_CHECKING:
    from ..pipeline import AnalysisResult
    from .narrative import Narrative

VIEWER_POCKETS = 8  # write_html_viewer draws this many pockets.
_COMPONENT_COLORS = {
    "size": "#2a78d6", "enclosure": "#1baf7a", "apolarity": "#eda100", "depth": "#6250d6", "charge": "#e87ba4",
}
_FIGURE_CAPTIONS = {
    "overview": "Structure with the top-ranked pockets",
    "surface_hydropathy": "Surface coloured by Kyte-Doolittle hydropathy",
    "surface_electrostatics": "Surface coloured by estimated electrostatic potential",
    "residue_profile": "Per-residue burial, hydropathy and potential",
    "pocket_ranking": "Druggability score by weighted component",
    "probe_map": "Best probe energies across pockets",
}


def _e(value: Any) -> str:
    return escape("" if value is None else str(value))


def _num(value: Any, digits: int = 2, unit: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}{unit}"
    return f"{value}{unit}"


def _rel(path: Path, base: Path) -> str:
    return escape(os.path.relpath(path, base).replace(os.sep, "/"))


def _figure_caption(name: str) -> str:
    if name in _FIGURE_CAPTIONS:
        return _FIGURE_CAPTIONS[name]
    if name.startswith("pocket_"):
        return f"Cut-away view of pocket {name.removeprefix('pocket_')}"
    if name.startswith("ligand_"):
        return f"Contact map for {name.removeprefix('ligand_').replace('_', ':')}"
    if name.startswith("chem_"):
        return f"2D structure of {name.removeprefix('chem_')}"
    return name.replace("_", " ")


def _tile(label: str, value: str, note: str = "", tone: str = "") -> str:
    tone_class = f" tone-{tone}" if tone else ""
    return (
        f'<div class="tile{tone_class}"><div class="tile-label">{_e(label)}</div>'
        f'<div class="tile-value">{_e(value)}</div>'
        + (f'<div class="tile-note">{_e(note)}</div>' if note else "") + "</div>"
    )


def _grade_tone(grade: str) -> str:
    return {"high": "good", "good": "good", "moderate": "warn", "low": "bad"}.get(grade, "")


def _class_tone(cls: str) -> str:
    return {"high": "good", "moderate": "warn", "low": "muted"}.get(cls, "muted")


def _tiles(result: AnalysisResult) -> str:
    audit = result.audit
    tiles = []
    if audit.is_predicted_model:
        tiles.append(_tile("Mean pLDDT", _num(audit.mean_plddt, 1), "AlphaFold model"))
    else:
        tiles.append(_tile("Resolution", _num(audit.resolution_A, 2, " Å"), audit.method or ""))
    tiles.append(_tile("Quality grade", audit.quality_grade, f"{len(audit.warnings)} audit warning(s)",
                       _grade_tone(audit.quality_grade)))
    tiles.append(_tile("Residues", str(audit.residue_count), f"{len(audit.chains)} chain(s)"))
    counts = {cls: sum(1 for p in result.pockets if p.druggability_class == cls) for cls in ("high", "moderate")}
    tiles.append(_tile("Pockets", str(len(result.pockets)),
                       f"{counts['high']} high · {counts['moderate']} moderate"))
    tiles.append(_tile("Ligands", str(len(result.ligand_profiles)), ", ".join(result.ligand_profiles)[:40]))
    targets = result.context.get("targets") or []
    if targets:
        activities = sum(row.get("measured_activities", 0) for row in targets)
        approved = sum(row.get("approved_drugs", 0) for row in targets)
        tiles.append(_tile("ChEMBL potencies", f"{activities:,}", f"{approved} approved drug(s)",
                           "good" if approved else ""))
    else:
        tiles.append(_tile("Net charge", f"{result.charge_model.net_charge:+.1f} e", "pH 7 estimate"))
    return '<div class="tiles">' + "".join(tiles) + "</div>"


def _narrative(narrative: Narrative) -> str:
    parts = [f'<p class="byline">Written by: {_e(narrative.source)}</p>']
    parts += [f"<p>{_e(block.strip())}</p>" for block in narrative.executive_summary.split("\n\n") if block.strip()]
    if narrative.recommendations:
        parts.append("<h3>Recommendations</h3><ul>" + "".join(f"<li>{_e(r)}</li>" for r in narrative.recommendations) + "</ul>")
    return "".join(parts)


def _score_bar(pocket: Any) -> str:
    segments = []
    for name, weight in WEIGHTS.items():
        value = pocket.score_components.get(name, 0.0) * weight
        if value > 0:
            segments.append(
                f'<span style="width:{value * 100:.1f}%;background:{_COMPONENT_COLORS[name]}" '
                f'title="{name}: {pocket.score_components.get(name, 0):.2f} × {weight:.2f}"></span>'
            )
    return f'<div class="bar" aria-hidden="true">{"".join(segments)}</div>'


def _probe_table(result: AnalysisResult, pocket_id: int) -> str:
    probes = result.probes.get(pocket_id)
    if not probes:
        return '<p class="muted">Probe simulation was not run for this pocket.</p>'
    rows = "".join(
        f"<tr><td>{_e(p.probe)}</td><td class='n'>{p.best_energy:.1f}</td>"
        f"<td class='n'>{p.favorable_fraction:.0%}</td><td>{_e(', '.join(p.hotspot_residues[:6]))}</td></tr>"
        for p in probes
    )
    return (
        "<table class='mini'><thead><tr><th>Probe</th><th class='n'>Best E (kcal/mol)</th>"
        "<th class='n'>Favourable</th><th>Residues at best position</th></tr></thead><tbody>"
        f"{rows}</tbody></table>"
    )


def _pockets(result: AnalysisResult, figures: dict[str, Path], base: Path, has_viewer: bool) -> str:
    if not result.pockets:
        return "<p>No pockets passed the detection thresholds.</p>"
    plddt = (result.context.get("plddt") or {}).get("pockets", {})
    annotations = ((result.context.get("annotations") or {}).get("pockets") or {})
    occupants: dict[int, list[str]] = {}
    for label, entries in result.ligand_pockets.items():
        for entry in entries:
            occupants.setdefault(entry["pocket_id"], []).append(label)
    legend = "".join(
        f'<span class="key"><i style="background:{_COMPONENT_COLORS[n]}"></i>{n} ({w:.2f})</span>'
        for n, w in WEIGHTS.items()
    )
    head = (
        "<thead><tr><th data-sort='n'>#</th><th data-sort='n' class='n'>Volume (Å³)</th>"
        "<th data-sort='n' class='n'>Enclosure</th><th data-sort='n' class='n'>Depth (Å)</th>"
        "<th data-sort='n' class='n'>Apolar</th><th data-sort='n' class='n'>Potential</th>"
        + ("<th data-sort='n' class='n'>pLDDT</th>" if plddt else "")
        + "<th data-sort='n' class='score-col'>Score</th><th data-sort='t'>Site</th>"
        + ("<th data-sort='t'>Function</th>" if annotations else "")
        + "<th>Ligand</th><th></th></tr></thead>"
    )
    rows = []
    columns = 11 + (1 if plddt else 0) + (1 if annotations else 0)
    for pocket in result.pockets:
        pid = pocket.pocket_id
        tags = "".join(f'<span class="chip">{_e(t)}</span>' for t in pocket.site_tags) or '<span class="muted">-</span>'
        ligands = ", ".join(occupants.get(pid, [])) or "-"
        confidence = (plddt.get(pid) or {}).get("mean_plddt")
        show = (
            f'<button class="ghost" data-focus="{pid}" title="Show pocket {pid} in the 3D viewer">3D</button>'
            if has_viewer and pid <= VIEWER_POCKETS else ""
        )
        rows.append(
            f'<tr class="pocket-row" data-pocket="{pid}" tabindex="0" aria-expanded="false">'
            f"<td data-v='{pid}'><b>{pid}</b></td>"
            f"<td class='n' data-v='{pocket.volume:.1f}'>{pocket.volume:.0f}</td>"
            f"<td class='n' data-v='{pocket.enclosure:.3f}'>{pocket.enclosure:.2f}</td>"
            f"<td class='n' data-v='{pocket.depth:.2f}'>{pocket.depth:.1f}</td>"
            f"<td class='n' data-v='{pocket.apolar_fraction:.3f}'>{pocket.apolar_fraction:.2f}</td>"
            f"<td class='n' data-v='{pocket.mean_potential:.2f}'>{pocket.mean_potential:+.1f}</td>"
            + (f"<td class='n' data-v='{confidence or 0}'>{_num(confidence, 0)}</td>" if plddt else "")
            + f"<td data-v='{pocket.druggability:.3f}' class='score-col'><div class='score'>"
            f"<span class='badge tone-{_class_tone(pocket.druggability_class)}'>{pocket.druggability:.2f} "
            f"{_e(pocket.druggability_class)}</span>{_score_bar(pocket)}</div></td>"
            f"<td class='tags' data-v='{_e(' '.join(pocket.site_tags))}'>{tags}</td>"
            + (f"<td class='fn' data-v='{_e(annotation_summary(annotations.get(pid)))}'>"
               f"{_function_cell(annotations.get(pid))}</td>" if annotations else "")
            + f"<td>{_e(ligands)}</td><td>{show}</td></tr>"
        )
        figure = figures.get(f"pocket_{pid}")
        detail = [
            f"<p><b>Lining residues ({len(pocket.lining_residues)}):</b> {_e(', '.join(pocket.lining_residues))}</p>",
            f"<p class='muted'>Charged residues {pocket.charged_residues}, net charge {pocket.net_charge:+.1f}, "
            f"polar atoms {pocket.donor_acceptor_atoms}, mean hydropathy {pocket.mean_hydropathy:+.2f}. "
            f"Hand-weighted heuristic {pocket.heuristic_score:.2f}; headline score from {_e(pocket.score_model)}.</p>",
            _annotation_detail(annotations.get(pid)),
            _probe_table(result, pid),
        ]
        if figure:
            detail.insert(0, f'<a class="thumb" href="{_rel(figure, base)}" data-lightbox><img loading="lazy" '
                             f'src="{_rel(figure, base)}" alt="{_e(_figure_caption(figure.stem))}"></a>')
        rows.append(f'<tr class="detail" data-detail="{pid}" hidden><td colspan="{columns}"><div class="detail-grid">'
                    + "".join(detail) + "</div></td></tr>")
    model = result.pockets[0].score_model
    explanation = (
        f"The score is a calibrated probability from the {_e(model)}; the bar breaks down the "
        f"hand-weighted heuristic for comparison."
        if model.startswith("fitted") else
        "The score is the hand-weighted heuristic, broken down by the bar."
    )
    return (
        f'<p class="muted">Click a row for lining residues, annotations and probe energies. '
        f'Column headers sort. {explanation} Segments: {legend}</p>'
        f'<div class="table-wrap"><table id="pocket-table" class="data">{head}<tbody>{"".join(rows)}</tbody></table></div>'
    )


def _properties(props: dict[str, Any]) -> str:
    items = [
        ("MW", f"{props['molecular_weight']:.0f}"), ("cLogP", f"{props['clogp']:.2f}"),
        ("HBD / HBA", f"{props['hbd']} / {props['hba']}"), ("TPSA", f"{props['tpsa']:.0f} Å²"),
        ("Rot. bonds", str(props["rotatable_bonds"])), ("Heavy atoms", str(props["heavy_atoms"])),
        ("Arom. rings", str(props["aromatic_rings"])), ("Fsp³", f"{props['fraction_csp3']:.2f}"),
        ("Formal charge", f"{props['formal_charge']:+d}"), ("QED", f"{props['qed']:.2f}"),
    ]
    grid = "".join(f"<div><dt>{_e(k)}</dt><dd>{_e(v)}</dd></div>" for k, v in items)
    badges = []
    lipinski = props["lipinski_violations"]
    badges.append(f'<span class="badge tone-{"good" if not lipinski else "warn"}">Rule of five: '
                  f'{"pass" if not lipinski else _e(", ".join(lipinski))}</span>')
    veber = props["veber_violations"]
    badges.append(f'<span class="badge tone-{"good" if not veber else "warn"}">Veber: '
                  f'{"pass" if not veber else _e(", ".join(veber))}</span>')
    if props["rule_of_three"]:
        badges.append('<span class="badge tone-good">Fragment-like (rule of three)</span>')
    return f'<dl class="props">{grid}</dl><div class="badges">{"".join(badges)}</div>'


def _ligands(result: AnalysisResult, figures: dict[str, Path], base: Path) -> str:
    if not result.ligand_profiles:
        return "<p>No drug-like ligand is bound in this structure.</p>"
    chemistry = result.context.get("ligand_chemistry") or {}
    cards = []
    for label, profile in result.ligand_profiles.items():
        entry = chemistry.get(label, {})
        hits = result.ligand_pockets.get(label, [])
        counts = profile.counts()
        polar = [c for c in profile.contacts if c.kind != "hydrophobic"]
        hydrophobic = sorted({c.residue for c in profile.contacts if c.kind == "hydrophobic"})
        contact_rows = "".join(
            f"<tr><td>{_e(c.kind)}</td><td>{_e(c.ligand_atom)}</td><td>{_e(c.residue)} {_e(c.residue_atom)}</td>"
            f"<td class='n'>{c.distance:.2f}</td></tr>"
            for c in sorted(polar, key=lambda c: c.distance)
        )
        strongest = sorted(profile.residue_vdw.items(), key=lambda item: item[1])[:8]
        vdw = "".join(f"<li>{_e(name)} <span class='muted'>{value:.2f}</span></li>" for name, value in strongest)
        safe = label.replace(":", "_")
        contact_map = figures.get(f"ligand_{safe}")
        depiction = entry.get("svg") or ""
        name = entry.get("name")
        smiles = entry.get("smiles")
        header = (
            f"<h3>{_e(label)}</h3>"
            + (f"<p class='muted'>{_e(name)}</p>" if name else "")
            + f"<p>{profile.atom_count} heavy atoms, {profile.buried_fraction:.0%} buried, "
            + (f"in pocket {hits[0]['pocket_id']}" if hits else "outside the ranked pockets")
            + ". " + _e(", ".join(f"{v} {k}" for k, v in sorted(counts.items()))) + "</p>"
        )
        chem = ""
        if entry.get("properties"):
            chem = _properties(entry["properties"])
        elif entry:
            chem = "<p class='muted'>No descriptors (RDKit missing or SMILES unavailable).</p>"
        cards.append(
            '<article class="ligand">'
            f'<div class="ligand-head"><div>{header}{chem}'
            + (f"<p class='smiles'><code>{_e(smiles)}</code></p>" if smiles else "")
            + "</div>"
            + (f'<figure class="depiction">{depiction}<figcaption>2D structure (CCD)</figcaption></figure>' if depiction else "")
            + "</div>"
            '<div class="ligand-body">'
            + (
                "<div><h4>Polar contacts</h4><table class='mini'><thead><tr><th>Type</th><th>Ligand atom</th>"
                f"<th>Protein atom</th><th class='n'>Å</th></tr></thead><tbody>{contact_rows}</tbody></table></div>"
                if contact_rows else "<div><h4>Polar contacts</h4><p class='muted'>None found.</p></div>"
            )
            + "<div><h4>Hydrophobic contacts</h4><p>" + (_e(", ".join(hydrophobic)) or "none") + "</p>"
            + (f"<h4>Strongest van der Waals residues (kcal/mol)</h4><ul class='cols'>{vdw}</ul>" if vdw else "")
            + "</div>"
            + (f'<a class="thumb" href="{_rel(contact_map, base)}" data-lightbox><img loading="lazy" '
               f'src="{_rel(contact_map, base)}" alt="Contact map for {_e(label)}"></a>' if contact_map else "")
            + "</div></article>"
        )
    return "".join(cards)


_CATEGORY_TONE = {
    "catalytic": "warn", "binding": "good", "functional": "good",
    "variant": "bad", "modification": "muted", "disulfide": "muted", "mutagenesis": "muted",
}


def _function_cell(row: dict[str, Any] | None, limit: int = 3) -> str:
    """The Function column: the most telling annotations only; the row detail has them all."""
    if not row:
        return '<span class="muted">-</span>'
    chips = [("good", f"binds {name}") for name in row.get("cofactors") or []]
    for category, label in CATEGORY_LABELS.items():
        count = row["categories"].get(category)
        if count:
            chips.append((_CATEGORY_TONE.get(category, "muted"), label if count == 1 else f"{label} ×{count}"))
    if row.get("variants"):
        chips.append(("bad", f"{row['variants']} variants"))
    if not chips:
        return '<span class="muted">-</span>'
    shown = "".join(f'<span class="chip tone-{tone}">{_e(text)}</span>' for tone, text in chips[:limit])
    rest = len(chips) - limit
    return shown + (f'<span class="chip" title="Open the row for all annotations">+{rest}</span>' if rest > 0 else "")


def _annotation_detail(row: dict[str, Any] | None, limit: int = 14) -> str:
    """The annotated residues of one pocket, functional features before variants."""
    if not row or not row.get("residues"):
        return ""
    order = list(CATEGORY_LABELS)
    residues = sorted(row["residues"], key=lambda r: min(
        order.index(f["category"]) if f["category"] in order else 9 for f in r["features"]
    ))
    rows = "".join(
        f"<tr><td>{_e(residue['residue'])}</td><td class='n'>{residue['uniprot_position']}</td><td>"
        + "".join(
            f'<span class="chip tone-{_CATEGORY_TONE.get(f["category"], "muted")}">{_e(f["type"])}</span> '
            f'<span class="muted">{_e(f["description"] or "")}</span><br>' for f in residue["features"][:4]
        )
        + "</td></tr>"
        for residue in residues[:limit]
    )
    more = (f"<p class='muted'>+{len(residues) - limit} further annotated residues.</p>"
            if len(residues) > limit else "")
    regions = (f"<p class='muted'>Within: {_e(', '.join(row['regions']))}.</p>" if row.get("regions") else "")
    return (
        "<div><h4>Functional annotation (UniProt)</h4>" + regions
        + "<table class='mini'><thead><tr><th>Residue</th><th class='n'>UniProt</th>"
        f"<th>Feature</th></tr></thead><tbody>{rows}</tbody></table>{more}</div>"
    )


def _annotations(result: AnalysisResult) -> str:
    """Protein-level card: identity of the mapped chains, domains and Pfam families."""
    proteins = ((result.context.get("annotations") or {}).get("proteins") or [])
    if not proteins:
        return ""
    cards = []
    for protein in proteins:
        accession = protein.get("accession") or "?"
        identity = ", ".join(f"{chain} {value:.0%}" for chain, value in (protein.get("chain_identity") or {}).items())
        pfam = " ".join(
            f"<a class='chip' href='https://www.ebi.ac.uk/interpro/entry/pfam/{_e(p['id'])}/' rel='noopener'>"
            f"{_e(p['id'])} {_e(p['name'] or '')}</a>" for p in (protein.get("pfam") or [])
        )
        domains = " ".join(
            f"<span class='chip'>{_e(region['description'] or region['type'])} "
            f"{region['start']}-{region['end']}</span>"
            for region in (protein.get("regions") or [])[:8]
        )
        cards.append(
            f"<article class='card'><h3>"
            f"<a href='https://www.uniprot.org/uniprotkb/{_e(accession)}' rel='noopener'>{_e(accession)}</a> "
            f"<span class='muted'>{_e(protein.get('protein_name') or '')}</span></h3>"
            f"<p>{_e(', '.join(protein.get('genes') or []) or 'no gene name')} · "
            f"reference sequence {_num(protein.get('length'), 0)} residues · "
            f"<b>{protein.get('mapped_residues', 0)}</b> observed residues mapped by alignment "
            f"(identity {_e(identity) or 'n/a'}) · "
            f"<b>{protein.get('annotated_positions', 0)}</b> annotated positions in the reference.</p>"
            + (f"<p>{domains}</p>" if domains else "")
            + (f"<p>{pfam}</p>" if pfam else "")
            + "</article>"
        )
    return "".join(cards)


def _targets(result: AnalysisResult) -> str:
    rows = result.context.get("targets")
    if not rows:
        return ""
    blocks = []
    for row in rows:
        chembl = row.get("chembl")
        accession = row["accession"]
        title = (
            f"<h3><a href='https://www.uniprot.org/uniprotkb/{_e(accession)}' rel='noopener'>{_e(accession)}</a> "
            f"<span class='muted'>{_e(row.get('description') or '')}</span></h3>"
        )
        if not chembl:
            blocks.append(f"<article class='card'>{title}<p class='muted'>ChEMBL has no target for this accession.</p></article>")
            continue
        target_link = f"https://www.ebi.ac.uk/chembl/explore/target/{_e(chembl['target_chembl_id'])}"
        compounds = "".join(
            f"<tr><td><a href='https://www.ebi.ac.uk/chembl/explore/compound/{_e(c['molecule_chembl_id'])}' rel='noopener'>"
            f"{_e(c['molecule_chembl_id'])}</a></td><td>{_e(c.get('name') or '-')}</td>"
            f"<td class='n'>{_num(c.get('pchembl'), 1)}</td><td>{_e(c.get('type') or '-')}</td></tr>"
            for c in row.get("top_compounds") or []
        )
        drugs = "".join(
            f"<li><b>{_e(d.get('name') or d['molecule_chembl_id'])}</b> "
            f"<span class='badge tone-{'good' if (d.get('max_phase') or 0) >= 4 else 'muted'}'>phase {_num(d.get('max_phase'), 0)}</span> "
            f"<span class='muted'>{_e(d.get('mechanism') or '')}</span></li>"
            for d in (row.get("drugs") or [])[:12]
        )
        blocks.append(
            f"<article class='card'>{title}"
            f"<p>ChEMBL target <a href='{target_link}' rel='noopener'>{_e(chembl['target_chembl_id'])}</a> "
            f"({_e(chembl.get('pref_name') or '')}, {_e(chembl.get('target_type') or '')}): "
            f"<b>{row.get('measured_activities', 0):,}</b> measured potencies, "
            f"<b>{len(row.get('drugs') or [])}</b> compounds with a known mechanism, "
            f"<b>{row.get('approved_drugs', 0)}</b> approved.</p>"
            + (
                "<h4>Most potent measured compounds</h4><div class='table-wrap'><table class='mini'><thead><tr>"
                "<th>ChEMBL ID</th><th>Name</th><th class='n'>pChEMBL</th><th>Assay</th></tr></thead>"
                f"<tbody>{compounds}</tbody></table></div>" if compounds else ""
            )
            + (f"<h4>Compounds with a known mechanism</h4><ul class='drugs'>{drugs}</ul>" if drugs else "")
            + "</article>"
        )
    return "".join(blocks)


def _audit(result: AnalysisResult) -> str:
    from .builder import _summary_rows

    rows = "".join(f"<tr><th scope='row'>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in _summary_rows(result))
    warnings = "".join(f"<li>{_e(w)}</li>" for w in result.audit.warnings)
    patches = result.surface.summary.patches[:5]
    patch_rows = "".join(
        f"<tr><td>{p.rank}</td><td class='n'>{p.apolar_area:.0f}</td><td>{_e(', '.join(p.residues[:12]))}</td></tr>"
        for p in patches
    )
    return (
        "<div class='two'><div><table class='kv'>" + rows + "</table></div><div>"
        + (f"<div class='callout'><h4>Audit warnings</h4><ul>{warnings}</ul></div>" if warnings else
           "<div class='callout ok'><h4>No audit warnings</h4></div>")
        + (
            "<h4>Hydrophobic surface patches</h4><table class='mini'><thead><tr><th>#</th>"
            f"<th class='n'>Apolar Å²</th><th>Residues</th></tr></thead><tbody>{patch_rows}</tbody></table>"
            if patch_rows else ""
        )
        + "</div></div>"
    )


def _gallery(figures: dict[str, Path], base: Path) -> str:
    items = "".join(
        f'<figure><a href="{_rel(path, base)}" data-lightbox><img loading="lazy" src="{_rel(path, base)}" '
        f'alt="{_e(_figure_caption(name))}"></a><figcaption>{_e(_figure_caption(name))}</figcaption></figure>'
        for name, path in figures.items() if path.is_file()
    )
    return f'<div class="gallery">{items}</div>'


CSS = """
:root{--bg:#fcfcfb;--panel:#f3f2ee;--card:#ffffff;--ink:#0b0b0b;--muted:#5d5c58;--line:#dcdbd5;--accent:#2a78d6;
--good:#137a52;--good-bg:#dff3ea;--warn:#8a5a00;--warn-bg:#fbefd0;--bad:#a8231f;--bad-bg:#f9dedc;--chip:#ecebe6;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--panel:#21211f;--card:#1c1c1b;--ink:#f0efec;
--muted:#b4b3ab;--line:#353532;--accent:#5a9ff0;--good:#6fd3a5;--good-bg:#173829;--warn:#f2c46b;--warn-bg:#3a2e12;
--bad:#f19c98;--bad-bg:#3e1d1b;--chip:#2c2c2a;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#161615;--panel:#21211f;--card:#1c1c1b;--ink:#f0efec;--muted:#b4b3ab;--line:#353532;
--accent:#5a9ff0;--good:#6fd3a5;--good-bg:#173829;--warn:#f2c46b;--warn-bg:#3a2e12;--bad:#f19c98;--bad-bg:#3e1d1b;--chip:#2c2c2a;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
a{color:var(--accent)}
header.top{padding:28px 16px 18px;border-bottom:1px solid var(--line);background:var(--panel)}
.wrap{max-width:1180px;margin:0 auto;padding:0 16px}
header.top .wrap{padding:0}
h1{margin:0;font-size:28px;letter-spacing:-.01em}
h2{font-size:20px;margin:0 0 12px}
h3{font-size:16px;margin:0 0 6px}
h4{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:14px 0 6px}
.subtitle{color:var(--muted);margin:4px 0 12px;max-width:70ch}
.meta{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.meta a,.meta button{font:inherit;font-size:13px;padding:5px 10px;border:1px solid var(--line);border-radius:6px;background:var(--card);
color:var(--ink);text-decoration:none;cursor:pointer}
.meta .byline{font-size:13px;color:var(--muted)}
nav.sections{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--bg) 92%,transparent);backdrop-filter:blur(6px);
border-bottom:1px solid var(--line)}
nav.sections .wrap{display:flex;gap:4px;overflow-x:auto;padding-top:6px;padding-bottom:6px}
nav.sections a{white-space:nowrap;text-decoration:none;color:var(--muted);font-size:14px;padding:6px 10px;border-radius:6px}
nav.sections a:hover,nav.sections a.active{background:var(--panel);color:var(--ink)}
section{padding:28px 0;border-bottom:1px solid var(--line);scroll-margin-top:52px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:20px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile-label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.tile-value{font-size:24px;font-weight:650;margin-top:2px;font-variant-numeric:tabular-nums}
.tile-note{font-size:12.5px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tile.tone-good .tile-value{color:var(--good)}.tile.tone-warn .tile-value{color:var(--warn)}.tile.tone-bad .tile-value{color:var(--bad)}
.summary{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;max-width:80ch}
.summary p{margin:0 0 10px}
.byline{font-size:13px;color:var(--muted)}
.muted{color:var(--muted)}
.viewer{position:relative;height:min(70vh,620px);border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--panel)}
.viewer iframe{width:100%;height:100%;border:0;display:block}
.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
thead th{font-size:12px;text-transform:uppercase;letter-spacing:.03em;color:var(--muted);background:var(--panel);white-space:nowrap}
th[data-sort]{cursor:pointer;user-select:none}th[data-sort]:after{content:" ↕";opacity:.4}
th[data-dir="asc"]:after{content:" ↑";opacity:1}th[data-dir="desc"]:after{content:" ↓";opacity:1}
.n{text-align:right;font-variant-numeric:tabular-nums}
tr.pocket-row{cursor:pointer}tr.pocket-row:hover{background:var(--panel)}
tr.pocket-row[aria-expanded="true"]{background:var(--panel)}
tr.detail td{background:var(--bg)}
.detail-grid{display:grid;grid-template-columns:minmax(200px,320px) 1fr;gap:6px 18px;
position:sticky;left:10px;max-width:min(1120px,calc(100vw - 54px))}
.detail-grid p{overflow-wrap:anywhere}
.detail-grid>:not(.thumb){grid-column:2}.detail-grid .thumb{grid-row:1/span 4}
.score{display:flex;flex-direction:column;gap:4px;min-width:130px}
.bar{display:flex;height:6px;border-radius:3px;background:var(--chip);overflow:hidden;width:130px}
.bar span{display:block;height:100%}
.key{display:inline-flex;align-items:center;gap:4px;margin-right:10px;white-space:nowrap}
.key i{width:10px;height:10px;border-radius:2px;display:inline-block}
.badge{display:inline-block;font-size:12px;font-weight:600;padding:2px 7px;border-radius:999px;white-space:nowrap}
.tone-good{background:var(--good-bg);color:var(--good)}.tone-warn{background:var(--warn-bg);color:var(--warn)}
.tone-bad{background:var(--bad-bg);color:var(--bad)}.tone-muted{background:var(--chip);color:var(--muted)}
#pocket-table td.fn{min-width:168px}#pocket-table td.tags{min-width:132px}
.provenance th{width:150px;white-space:nowrap;text-transform:none;letter-spacing:0;font-size:13px}
.provenance td{overflow-wrap:anywhere}
.chip{display:inline-block;font-size:12px;background:var(--chip);border-radius:6px;padding:1px 7px;margin:1px 3px 1px 0}
a.chip{text-decoration:none}a.chip:hover{text-decoration:underline}
.chip.tone-good{background:var(--good-bg);color:var(--good)}
.chip.tone-warn{background:var(--warn-bg);color:var(--warn)}
.chip.tone-bad{background:var(--bad-bg);color:var(--bad)}
button.ghost{font:inherit;font-size:12.5px;padding:3px 9px;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:6px;cursor:pointer}
button.ghost:hover{border-color:var(--accent);color:var(--accent)}
table.mini{font-size:13px;border:1px solid var(--line);border-radius:8px;overflow:hidden}
table.mini th,table.mini td{padding:5px 8px}
table.kv th{width:46%;font-weight:500;color:var(--muted);background:transparent;text-transform:none;letter-spacing:0;font-size:14px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:24px}
.callout{border:1px solid var(--warn);background:var(--warn-bg);border-radius:10px;padding:4px 16px 8px}
.callout.ok{border-color:var(--good);background:var(--good-bg)}
.callout ul{margin:4px 0;padding-left:18px}
.ligand,.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-bottom:16px}
.ligand-head{display:grid;grid-template-columns:1fr auto;gap:18px}
.depiction{margin:0;background:#fff;border-radius:8px;padding:6px;border:1px solid var(--line);width:320px}
.depiction svg{display:block;width:100%;height:auto}
.depiction figcaption{font-size:12px;color:#5d5c58;text-align:center}
.ligand-body{display:grid;grid-template-columns:1.3fr 1fr minmax(180px,260px);gap:18px;margin-top:8px}
dl.props{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:6px 12px;margin:10px 0}
dl.props div{background:var(--panel);border-radius:6px;padding:6px 8px}
dl.props dt{font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.03em}
dl.props dd{margin:0;font-weight:600;font-variant-numeric:tabular-nums}
.badges{display:flex;flex-wrap:wrap;gap:6px}
.smiles code{font-size:12px;word-break:break-all;color:var(--muted)}
ul.cols{columns:2;margin:0;padding-left:18px;font-size:13.5px}
ul.drugs{margin:0;padding-left:18px}ul.drugs li{margin:3px 0}
.thumb img,.gallery img{width:100%;height:auto;display:block;border-radius:8px;border:1px solid var(--line);background:#fcfcfb}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}
.gallery figure{margin:0}.gallery figcaption{font-size:13px;color:var(--muted);margin-top:4px}
dialog{border:0;padding:0;background:transparent;max-width:96vw;max-height:94vh}
dialog::backdrop{background:rgba(0,0,0,.75)}
dialog img{max-width:96vw;max-height:88vh;display:block;border-radius:8px;background:#fcfcfb}
dialog p{color:#fff;text-align:center;margin:8px 0 0}
ul.methods li{margin-bottom:6px;max-width:90ch}
footer{padding:24px 16px 40px;color:var(--muted);font-size:13px}
@media (max-width:860px){.two,.ligand-body,.ligand-head,.detail-grid{grid-template-columns:1fr}
.detail-grid>:not(.thumb){grid-column:auto}.detail-grid .thumb{grid-row:auto}.depiction{width:100%;max-width:340px}h1{font-size:23px}}
@media print{nav.sections,.meta button,#viewer-section,button.ghost{display:none!important}
tr.detail{display:table-row!important}section{break-inside:avoid-page;border:0}body{font-size:12px}}
"""

# Shared with the overview page in `proteinscope.web`, so both remember the same choice.
THEME_JS = """
(function(){
  const root=document.documentElement;
  const themes=['auto','light','dark'];
  const button=document.getElementById('theme');
  function apply(t){if(t==='auto')root.removeAttribute('data-theme');else root.setAttribute('data-theme',t);
    if(button)button.textContent='Theme: '+t;}
  let theme='auto';try{theme=localStorage.getItem('ps-theme')||'auto';}catch(e){}
  apply(theme);
  if(button)button.addEventListener('click',()=>{theme=themes[(themes.indexOf(theme)+1)%3];apply(theme);
    try{localStorage.setItem('ps-theme',theme);}catch(e){}});
})();
"""

JS = THEME_JS + """
(function(){

  document.querySelectorAll('tr.pocket-row').forEach(row=>{
    const toggle=()=>{const d=document.querySelector('tr[data-detail="'+row.dataset.pocket+'"]');
      const open=d.hidden;d.hidden=!open;row.setAttribute('aria-expanded',open);};
    row.addEventListener('click',e=>{if(e.target.closest('button'))return;toggle();});
    row.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();toggle();}});
  });

  const table=document.getElementById('pocket-table');
  if(table){table.querySelectorAll('th[data-sort]').forEach((th,index)=>{th.addEventListener('click',()=>{
    const dir=th.dataset.dir==='asc'?'desc':'asc';table.querySelectorAll('th').forEach(o=>delete o.dataset.dir);th.dataset.dir=dir;
    const body=table.tBodies[0];const pairs=[...body.querySelectorAll('tr.pocket-row')].map(r=>[r,body.querySelector('tr[data-detail="'+r.dataset.pocket+'"]')]);
    const value=r=>r.children[index].dataset.v;const numeric=th.dataset.sort==='n';
    pairs.sort((a,b)=>{const x=value(a[0]),y=value(b[0]);const c=numeric?parseFloat(x)-parseFloat(y):String(x).localeCompare(String(y));return dir==='asc'?c:-c;});
    pairs.forEach(([r,d])=>{body.append(r);if(d)body.append(d);});});});}

  const frame=document.getElementById('viewer-frame');
  document.querySelectorAll('[data-focus]').forEach(b=>b.addEventListener('click',()=>{
    if(!frame)return;frame.contentWindow.postMessage({type:'focusPocket',id:Number(b.dataset.focus)},'*');
    document.getElementById('viewer-section').scrollIntoView({behavior:'smooth'});}));

  const dialog=document.getElementById('lightbox');
  document.querySelectorAll('[data-lightbox]').forEach(a=>a.addEventListener('click',e=>{
    if(!dialog||!dialog.showModal)return;e.preventDefault();const img=a.querySelector('img');
    dialog.querySelector('img').src=a.getAttribute('href');dialog.querySelector('img').alt=img?img.alt:'';
    dialog.querySelector('p').textContent=img?img.alt:'';dialog.showModal();}));
  if(dialog)dialog.addEventListener('click',()=>dialog.close());

  const links=[...document.querySelectorAll('nav.sections a')];
  const observer=new IntersectionObserver(entries=>entries.forEach(en=>{if(en.isIntersecting){
    links.forEach(l=>l.classList.toggle('active',l.getAttribute('href')==='#'+en.target.id));}}),{rootMargin:'-40% 0px -55% 0px'});
  document.querySelectorAll('main section[id]').forEach(s=>observer.observe(s));
})();
"""


def _provenance(result: AnalysisResult) -> str:
    from .builder import _attribution_lines, _provenance_rows, _source_rows

    rows = _provenance_rows(result)
    if not rows:
        return ""
    facts = "".join(f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in rows)
    sources = _source_rows(result)
    source_html = ""
    if sources:
        head = "".join(f"<th>{_e(c)}</th>" for c in sources[0])
        body = "".join("<tr>" + "".join(f"<td>{_e(c)}</td>" for c in row) + "</tr>" for row in sources[1:])
        source_html = (f"<h4>Data sources</h4><div class='table-wrap'><table class='mini'><thead><tr>{head}"
                       f"</tr></thead><tbody>{body}</tbody></table></div>")
    credits = "".join(f"<li>{_e(line)}</li>" for line in _attribution_lines(result))
    return (
        "<p class='muted'>Everything needed to trace this report: the software and model that computed it, "
        "a checksum of the input, and when each piece of external data was retrieved. Full detail is in "
        "analysis.json.</p>"
        f"<div class='table-wrap'><table class='mini provenance'><tbody>{facts}</tbody></table></div>"
        + source_html
        + (f"<h4>Licences and attribution</h4><ul class='methods'>{credits}</ul>" if credits else "")
    )


def write_html_report(
    result: AnalysisResult,
    narrative: Narrative,
    figures: dict[str, Path],
    path: Path,
    viewer: Path | None,
    pymol: Path | None,
) -> Path:
    from .builder import METHOD_NOTES

    base = path.parent
    audit = result.audit
    sections: list[tuple[str, str, str]] = [
        ("summary", "Summary", _tiles(result) + f'<div class="summary">{_narrative(narrative)}</div>'),
    ]
    if viewer:
        sections.append(("viewer-section", "3D viewer", (
            f'<div class="viewer"><iframe id="viewer-frame" src="{_rel(viewer, base)}" title="3D viewer of '
            f'{_e(result.identifier)}" loading="lazy"></iframe></div>'
            f'<p class="muted">Needs internet access for 3Dmol.js. '
            f'<a href="{_rel(viewer, base)}" target="_blank" rel="noopener">Open full screen</a>.</p>'
        )))
    pocket_html = _pockets(result, figures, base, viewer is not None)
    if narrative.pocket_interpretation:
        pocket_html = (f'<div class="summary"><p class="byline">Written by: {_e(narrative.source)}</p>'
                       f'<p>{_e(narrative.pocket_interpretation)}</p></div><br>') + pocket_html
    sections.append(("pockets", f"Pockets ({len(result.pockets)})", pocket_html))
    ligand_html = _ligands(result, figures, base)
    if narrative.ligand_interpretation:
        ligand_html = (f'<div class="summary"><p class="byline">Written by: {_e(narrative.source)}</p>'
                       f'<p>{_e(narrative.ligand_interpretation)}</p></div><br>') + ligand_html
    sections.append(("ligands", f"Ligands ({len(result.ligand_profiles)})", ligand_html))
    annotation_html = _annotations(result)
    if annotation_html:
        sections.append(("function", "Function", annotation_html))
    target_html = _targets(result)
    if target_html:
        sections.append(("target", "Target context", target_html))
    sections.append(("audit", "Structure audit", _audit(result)))
    sections.append(("figures", "Figures", _gallery(figures, base)))
    notes = result.context.get("notes") or []
    provenance_html = _provenance(result)
    if provenance_html:
        sections.append(("provenance", "Provenance", provenance_html))
    sections.append(("methods", "Methods and limits", (
        "<ul class='methods'>" + "".join(f"<li>{_e(n)}</li>" for n in METHOD_NOTES) + "</ul>"
        + ("<h4>Context notes</h4><ul>" + "".join(f"<li>{_e(n)}</li>" for n in notes) + "</ul>" if notes else "")
    )))

    downloads = [("PDF", base / "report.pdf"), ("Markdown", base / "report.md"), ("JSON data", base / "analysis.json")]
    if pymol:
        downloads.append(("PyMOL script", pymol))
    links = "".join(f'<a href="{_rel(p, base)}">{_e(label)}</a>' for label, p in downloads)
    nav = "".join(f'<a href="#{sid}">{_e(title.split(" (")[0])}</a>' for sid, title, _ in sections)
    body = "".join(f'<section id="{sid}"><h2>{_e(title)}</h2>{content}</section>' for sid, title, content in sections)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(result.identifier)} Structure Report</title>
<style>{CSS}</style>
</head>
<body>
<header class="top"><div class="wrap">
  <h1>{_e(result.identifier)}</h1>
  <p class="subtitle">{_e(audit.title or '')}</p>
  <div class="meta"><a href="../" title="All reports in the output directory">&#8592; All reports</a>{links}
    <button id="theme" type="button">Theme: auto</button>
    <span class="byline">Generated {date.today().isoformat()} by ProteinScope {__version__} · text: {_e(narrative.source)}</span></div>
</div></header>
<nav class="sections" aria-label="Sections"><div class="wrap">{nav}</div></nav>
<main class="wrap">{body}</main>
<footer class="wrap">All numbers are computed from the coordinates by ProteinScope. The pocket score ranks the
cavities of this structure and says nothing about a target's tractability; functional annotation is curated
UniProt evidence, not a measurement.</footer>
<dialog id="lightbox"><img alt=""><p></p></dialog>
<script>{JS}</script>
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")
    return path

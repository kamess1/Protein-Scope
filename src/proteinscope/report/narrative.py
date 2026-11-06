"""Rule-based narrative for reports that are generated without an LLM."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..pipeline import AnalysisResult


@dataclass
class Narrative:
    """Text sections of a report. `source` says who wrote them, and the report shows it."""

    executive_summary: str
    pocket_interpretation: str = ""
    ligand_interpretation: str = ""
    recommendations: list[str] = field(default_factory=list)
    source: str = "rule-based summary"


def rule_based_narrative(result: AnalysisResult) -> Narrative:
    audit = result.audit
    resolution = f"{audit.resolution_A:.2f} A" if audit.resolution_A else "unknown resolution"
    summary = [
        f"{result.identifier} ({audit.method or 'method unknown'}, {resolution}) has "
        f"{audit.residue_count} protein residues in {len(audit.chains)} chain(s); the audit grades "
        f"its quality as {audit.quality_grade}."
    ]
    recommendations = []
    if result.pockets:
        top = result.pockets[0]
        summary.append(
            f"{len(result.pockets)} pocket(s) were detected. The top-ranked pocket has a volume of "
            f"{top.volume:.0f} A3, an apolar lining fraction of {top.apolar_fraction:.2f} and a "
            f"heuristic druggability score of {top.druggability:.2f} ({top.druggability_class})."
        )
    else:
        summary.append("No pocket passed the volume and enclosure thresholds.")
        recommendations.append(
            "Consider an alternative conformation, because no pocket was detected in this one."
        )

    ligand_lines = []
    for label, profile in result.ligand_profiles.items():
        hits = result.ligand_pockets.get(label, [])
        location = f"lies in pocket {hits[0]['pocket_id']}" if hits else "does not overlap a ranked pocket"
        counts = ", ".join(f"{count} {kind}" for kind, count in sorted(profile.counts().items()))
        ligand_lines.append(
            f"{label} {location}; {profile.buried_fraction:.0%} of its surface is buried ({counts or 'no contacts'})."
        )
    if result.pockets and result.ligand_pockets:
        hit_ids = {entry["pocket_id"] for entries in result.ligand_pockets.values() for entry in entries}
        unoccupied = [p for p in result.pockets[:3] if p.pocket_id not in hit_ids and p.druggability_class != "low"]
        for pocket in unoccupied[:2]:
            recommendations.append(
                f"Pocket {pocket.pocket_id} scores {pocket.druggability:.2f} and holds no ligand in this "
                "structure; check it against other entries before treating it as a new site."
            )
    if audit.warnings:
        recommendations.append("Review the audit warnings before relying on residue-level findings.")
    return Narrative(
        executive_summary=" ".join(summary),
        ligand_interpretation=" ".join(ligand_lines),
        recommendations=recommendations,
    )

"""Functional annotation from UniProtKB, mapped onto the residues of a structure.

A pocket score says how well a cavity is shaped; it says nothing about what the cavity does.
UniProt knows the catalytic residues, cofactor and substrate binding sites, modification
sites and disease variants of the same protein, so a pocket that contains the active site or
three pathogenic variants can be told apart from an equally well-shaped surface dent.

Positions are UniProt sequence numbers and do not match PDB residue numbers: constructs are
truncated, tagged, renumbered and mutated. The mapping is therefore built by aligning each
chain's observed sequence against the reference sequence from the same entry, which also
survives missing loops and engineered point mutations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.PDB.Polypeptide import protein_letters_3to1

from .structure import ResidueKey, canonical_name, residue_key
from .webdata import get_json

FIELDS = (
    "accession,id,protein_name,gene_names,sequence,ft_act_site,ft_binding,ft_site,ft_mod_res,"
    "ft_carbohyd,ft_lipid,ft_disulfid,ft_crosslnk,ft_variant,ft_mutagen,ft_domain,ft_region,xref_pfam"
)
ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json?fields=" + FIELDS

# UniProt feature type -> the category the report groups by. Types absent here are ignored.
CATEGORIES = {
    "Active site": "catalytic",
    "Binding site": "binding",
    "Site": "functional",
    "Modified residue": "modification",
    "Glycosylation": "modification",
    "Lipidation": "modification",
    "Disulfide bond": "disulfide",
    "Cross-link": "disulfide",
    "Natural variant": "variant",
    "Mutagenesis": "mutagenesis",
}
# A disulfide or cross-link names its two partners, not an interval.
PAIRED_TYPES = frozenset({"Disulfide bond", "Cross-link"})
# Longer intervals describe a region, not a residue, and would tag every pocket.
MAX_SPAN = 25
MIN_IDENTITY = 0.6  # Below this the chain is a different protein and the mapping is dropped.


def entry(accession: str, cache_dir: Path | None = None) -> dict[str, Any]:
    return get_json(ENTRY_URL.format(accession=accession.strip().upper()), cache_dir)


def sequence(data: dict[str, Any]) -> str:
    return (data.get("sequence") or {}).get("value") or ""


def description(data: dict[str, Any]) -> dict[str, Any]:
    names = data.get("proteinDescription") or {}
    recommended = (names.get("recommendedName") or {}).get("fullName") or {}
    genes = [g.get("geneName", {}).get("value") for g in data.get("genes") or []]
    return {
        "accession": data.get("primaryAccession"),
        "uniprot_id": data.get("uniProtkbId"),
        "protein_name": recommended.get("value"),
        "genes": [gene for gene in genes if gene],
        "length": (data.get("sequence") or {}).get("length"),
        "pfam": [
            {"id": ref.get("id"), "name": (ref.get("properties") or [{}])[0].get("value")}
            for ref in data.get("uniProtKBCrossReferences") or []
            if ref.get("database") == "Pfam"
        ],
    }


def _positions(feature: dict[str, Any]) -> list[int]:
    location = feature.get("location") or {}
    start = (location.get("start") or {}).get("value")
    end = (location.get("end") or {}).get("value")
    if start is None or end is None:
        return []
    if feature["type"] in PAIRED_TYPES:
        return sorted({int(start), int(end)})
    if end - start + 1 > MAX_SPAN:
        return []
    return list(range(int(start), int(end) + 1))


def features(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Residue-level features, each with the UniProt positions it covers."""
    found = []
    for feature in data.get("features") or []:
        category = CATEGORIES.get(feature.get("type"))
        positions = _positions(feature) if category else []
        if not positions:
            continue
        # A binding site carries its partner in `ligand`, not in the description.
        ligand = (feature.get("ligand") or {}).get("name")
        text = (feature.get("description") or "").strip()
        found.append({
            "type": feature["type"],
            "category": category,
            "positions": positions,
            "description": text or (f"binds {ligand}" if ligand else None),
            "ligand": ligand,
            "feature_id": feature.get("featureId"),
        })
    return found


def regions(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Domains and named regions, kept separate because they span hundreds of residues."""
    found = []
    for feature in data.get("features") or []:
        if feature.get("type") not in {"Domain", "Region"}:
            continue
        location = feature.get("location") or {}
        start = (location.get("start") or {}).get("value")
        end = (location.get("end") or {}).get("value")
        if start is None or end is None:
            continue
        found.append({
            "type": feature["type"],
            "start": int(start),
            "end": int(end),
            "description": (feature.get("description") or "").strip() or None,
        })
    return found


def chain_sequence(residues: list[Any], chain: str) -> tuple[str, list[ResidueKey]]:
    """One-letter sequence of one chain in file order, with the matching residue keys."""
    letters, keys = [], []
    for residue in residues:
        key = residue_key(residue)
        if key.chain != chain:
            continue
        letters.append(protein_letters_3to1.get(canonical_name(key.name), "X"))
        keys.append(key)
    return "".join(letters), keys


def _aligner() -> PairwiseAligner:
    aligner = PairwiseAligner(mode="global")
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11.0
    aligner.extend_gap_score = -1.0
    aligner.end_gap_score = 0.0  # The construct is a fragment: free flanks on both sides.
    return aligner


def map_to_reference(observed: str, reference: str) -> tuple[dict[int, int], float]:
    """Align an observed chain against the reference and map its index to a UniProt position.

    Returns the index map (0-based observed -> 1-based UniProt) and the sequence identity
    over the aligned pairs, which tells the caller whether the mapping can be trusted.
    """
    if not observed or not reference:
        return {}, 0.0
    alignment = _aligner().align(reference, observed)[0]
    mapping, matches, pairs = {}, 0, 0
    for (ref_start, ref_end), (obs_start, obs_end) in zip(*alignment.aligned):
        for offset in range(int(ref_end) - int(ref_start)):
            # Cast: the aligner returns numpy integers, which are not JSON serialisable.
            mapping[int(obs_start) + offset] = int(ref_start) + offset + 1
            pairs += 1
            matches += reference[ref_start + offset] == observed[obs_start + offset]
    return mapping, (matches / pairs if pairs else 0.0)


def residue_positions(
    residues: list[Any], chains: list[str], reference: str
) -> tuple[dict[ResidueKey, int], dict[str, float]]:
    """UniProt position per residue key, for every chain that matches the reference."""
    positions: dict[ResidueKey, int] = {}
    identities: dict[str, float] = {}
    for chain in chains:
        observed, keys = chain_sequence(residues, chain)
        mapping, identity = map_to_reference(observed, reference)
        identities[chain] = round(identity, 3)
        if identity < MIN_IDENTITY:
            continue
        for index, position in mapping.items():
            positions[keys[index]] = position
    return positions, identities

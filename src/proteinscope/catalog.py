"""The target catalogue offered in the browser, and the identifiers the browser may submit.

The catalogue is a JSON file shipped in `proteinscope/data`: well-known drug targets, one
structure each, with a sentence on why it is interesting. It exists so that someone who
does not know PDB IDs can still pick a protein and get a report.

`validate_identifier` is the gate for anything typed into the web page. It accepts PDB IDs
and AlphaFold IDs only, never a file path, so a page in the browser can make the server
download public structures but never read a file from this machine.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

from . import alphafold

CATALOG_FILE = "catalog.json"
_PDB_ID = re.compile(r"^[0-9][A-Z0-9]{3}$")
MAX_IDENTIFIER_LENGTH = 24


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    name: str
    gene: str
    area: str
    ligand: str
    why: str


def validate_identifier(text: str) -> str:
    """Normalise a PDB or AlphaFold ID, or raise ValueError for anything else.

    Paths, URLs and anything with a separator are refused by construction: only the two
    identifier patterns are accepted, after trimming whitespace.
    """
    if not isinstance(text, str):
        raise ValueError("Enter a PDB ID such as 1HSG or an AlphaFold ID such as AF-P00533.")
    candidate = text.strip()
    if not candidate or len(candidate) > MAX_IDENTIFIER_LENGTH:
        raise ValueError("Enter a PDB ID such as 1HSG or an AlphaFold ID such as AF-P00533.")
    upper = candidate.upper()
    if _PDB_ID.match(upper):
        return upper
    accession = alphafold.parse_alphafold_id(candidate)
    if accession:
        return alphafold.identifier(accession)
    raise ValueError(
        f"{candidate[:MAX_IDENTIFIER_LENGTH]!r} is not a PDB ID (4 characters, such as 1HSG) "
        "or an AlphaFold ID (such as AF-P00533)."
    )


@lru_cache(maxsize=1)
def load_catalog() -> tuple[tuple[str, ...], tuple[CatalogEntry, ...]]:
    """The areas in display order and the entries, validated once."""
    data = json.loads(files("proteinscope.data").joinpath(CATALOG_FILE).read_text(encoding="utf-8"))
    areas = tuple(data["areas"])
    entries = []
    for raw in data["entries"]:
        entry = CatalogEntry(**{field: str(raw[field]) for field in CatalogEntry.__dataclass_fields__})
        if validate_identifier(entry.id) != entry.id:
            raise ValueError(f"Catalogue entry {entry.id!r} is not in normalised form")
        if entry.area not in areas:
            raise ValueError(f"Catalogue entry {entry.id} has unknown area {entry.area!r}")
        entries.append(entry)
    return areas, tuple(entries)

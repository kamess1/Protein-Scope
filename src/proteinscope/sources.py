"""The public data services ProteinScope reads, with their licences and how to credit them.

A report mixes numbers computed here with data taken from these services, and each comes
with terms. The reports list the services they actually used, the licence that applies to
that part and the reference the provider asks for. Terms were checked at the providers'
pages on 2026-11-11 (links below); confirm them before redistributing reports widely.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class DataSource:
    name: str
    hosts: tuple[str, ...]
    path_prefix: str
    licence: str
    licence_url: str
    terms_url: str
    credit: str
    citation: str


SOURCES: tuple[DataSource, ...] = (
    DataSource(
        name="RCSB PDB",
        hosts=("files.rcsb.org", "data.rcsb.org", "search.rcsb.org"),
        path_prefix="",
        licence="CC0 1.0",
        licence_url="https://creativecommons.org/publicdomain/zero/1.0/",
        terms_url="https://www.rcsb.org/pages/usage-policy",
        credit="Structures and entry data from the RCSB Protein Data Bank (rcsb.org). Credit the authors "
               "of each structure through its primary citation.",
        citation="Berman, H. M. et al. The Protein Data Bank. Nucleic Acids Res. 28, 235-242 (2000).",
    ),
    DataSource(
        name="AlphaFold DB",
        hosts=("alphafold.ebi.ac.uk",),
        path_prefix="",
        licence="CC BY 4.0",
        licence_url="https://creativecommons.org/licenses/by/4.0/",
        terms_url="https://alphafold.ebi.ac.uk/assets/License-Disclaimer.pdf",
        credit="AlphaFold Data Copyright (2021) DeepMind Technologies Limited. Predicted models from the "
               "AlphaFold Protein Structure Database (EMBL-EBI), for theoretical modelling only.",
        citation="Jumper, J. et al. Highly accurate protein structure prediction with AlphaFold. "
                 "Nature 596, 583-589 (2021).",
    ),
    DataSource(
        name="UniProt",
        hosts=("rest.uniprot.org",),
        path_prefix="",
        licence="CC BY 4.0",
        licence_url="https://creativecommons.org/licenses/by/4.0/",
        terms_url="https://ftp.uniprot.org/pub/databases/uniprot/LICENSE",
        credit="Functional annotation from UniProtKB (uniprot.org).",
        citation="The UniProt Consortium. UniProt: the Universal Protein Knowledgebase in 2025. "
                 "Nucleic Acids Res. 53, D609-D617 (2025).",
    ),
    DataSource(
        name="ChEMBL",
        hosts=("www.ebi.ac.uk",),
        path_prefix="/chembl/",
        licence="CC BY-SA 3.0",
        licence_url="https://creativecommons.org/licenses/by-sa/3.0/",
        terms_url="https://chembl.gitbook.io/chembl-interface-documentation/about",
        credit="Bioactivity and drug data from ChEMBL (EMBL-EBI). Share-alike: material derived from "
               "it must be passed on under the same licence.",
        citation="Zdrazil, B. et al. The ChEMBL Database in 2023: a drug discovery platform spanning "
                 "multiple bioactivity data types and time periods. Nucleic Acids Res. 52, D1180-D1192 (2024).",
    ),
)

BY_NAME = {source.name: source for source in SOURCES}


def source_for_url(url: str) -> DataSource | None:
    parts = urlsplit(url)
    for source in SOURCES:
        if parts.hostname in source.hosts and parts.path.startswith(source.path_prefix):
            return source
    return None

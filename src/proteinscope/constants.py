"""Physicochemical reference data used across the analysis modules."""

from __future__ import annotations

# Bondi van der Waals radii (Angstrom); DEFAULT_RADIUS covers anything rarer.
VDW_RADII = {
    "H": 1.10, "D": 1.10, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "P": 1.80,
    "S": 1.80, "CL": 1.75, "BR": 1.85, "I": 1.98, "SE": 1.90, "B": 1.92,
    "NA": 2.27, "K": 2.75, "MG": 1.73, "CA": 2.31, "ZN": 1.39, "FE": 1.56,
    "MN": 1.61, "CU": 1.40, "NI": 1.63, "CO": 1.52, "CD": 1.58, "HG": 1.55,
}
DEFAULT_RADIUS = 1.8

# Kyte & Doolittle (1982) hydropathy index.
KYTE_DOOLITTLE = {
    "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5, "GLN": -3.5,
    "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5, "LEU": 3.8, "LYS": -3.9,
    "MET": 1.9, "PHE": 2.8, "PRO": -1.6, "SER": -0.8, "THR": -0.7, "TRP": -0.9,
    "TYR": -1.3, "VAL": 4.2,
}

# Maximum residue SASA in Angstrom^2 (Tien et al. 2013, theoretical scale).
MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}

STANDARD_RESIDUES = frozenset(KYTE_DOOLITTLE)

# Common modified residues, mapped to the parent amino acid.
MODIFIED_RESIDUES = {
    "MSE": "MET", "SEP": "SER", "TPO": "THR", "PTR": "TYR", "CSO": "CYS",
    "HIP": "HIS", "HID": "HIS", "HIE": "HIS", "CME": "CYS", "KCX": "LYS",
    "LLP": "LYS", "M3L": "LYS", "MLY": "LYS", "PCA": "GLU", "CSS": "CYS",
    "OCS": "CYS", "HYP": "PRO", "CSD": "CYS", "TYS": "TYR",
}

WATER_NAMES = frozenset({"HOH", "WAT", "H2O", "DOD", "DIS"})

METAL_ELEMENTS = frozenset({
    "LI", "NA", "K", "RB", "CS", "MG", "CA", "SR", "BA", "MN", "FE", "CO", "NI",
    "CU", "ZN", "CD", "HG", "PT", "AU", "AG", "MO", "W",
})

# Formal charges of monoatomic ions, in units of e.
ION_CHARGES = {
    "NA": 1.0, "K": 1.0, "LI": 1.0, "RB": 1.0, "CS": 1.0, "MG": 2.0, "CA": 2.0,
    "SR": 2.0, "BA": 2.0, "MN": 2.0, "FE": 2.0, "CO": 2.0, "NI": 2.0, "CU": 2.0,
    "ZN": 2.0, "CD": 2.0, "HG": 2.0, "CL": -1.0, "BR": -1.0, "I": -1.0,
}

# Crystallisation additives and buffers: reported, but not treated as ligands.
CRYSTAL_ADDITIVES = frozenset({
    "GOL", "EDO", "PEG", "PGE", "PG4", "1PE", "P6G", "MPD", "DMS", "ACT", "ACE",
    "SO4", "PO4", "NO3", "CO3", "BME", "DTT", "FMT", "IMD", "TRS", "MES", "EPE",
    "HEZ", "IPA", "EOH", "MOH", "BU3", "CIT", "TAR", "NH4", "SCN", "AZI", "CAC",
    "UNX", "UNL", "PE4", "PEU", "BOG", "LDA", "MLI", "PGO", "POL", "FLC", "BCT",
    "CL", "NA", "K", "BR", "IOD",
})

# Partial charges for the titratable groups at pH 7, in units of e.
# HIS carries about 10 % protonation; termini are handled in electrostatics.py.
SIDECHAIN_CHARGES = {
    "ASP": {"OD1": -0.5, "OD2": -0.5},
    "GLU": {"OE1": -0.5, "OE2": -0.5},
    "LYS": {"NZ": 1.0},
    "ARG": {"NH1": 0.5, "NH2": 0.5},
    "HIS": {"ND1": 0.05, "NE2": 0.05},
}

AROMATIC_RING_ATOMS = {
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "TRP": ("CG", "CD1", "NE1", "CE2", "CD2", "CE3", "CZ2", "CZ3", "CH2"),
}

# Heavy-atom hydrogen-bond capabilities of protein atoms (backbone N/O included).
DONOR_ATOMS = {
    "ARG": {"NE", "NH1", "NH2"}, "ASN": {"ND2"}, "GLN": {"NE2"}, "HIS": {"ND1", "NE2"},
    "LYS": {"NZ"}, "SER": {"OG"}, "THR": {"OG1"}, "TRP": {"NE1"}, "TYR": {"OH"},
    "CYS": {"SG"},
}
ACCEPTOR_ATOMS = {
    "ASN": {"OD1"}, "ASP": {"OD1", "OD2"}, "GLN": {"OE1"}, "GLU": {"OE1", "OE2"},
    "HIS": {"ND1", "NE2"}, "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"}, "MET": {"SD"},
    "CYS": {"SG"},
}


def vdw_radius(element: str) -> float:
    return VDW_RADII.get(element.upper(), DEFAULT_RADIUS)

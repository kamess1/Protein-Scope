"""Heuristic druggability score from pocket shape and chemistry.

The weights encode established trends: druggable sites are large enough for a drug-like
molecule, enclosed and predominantly apolar, with few buried charges (Cheng et al. 2007;
Volkamer et al. 2012). They were set by hand and not fitted, so the score ranks pockets
within a structure. It is not a calibrated probability.
"""

from __future__ import annotations

from .pockets import Pocket

WEIGHTS = {"size": 0.30, "enclosure": 0.25, "apolarity": 0.25, "depth": 0.10, "charge": 0.10}
HIGH_THRESHOLD = 0.65
MODERATE_THRESHOLD = 0.45


def _ramp(value: float, low: float, high: float) -> float:
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _size_score(volume: float) -> float:
    """Full score across a drug-sized window, tapering for tiny pockets and huge channels."""
    if volume <= 1200.0:
        return _ramp(volume, 100.0, 500.0)
    return 1.0 - 0.6 * _ramp(volume, 1200.0, 4000.0)


def score_pocket(pocket: Pocket) -> Pocket:
    charge_density = pocket.charged_residues / max(len(pocket.lining_residues), 1)
    components = {
        "size": _size_score(pocket.volume),
        "enclosure": _ramp(pocket.enclosure, 0.70, 0.95),
        "apolarity": _ramp(pocket.apolar_fraction, 0.40, 0.75),
        "depth": _ramp(pocket.depth, 2.0, 5.0),
        "charge": 1.0 - _ramp(charge_density, 0.05, 0.35),
    }
    pocket.score_components = components
    pocket.druggability = sum(WEIGHTS[name] * value for name, value in components.items())
    if pocket.druggability >= HIGH_THRESHOLD:
        pocket.druggability_class = "high"
    elif pocket.druggability >= MODERATE_THRESHOLD:
        pocket.druggability_class = "moderate"
    else:
        pocket.druggability_class = "low"
    return pocket


def rank_pockets(pockets: list[Pocket]) -> list[Pocket]:
    """Score pockets, sort them best first and number them from 1 in that order."""
    ranked = sorted((score_pocket(pocket) for pocket in pockets), key=lambda p: -p.druggability)
    for rank, pocket in enumerate(ranked, start=1):
        pocket.pocket_id = rank
    return ranked

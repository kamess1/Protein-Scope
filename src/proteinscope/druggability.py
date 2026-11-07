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


ANCHOR_POTENTIAL = 2.5  # kT/e; above this the pocket field steers charged ligand groups.


def site_tags(pocket: Pocket) -> list[str]:
    """Describe what kind of site a pocket is, independent of the drug-likeness score.

    The score favours large apolar pockets, so small charged sites that bind fragments or
    charged ligands (trypsin S1 and benzamidine, for example) rank low. The tags make such
    sites visible instead of reweighting the score for one binding mode.
    """
    tags = []
    if pocket.volume < 300:
        tags.append("fragment-sized")
    elif pocket.volume > 2000:
        tags.append("large cavity or channel")
    if pocket.enclosure < 0.75:
        tags.append("shallow")
    if pocket.apolar_fraction >= 0.5 and 300 <= pocket.volume <= 2000 and pocket.enclosure >= 0.8:
        tags.append("drug-like apolar")
    elif pocket.apolar_fraction < 0.4:
        tags.append("polar")
    if pocket.mean_potential <= -ANCHOR_POTENTIAL:
        tags.append("anionic anchor: favours cationic groups")
    elif pocket.mean_potential >= ANCHOR_POTENTIAL:
        tags.append("cationic anchor: favours anionic groups")
    return tags


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
    pocket.site_tags = site_tags(pocket)
    return pocket


def rank_pockets(pockets: list[Pocket]) -> list[Pocket]:
    """Score pockets, sort them best first and number them from 1 in that order."""
    ranked = sorted((score_pocket(pocket) for pocket in pockets), key=lambda p: -p.druggability)
    for rank, pocket in enumerate(ranked, start=1):
        pocket.pocket_id = rank
    return ranked

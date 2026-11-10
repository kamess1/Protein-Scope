"""How likely a pocket is to be the ligand-binding site, from its shape and chemistry.

Two scores live here. The **heuristic** encodes established trends by hand: druggable sites
are large enough for a drug-like molecule, enclosed and predominantly apolar, with few buried
charges (Cheng et al. 2007; Volkamer et al. 2012). The **fitted** score is a logistic
regression over the same pocket descriptors, trained on a benchmark of structures with a
drug-like ligand bound, where the label says whether the pocket holds that ligand
(`scripts/pocket_model.py`). Its output is a calibrated probability and it is used when
`data/ligandability.json` ships with the package; the heuristic is the fallback.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from importlib.resources import files
from typing import Any

from .pockets import Pocket

WEIGHTS = {"size": 0.30, "enclosure": 0.25, "apolarity": 0.25, "depth": 0.10, "charge": 0.10}
HIGH_THRESHOLD = 0.65
MODERATE_THRESHOLD = 0.45

MODEL_FILE = "ligandability.json"
FEATURES = (
    "log_volume", "enclosure", "apolar_fraction", "depth",
    "charge_density", "polar_density", "mean_hydropathy", "mean_potential",
)


def feature_vector(pocket: Pocket) -> dict[str, float]:
    """The descriptors the fitted model reads. Training and scoring must share this function."""
    lining = max(len(pocket.lining_residues), 1)
    return {
        "log_volume": math.log10(max(pocket.volume, 1.0)),
        "enclosure": pocket.enclosure,
        "apolar_fraction": pocket.apolar_fraction,
        "depth": pocket.depth,
        "charge_density": pocket.charged_residues / lining,
        "polar_density": pocket.donor_acceptor_atoms / lining,
        "mean_hydropathy": pocket.mean_hydropathy,
        "mean_potential": pocket.mean_potential,
    }


@lru_cache(maxsize=1)
def load_model() -> dict[str, Any] | None:
    """The fitted coefficients shipped in `proteinscope/data`, or None if absent."""
    resource = files("proteinscope.data").joinpath(MODEL_FILE)
    if not resource.is_file():
        return None
    model = json.loads(resource.read_text(encoding="utf-8"))
    if list(model.get("features", ())) != list(FEATURES):
        raise ValueError(
            f"{MODEL_FILE} was fitted on {model.get('features')}, but this version expects {list(FEATURES)}."
        )
    return model


def ligandability(pocket: Pocket, model: dict[str, Any] | None = None) -> float | None:
    """Fitted probability that this pocket is the ligand-binding site, or None without a model."""
    model = model or load_model()
    if model is None:
        return None
    values = feature_vector(pocket)
    logit = model["intercept"]
    for name, coefficient, mean, scale in zip(
        model["features"], model["coefficients"], model["mean"], model["scale"]
    ):
        logit += coefficient * (values[name] - mean) / (scale or 1.0)
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logit))))


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


def heuristic_score(pocket: Pocket) -> tuple[float, dict[str, float]]:
    charge_density = pocket.charged_residues / max(len(pocket.lining_residues), 1)
    components = {
        "size": _size_score(pocket.volume),
        "enclosure": _ramp(pocket.enclosure, 0.70, 0.95),
        "apolarity": _ramp(pocket.apolar_fraction, 0.40, 0.75),
        "depth": _ramp(pocket.depth, 2.0, 5.0),
        "charge": 1.0 - _ramp(charge_density, 0.05, 0.35),
    }
    return sum(WEIGHTS[name] * value for name, value in components.items()), components


def score_pocket(pocket: Pocket, model: dict[str, Any] | None = None) -> Pocket:
    """Attach both scores, the class and the site tags. `druggability` is the headline value."""
    score, components = heuristic_score(pocket)
    pocket.heuristic_score = score
    pocket.score_components = components

    model = model if model is not None else load_model()
    fitted = ligandability(pocket, model) if model else None
    if fitted is None:
        pocket.druggability = score
        pocket.score_model = "heuristic"
        high, moderate = HIGH_THRESHOLD, MODERATE_THRESHOLD
    else:
        pocket.druggability = fitted
        pocket.score_model = f"fitted logistic ({model['version']})"
        high = model["thresholds"]["high"]
        moderate = model["thresholds"]["moderate"]

    if pocket.druggability >= high:
        pocket.druggability_class = "high"
    elif pocket.druggability >= moderate:
        pocket.druggability_class = "moderate"
    else:
        pocket.druggability_class = "low"
    pocket.site_tags = site_tags(pocket)
    return pocket


def rank_pockets(pockets: list[Pocket]) -> list[Pocket]:
    """Score pockets, sort them best first and number them from 1 in that order."""
    model = load_model()
    ranked = sorted((score_pocket(pocket, model) for pocket in pockets), key=lambda p: -p.druggability)
    for rank, pocket in enumerate(ranked, start=1):
        pocket.pocket_id = rank
    return ranked


def score_description(model: dict[str, Any] | None = None) -> str:
    """One paragraph on what the pocket score is, read from the model actually shipped.

    The reports, the MCP instructions and the agent prompts all use this, so retraining the
    model updates every statement about it instead of leaving stale numbers behind.
    """
    model = model if model is not None else load_model()
    if model is None:
        return (
            "Pocket score: hand-weighted heuristic of size, enclosure, apolarity, depth and buried "
            "charge. It ranks the pockets of one structure; it is not a probability and not "
            "comparable across proteins."
        )
    training, performance = model["training"], model["performance"]
    baseline = performance.get("baseline_heuristic") or {}
    comparison = (
        f", against {baseline['roc_auc']:.2f} and {baseline['top1_accuracy']:.0%} for the "
        "hand-weighted heuristic it replaces"
        if baseline else ""
    )
    return (
        f"Pocket score: logistic regression over {len(model['features'])} descriptors, fitted on "
        f"{training['entries']} PDB entries ({training['pockets']} pockets) where the label says "
        f"whether the entry's ligand sits in the pocket. Cross-validated "
        f"{performance['cv_folds']}-fold grouped by entry: ROC-AUC {performance['roc_auc']:.2f} "
        f"and the top-ranked pocket correct in {performance['top1_accuracy']:.0%} of entries"
        f"{comparison}. It is a probability for the cavities of this structure, not a measure of "
        "how tractable the target is."
    )


SCORE_WEAKNESS = (
    "Where the pocket score fails: the training set has no entry that is mostly polar, so the "
    "model treats a high density of polar atoms as evidence against a binding site and can rank "
    "a nucleotide or cofactor site below a smaller apolar one. Check the functional annotation "
    "before trusting the ranking alone."
)

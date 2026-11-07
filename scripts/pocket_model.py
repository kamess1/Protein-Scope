#!/usr/bin/env python3
"""Fit the pocket ligandability model used by `proteinscope.druggability`.

Two steps, because collecting is slow and fitting is instant:

    python scripts/pocket_model.py collect        # downloads and analyses the benchmark
    python scripts/pocket_model.py fit            # fits and writes the shipped coefficients

The label comes from the structure itself: a pocket is positive when the entry's drug-like
ligand sits in it (>= 50 % of its atoms within 2.5 A of a pocket point), and every other
pocket of the same entry is negative. No external druggability annotation is needed, and
the question the model answers is the one that matters when ranking: of the cavities in
this structure, which one is the real binding site?
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proteinscope.druggability import FEATURES, feature_vector  # noqa: E402
from proteinscope.pockets import ligand_pocket_overlap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ROWS_PATH = ROOT / "scripts" / "pocket_benchmark.json"
MODEL_PATH = ROOT / "src" / "proteinscope" / "data" / "ligandability.json"

MIN_LIGAND_ATOMS = 9      # Below this a component is a buffer or cryoprotectant, not a ligand.
POSITIVE_OVERLAP = 0.5    # Fraction of ligand atoms that must lie in the pocket.

# A spread of families with a well-characterised drug-like ligand bound: proteases,
# kinases, nuclear receptors, GPCRs, oxidoreductases, chaperones, transferases.
# Entries that fail to download, carry no drug-like ligand or yield no pocket are skipped.
BENCHMARK = (
    "1HSG", "1HPV", "1HVR", "3PTB", "1BJU", "1PPC", "1DWD", "1ETS", "1A42", "1AZM",
    "4DFR", "1STP", "1FKF", "1ACJ", "1EVE", "1E66", "2HYY", "1IEP", "1M17", "1XKK",
    "1T46", "1AQ1", "1DI8", "2VTA", "1HCK", "1STC", "1YDT", "3ERT", "1GWR", "2AM9",
    "1E3G", "2PRG", "1FM6", "2RH1", "3EML", "4EIY", "2VT4", "1UYG", "2BSM", "1L2S",
    "1QPE", "1OYT", "3POZ", "1KE5", "1CX2", "4COX", "3LN1", "1O86", "1C1B", "1EJN",
    "1NVQ", "1SJ0", "3NY8", "1OPJ", "1FPU", "2OIQ", "3PP0", "4ASD", "2RKU", "1Y57",
    "1BZC", "1CBX", "1E1X", "1H1S", "1MMV", "1N46", "1OF6", "1P44", "1R58", "1S19",
    "1UOU", "1V0P", "1W2G", "1YVF", "1Z95", "2BM2", "2CET", "2F7O", "2IWI", "2OBJ",
)


def _row(pdb_id: str) -> dict:
    """Analyse one entry and return its pockets with labels, or an error note."""
    from proteinscope.pipeline import run_analysis

    try:
        result = run_analysis(pdb_id, probe_pockets=0, context=False)
    except Exception as exc:  # Any entry may be missing, obsolete or too large; just skip it.
        return {"pdb_id": pdb_id, "error": f"{type(exc).__name__}: {exc}"}

    ligands = [lig for lig in result.structure.ligands if len(lig.atoms) >= MIN_LIGAND_ATOMS]
    if not ligands:
        return {"pdb_id": pdb_id, "error": f"no ligand with >= {MIN_LIGAND_ATOMS} heavy atoms"}
    if not result.pockets:
        return {"pdb_id": pdb_id, "error": "no pocket detected"}

    pockets = []
    for pocket in result.pockets:
        overlap = max(ligand_pocket_overlap(lig.atoms.coords, pocket) for lig in ligands)
        pockets.append({
            "pocket_id": pocket.pocket_id,
            "label": int(overlap >= POSITIVE_OVERLAP),
            "overlap": round(overlap, 3),
            "volume_A3": round(pocket.volume, 1),
            "features": {name: round(value, 4) for name, value in feature_vector(pocket).items()},
        })
    if not any(p["label"] for p in pockets):
        return {"pdb_id": pdb_id, "error": "ligand matched no pocket"}
    return {
        "pdb_id": pdb_id,
        "resolution_A": result.audit.resolution_A,
        "ligands": [lig.label for lig in ligands],
        "pockets": pockets,
    }


def collect(entries: tuple[str, ...], workers: int) -> None:
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, row in enumerate(pool.map(_row, entries), start=1):
            state = row["error"] if "error" in row else (
                f"{sum(p['label'] for p in row['pockets'])}/{len(row['pockets'])} positive"
            )
            print(f"[{done}/{len(entries)}] {row['pdb_id']}: {state}", flush=True)
            results.append(row)
    usable = [row for row in results if "error" not in row]
    ROWS_PATH.write_text(json.dumps({
        "collected": date.today().isoformat(),
        "min_ligand_atoms": MIN_LIGAND_ATOMS,
        "positive_overlap": POSITIVE_OVERLAP,
        "skipped": {row["pdb_id"]: row["error"] for row in results if "error" in row},
        "entries": usable,
    }, indent=1), encoding="utf-8")
    pockets = sum(len(row["pockets"]) for row in usable)
    positives = sum(p["label"] for row in usable for p in row["pockets"])
    print(f"\n{len(usable)}/{len(entries)} entries usable, {pockets} pockets, {positives} positive")
    print(f"written to {ROWS_PATH}")


def _fit(x: np.ndarray, y: np.ndarray, penalty: float = 1.0, steps: int = 400) -> np.ndarray:
    """L2-penalised logistic regression by Newton-Raphson (IRLS). Returns [intercept, *coef]."""
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(design.shape[1])
    ridge = penalty * np.eye(design.shape[1])
    ridge[0, 0] = 0.0  # Never shrink the intercept: it carries the class balance.
    for _ in range(steps):
        probability = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -60, 60)))
        weights = np.clip(probability * (1.0 - probability), 1e-6, None)
        gradient = design.T @ (y - probability) - ridge @ beta
        hessian = design.T @ (design * weights[:, None]) + ridge
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def _auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via the rank form of the Mann-Whitney statistic."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks inside ties so equal scores count as a coin flip.
    sorted_scores = scores[order]
    start = 0
    for index in range(1, len(scores) + 1):
        if index == len(scores) or sorted_scores[index] != sorted_scores[start]:
            ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    positives, negatives = y.sum(), len(y) - y.sum()
    if not positives or not negatives:
        return float("nan")
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def fit(folds: int = 5, penalty: float = 1.0) -> None:
    data = json.loads(ROWS_PATH.read_text(encoding="utf-8"))
    entries = data["entries"]
    groups = np.array([i for i, row in enumerate(entries) for _ in row["pockets"]])
    x = np.array([[p["features"][name] for name in FEATURES] for row in entries for p in row["pockets"]])
    y = np.array([p["label"] for row in entries for p in row["pockets"]], dtype=float)

    # Cross-validate by entry, never by pocket: pockets of one structure are not independent.
    rng = np.random.default_rng(0)
    assignment = rng.permutation(len(entries)) % folds
    held_out = np.empty(len(y))
    for fold in range(folds):
        test = np.isin(groups, np.flatnonzero(assignment == fold))
        mean, scale = x[~test].mean(axis=0), x[~test].std(axis=0)
        scale[scale == 0] = 1.0
        beta = _fit((x[~test] - mean) / scale, y[~test], penalty)
        design = np.column_stack([np.ones(test.sum()), (x[test] - mean) / scale])
        held_out[test] = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -60, 60)))

    auc = _auc(y, held_out)
    top1 = np.mean([
        y[groups == i][np.argmax(held_out[groups == i])] for i in range(len(entries))
    ])
    heuristic_top1 = np.mean([
        1.0 if entries[i]["pockets"][0]["label"] else 0.0 for i in range(len(entries))
    ])

    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    beta = _fit((x - mean) / scale, y, penalty)

    # Thresholds from the held-out probabilities: "high" is where precision is already good,
    # "moderate" is the point that still recalls most true sites.
    positives = np.sort(held_out[y == 1])
    high = float(np.quantile(positives, 0.5).round(3))
    moderate = float(np.quantile(positives, 0.1).round(3))

    model = {
        "version": f"logistic-{data['collected']}",
        "trained": date.today().isoformat(),
        "features": list(FEATURES),
        "mean": [round(v, 6) for v in mean],
        "scale": [round(v, 6) for v in scale],
        "coefficients": [round(v, 6) for v in beta[1:]],
        "intercept": round(float(beta[0]), 6),
        "thresholds": {"high": high, "moderate": moderate},
        "training": {
            "entries": len(entries),
            "pockets": int(len(y)),
            "positive_pockets": int(y.sum()),
            "l2_penalty": penalty,
            "label": f"pocket holds a ligand of >= {data["min_ligand_atoms"]} heavy atoms "
                     f"with >= {data['positive_overlap']:.0%} of its atoms inside",
            "pdb_ids": [row["pdb_id"] for row in entries],
        },
        "performance": {
            "cv_folds": folds,
            "grouped_by": "pdb entry",
            "roc_auc": round(float(auc), 3),
            "top1_accuracy": round(float(top1), 3),
            "top1_accuracy_heuristic": round(float(heuristic_top1), 3),
        },
    }
    MODEL_PATH.write_text(json.dumps(model, indent=1), encoding="utf-8")

    print(f"{len(entries)} entries, {len(y)} pockets, {int(y.sum())} positive")
    print(f"grouped {folds}-fold CV: ROC-AUC {auc:.3f}, top-1 {top1:.3f} (heuristic {heuristic_top1:.3f})")
    print("thresholds:", model["thresholds"])
    for name, coefficient in zip(FEATURES, beta[1:]):
        print(f"  {name:>17}  {coefficient:+.3f}")
    print(f"written to {MODEL_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    gather = sub.add_parser("collect", help="download and analyse the benchmark entries")
    gather.add_argument("--workers", type=int, default=4)
    gather.add_argument("--entries", nargs="*", help="override the built-in benchmark list")
    model = sub.add_parser("fit", help="fit the logistic model from the collected rows")
    model.add_argument("--folds", type=int, default=5)
    model.add_argument("--penalty", type=float, default=1.0)
    args = parser.parse_args()
    if args.command == "collect":
        collect(tuple(args.entries) if args.entries else BENCHMARK, args.workers)
    else:
        fit(args.folds, args.penalty)


if __name__ == "__main__":
    main()

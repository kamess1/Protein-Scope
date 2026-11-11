#!/usr/bin/env python3
"""External validation: ProteinScope against fpocket and P2Rank on COACH420.

COACH420 is a standard benchmark of 420 single-chain protein-ligand complexes (Krivák &
Hoksza, J. Cheminform. 10, 39 (2018)). The P2Rank authors publish it, together with the
predictions of fpocket and P2Rank, at https://github.com/rdk/p2rank-datasets. This script
uses the `coach420(mlig)` list, in which the relevant ligands of every protein are named
explicitly from Binding MOAD, so the ligand definition does not depend on a filter we would
have to reimplement.

All three methods are scored by the same code on the same ligands:

* a ligand counts as found when the centre of one of the top-ranked pockets lies within
  4 Å of any of its heavy atoms (the DCA criterion of the P2Rank paper);
* Top-n considers the n best pockets, where n is the number of relevant ligands in that
  structure, and Top-(n+2) two more; every ligand has equal weight (ligand-centric count).

Entries that also appear in ProteinScope's training set are reported separately, so the
headline numbers measure generalisation. The files are downloaded into the ProteinScope
cache, not into this repository: the dataset repository states no licence of its own,
although its structures come from the PDB (CC0).

    python scripts/benchmark_coach420.py            # download, evaluate, write docs/validation*
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DATASET_REPO = "rdk/p2rank-datasets"
RAW = f"https://raw.githubusercontent.com/{DATASET_REPO}/{{ref}}/{{path}}"
LIGAND_LIST = "coach420(mlig).ds"
FPOCKET_LIST = "coach420-fpocket.ds"
P2RANK_LIST = "coach420-p2rank.ds"
DCA_CUTOFF = 4.0
RESULTS_JSON = ROOT / "docs" / "validation" / "coach420.json"
REPORT_MD = ROOT / "docs" / "validation.md"

# Table 3 of Krivák & Hoksza (2018), DCA 4 Å, success rate in %. These use the default
# COACH420 list and the authors' own ligand filter, not the MOAD-based list used here,
# so they are context, not a like-for-like comparison.
PUBLISHED = {
    "fpocket": {"top_n": 56.4, "top_n2": 68.9},
    "P2Rank": {"top_n": 72.0, "top_n2": 78.3},
}


# --- data ----------------------------------------------------------------------------


@dataclass
class Case:
    protein: str             # Path inside the dataset repository, e.g. coach420/1a26A.pdb.
    codes: tuple[str, ...]   # Relevant ligand codes from MOAD.
    fpocket: str | None = None
    p2rank: str | None = None

    @property
    def pdb_id(self) -> str:
        return Path(self.protein).stem[:4].upper()


def _cache_dir() -> Path:
    from proteinscope.rcsb import default_cache_dir

    return default_cache_dir() / "benchmarks" / "p2rank-datasets"


def _get(url: str, attempts: int = 4) -> bytes:
    import requests

    for attempt in range(attempts):
        try:
            response = requests.get(url, timeout=60, headers={"User-Agent": "ProteinScope-benchmark"})
            if response.status_code == 200:
                return response.content
            if response.status_code == 404:
                raise FileNotFoundError(url)
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not download {url}")


def dataset_ref() -> str:
    """The commit the files are taken from, pinned so the result can be reproduced."""
    import requests

    response = requests.get(f"https://api.github.com/repos/{DATASET_REPO}/commits/master", timeout=60)
    response.raise_for_status()
    return response.json()["sha"]


def fetch(path: str, ref: str) -> Path:
    target = _cache_dir() / ref[:12] / path
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        data = _get(RAW.format(ref=ref, path=path.replace("(", "%28").replace(")", "%29")))
        target.with_suffix(target.suffix + ".part").write_bytes(data)
        target.with_suffix(target.suffix + ".part").replace(target)
    return target


def _rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("HEADER") or line.startswith("PARAM"):
            continue
        rows.append(line.split())
    return rows


def load_cases(ref: str) -> list[Case]:
    cases = {row[0]: Case(row[0], tuple(row[1].split(","))) for row in _rows(fetch(LIGAND_LIST, ref).read_text())}
    for row in _rows(fetch(FPOCKET_LIST, ref).read_text()):
        if row[1] in cases:
            cases[row[1]].fpocket = row[0]
    for row in _rows(fetch(P2RANK_LIST, ref).read_text()):
        if row[1] in cases:
            cases[row[1]].p2rank = row[0]
    return sorted(cases.values(), key=lambda case: case.protein)


# --- parsing -------------------------------------------------------------------------


def _atom(line: str) -> tuple[str, str, str, np.ndarray, str]:
    """(residue name, residue id, element, coordinates, record) of one PDB atom line."""
    name = line[17:20].strip()
    residue = f"{line[21]}:{line[22:26].strip()}{line[26].strip()}"
    element = line[76:78].strip() or line[12:14].strip().lstrip("0123456789")[:1]
    xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return name, residue, element.upper(), xyz, line[:6].strip()


BOND_DISTANCE = 2.0     # Residues this close are covalently joined: one ligand (a sugar chain, say).
CONTACT_DISTANCE = 4.0  # A ligand farther than this from every protein atom binds another chain.


def ligand_residues(pdb_text: str, codes: tuple[str, ...]) -> list[np.ndarray]:
    """Heavy-atom coordinates of every residue whose name is one of the relevant codes."""
    groups: dict[tuple[str, str], list[np.ndarray]] = {}
    wanted = {code.upper() for code in codes}
    for line in pdb_text.splitlines():
        if not line.startswith("HETATM"):
            continue
        name, residue, element, xyz, _ = _atom(line)
        if name in wanted and element not in {"H", "D"}:
            groups.setdefault((name, residue), []).append(xyz)
    return [np.array(atoms) for atoms in groups.values()]


def _min_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)))


def ligands(pdb_text: str, codes: tuple[str, ...]) -> list[np.ndarray]:
    """The relevant ligands of one structure, as the benchmark counts them.

    Covalently joined residues (an oligosaccharide, a modified peptide) form one ligand, and a
    ligand counts only if it touches this protein chain: COACH420 files hold one chain, and a
    ligand bound to a chain that was removed is no binding site of this one.
    """
    residues = ligand_residues(pdb_text, codes)
    parent = list(range(len(residues)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(residues)):
        for j in range(i + 1, len(residues)):
            if _min_distance(residues[i], residues[j]) <= BOND_DISTANCE:
                parent[root(i)] = root(j)
    clusters: dict[int, list[np.ndarray]] = {}
    for i, atoms in enumerate(residues):
        clusters.setdefault(root(i), []).append(atoms)
    joined = [np.vstack(parts) for parts in clusters.values()]

    protein = np.array([_atom(line)[3] for line in pdb_text.splitlines() if line.startswith("ATOM")])
    if not len(protein):
        return joined
    return [lig for lig in joined if _min_distance(lig, protein) <= CONTACT_DISTANCE]


def fpocket_centers(out_pdb: str) -> list[np.ndarray]:
    """Pocket centres from fpocket's *_out.pdb: mean of each pocket's sphere centres (STP)."""
    pockets: dict[int, list[np.ndarray]] = {}
    for line in out_pdb.splitlines():
        if line.startswith("HETATM") and line[17:20].strip() == "STP":
            pockets.setdefault(int(line[22:26]), []).append(_atom(line)[3])
    return [np.mean(pockets[number], axis=0) for number in sorted(pockets)]


def p2rank_centers(predictions_csv: str) -> list[np.ndarray]:
    reader = csv.DictReader(io.StringIO(predictions_csv), skipinitialspace=True)
    rows = [{key.strip(): value.strip() for key, value in row.items() if key} for row in reader]
    rows.sort(key=lambda row: int(row["rank"]))
    return [np.array([float(row["center_x"]), float(row["center_y"]), float(row["center_z"])]) for row in rows]


# --- scoring -------------------------------------------------------------------------


def found(ligand: np.ndarray, centers: list[np.ndarray], cutoff: float = DCA_CUTOFF) -> bool:
    return any(float(np.min(np.linalg.norm(ligand - c, axis=1))) <= cutoff for c in centers)


def score_case(ligs: list[np.ndarray], centers: list[np.ndarray]) -> tuple[int, int, int]:
    """Ligands found within the Top-n, the Top-(n+2) and any of the pockets.

    The last count ignores the ranking: it is the ceiling set by pocket detection alone, so
    a gap between it and Top-n is a ranking problem, and a low value a detection problem.
    """
    n = len(ligs)
    return (sum(found(lig, centers[:n]) for lig in ligs),
            sum(found(lig, centers[:n + 2]) for lig in ligs),
            sum(found(lig, centers) for lig in ligs))


def proteinscope_centers(pdb_path: str) -> dict[str, list[list[float]]]:
    """Pocket centres as ranked by the fitted score and, for comparison, by the heuristic."""
    from proteinscope.pipeline import run_analysis

    result = run_analysis(pdb_path, probe_pockets=0, context=False)
    fitted = [p.center.tolist() for p in result.pockets]
    heuristic = [p.center.tolist() for p in sorted(result.pockets, key=lambda p: -p.heuristic_score)]
    return {"fitted": fitted, "heuristic": heuristic}


# --- run -----------------------------------------------------------------------------


def run(workers: int) -> dict:
    from proteinscope import __version__
    from proteinscope.druggability import load_model

    ref = dataset_ref()
    cases = load_cases(ref)
    print(f"{len(cases)} proteins in {LIGAND_LIST} at {DATASET_REPO}@{ref[:12]}", flush=True)

    paths = sorted({p for case in cases for p in (case.protein, case.fpocket, case.p2rank) if p})
    with ThreadPoolExecutor(8) as pool:
        local = dict(zip(paths, pool.map(lambda p: fetch(p, ref), paths)))
    print(f"{len(paths)} files ready", flush=True)

    with ProcessPoolExecutor(workers) as pool:
        ours = list(pool.map(proteinscope_centers, [str(local[c.protein]) for c in cases], chunksize=4))

    training = set(load_model()["training"]["pdb_ids"])
    per_case = []
    for case, centers in zip(cases, ours):
        text = local[case.protein].read_text()
        residues = ligand_residues(text, case.codes)
        ligs = ligands(text, case.codes)
        if not ligs:
            reason = ("no atoms of the named ligands in the file" if not residues
                      else "the named ligands do not touch this chain")
            per_case.append({"protein": case.protein, "skipped": reason})
            continue
        methods = {
            "ProteinScope": [np.array(c) for c in centers["fitted"]],
            "ProteinScope (heuristic ranking)": [np.array(c) for c in centers["heuristic"]],
        }
        if case.fpocket:
            methods["fpocket"] = fpocket_centers(local[case.fpocket].read_text())
        if case.p2rank:
            methods["P2Rank"] = p2rank_centers(local[case.p2rank].read_text())
        per_case.append({
            "protein": case.protein,
            "pdb_id": case.pdb_id,
            "ligand_codes": list(case.codes),
            "ligand_residues": len(residues),
            "ligands": len(ligs),
            "in_training_set": case.pdb_id in training,
            "pockets": {name: len(c) for name, c in methods.items()},
            "found": {name: score_case(ligs, c) for name, c in methods.items()},
        })

    def summary(rows: list[dict]) -> dict:
        names = sorted({name for row in rows for name in row["found"]})
        out = {}
        for name in names:
            scored = [row for row in rows if name in row["found"]]
            total = sum(row["ligands"] for row in scored)
            out[name] = {
                "proteins": len(scored),
                "ligands": total,
                "top_n": round(100 * sum(row["found"][name][0] for row in scored) / total, 1),
                "top_n2": round(100 * sum(row["found"][name][1] for row in scored) / total, 1),
                "any_pocket": round(100 * sum(row["found"][name][2] for row in scored) / total, 1),
                "mean_pockets": round(float(np.mean([row["pockets"][name] for row in scored])), 1),
            }
        return out

    scored = [row for row in per_case if "found" in row]
    return {
        "date": date.today().isoformat(),
        "dataset": {"repository": DATASET_REPO, "commit": ref, "list": LIGAND_LIST,
                    "fpocket_list": FPOCKET_LIST, "p2rank_list": P2RANK_LIST},
        "criterion": f"DCA <= {DCA_CUTOFF} A, ligand-centric, Top-n and Top-(n+2)",
        "proteinscope": __version__,
        "pocket_model": load_model()["version"],
        "excluded_overlap": sorted({row["pdb_id"] for row in scored if row["in_training_set"]}),
        "ligand_rules": {
            "bond_distance_A": BOND_DISTANCE,
            "contact_distance_A": CONTACT_DISTANCE,
            "residues_named": sum(row["ligand_residues"] for row in scored),
            "ligands_counted": sum(row["ligands"] for row in scored),
        },
        "summary_disjoint": summary([row for row in scored if not row["in_training_set"]]),
        "summary_all": summary(scored),
        "published_table3": PUBLISHED,
        "skipped": [row for row in per_case if "skipped" in row],
        "cases": scored,
    }


def write_report(results: dict) -> None:
    disjoint, everything = results["summary_disjoint"], results["summary_all"]
    order = ["P2Rank", "ProteinScope", "ProteinScope (heuristic ranking)", "fpocket"]

    def table(summary: dict) -> str:
        lines = ["| Method | Top-n | Top-(n+2) | Any pocket | Pockets per protein | Proteins | Ligands |",
                 "|---|---|---|---|---|---|---|"]
        for name in order:
            if name in summary:
                s = summary[name]
                lines.append(f"| {name} | {s['top_n']:.1f} % | {s['top_n2']:.1f} % | {s['any_pocket']:.1f} % | "
                             f"{s['mean_pockets']:.1f} | {s['proteins']} | {s['ligands']} |")
        return "\n".join(lines)

    ps = disjoint["ProteinScope"]
    heuristic = disjoint["ProteinScope (heuristic ranking)"]
    fp, p2 = disjoint["fpocket"], disjoint["P2Rank"]
    reading = f"""## Reading the numbers

- **ProteinScope ranks below both reference methods.** Top-n {ps['top_n']:.1f} % against {fp['top_n']:.1f} % for
  fpocket and {p2['top_n']:.1f} % for P2Rank; Top-(n+2) {ps['top_n2']:.1f} % against {fp['top_n2']:.1f} % and {p2['top_n2']:.1f} %.
- **Most of the gap is detection, not ranking.** Counting every pocket regardless of rank,
  ProteinScope reaches {ps['any_pocket']:.1f} % of the ligands, fpocket {fp['any_pocket']:.1f} % and P2Rank {p2['any_pocket']:.1f} %. ProteinScope's
  Top-(n+2) is already within {ps['any_pocket'] - ps['top_n2']:.1f} points of its own ceiling; better ranking alone cannot
  close a gap of {fp['any_pocket'] - ps['any_pocket']:.1f} points to fpocket.
- **It reports far fewer pockets:** {ps['mean_pockets']:.1f} per protein against {fp['mean_pockets']:.1f} for fpocket and {p2['mean_pockets']:.1f} for
  P2Rank. That keeps reports short but leaves sites out.
- **The fitted score helps at the top:** Top-n {ps['top_n']:.1f} % against {heuristic['top_n']:.1f} % for the hand-weighted
  heuristic on the same pockets; at Top-(n+2) the two are level ({ps['top_n2']:.1f} % and {heuristic['top_n2']:.1f} %).
- **A possible artefact, not yet tested:** DCA measures the pocket *centre*. ProteinScope merges
  neighbouring cavities into larger pockets, whose centre can lie more than {DCA_CUTOFF:g} Å from a ligand the
  pocket still covers. How much of the detection gap this explains is the next thing to measure.
"""

    published = "\n".join(
        f"| {name} | {values['top_n']:.1f} % | {values['top_n2']:.1f} % |" for name, values in PUBLISHED.items()
    )
    skipped = len(results["skipped"])
    text = f"""# External validation on COACH420

ProteinScope's pocket ranking compared with fpocket and P2Rank on a standard benchmark it was
not built on. Generated by `scripts/benchmark_coach420.py` on {results['date']} with ProteinScope
{results['proteinscope']} and pocket model `{results['pocket_model']}`. Per-protein results are in
[`docs/validation/coach420.json`](validation/coach420.json).

## Setup

- **Dataset:** COACH420, 420 single-chain protein-ligand complexes (Krivák & Hoksza 2018), in the
  `{LIGAND_LIST}` variant where the relevant ligands of each protein are named from Binding MOAD.
  {len(results['cases']) + skipped} proteins are listed; {skipped} are skipped because none of their named ligands
  is present and in contact with the chain. Files from [{results['dataset']['repository']}](https://github.com/{results['dataset']['repository']})
  at commit `{results['dataset']['commit'][:12]}`.
- **Other methods:** the fpocket and P2Rank predictions published in that repository (fpocket 1.0
  and P2Rank 2.0, default parameters, per its README). They are not re-run here; they are scored
  by the same code as ProteinScope, on the same ligands.
- **Ligands:** residues with a named code that lie within {BOND_DISTANCE:g} Å of each other (a sugar chain,
  a modified peptide) form one ligand, and a ligand counts only if it comes within
  {CONTACT_DISTANCE:g} Å of the protein chain; COACH420 files hold a single chain, and a ligand bound to
  a chain that was removed is no site of this one. {results['ligand_rules']['residues_named']} named residues become
  {results['ligand_rules']['ligands_counted']} ligands. The same rules apply to every method.
- **Criterion:** a ligand is found when the centre of one of the top-ranked pockets lies within
  {DCA_CUTOFF:g} Å of any of its heavy atoms (DCA). Top-n counts the n best pockets, n being the number
  of relevant ligands in the structure; Top-(n+2) two more. Every ligand has equal weight. "Any pocket"
  ignores the ranking: it is the ceiling that pocket detection alone allows.
- **Leakage:** {len(results['excluded_overlap'])} proteins ({', '.join(results['excluded_overlap'])}) are also in
  ProteinScope's training set. The headline table leaves them out for every method.

## Results without the training overlap

{table(disjoint)}

{reading}
## All proteins

{table(everything)}

## Published numbers, for context

Table 3 of the P2Rank paper, DCA 4 Å, on the default COACH420 list with the authors' own ligand
filter. The ligand set differs from the MOAD list used above, so these are not directly comparable.

| Method | Top-n | Top-(n+2) |
|---|---|---|
{published}

Krivák, R. & Hoksza, D. P2Rank: machine learning based tool for rapid and accurate prediction of
ligand binding sites from protein structure. *J. Cheminform.* 10, 39 (2018).

## What this does and does not show

- It measures whether the **top of the ranking** points at a known ligand site. That is what the
  report's pocket order is used for; it says nothing about how tractable a target is.
- ProteinScope's score was trained on 72 complexes chosen by hand and was not tuned on COACH420.
  P2Rank was trained on a separate set (CHEN11) by its authors.
- Ligand sites in COACH420 include cofactors and natural ligands as well as drugs, so the benchmark
  also probes the known weakness of the score on polar nucleotide and cofactor sites.
- One dataset, one criterion. HOLO4K (larger, multi-chain structures) is not run here.

## Reproduce

```bash
python scripts/benchmark_coach420.py
```

The files are downloaded into the ProteinScope cache and are not part of this repository.
"""
    REPORT_MD.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report-only", action="store_true",
                        help="rewrite docs/validation.md from the saved JSON without rerunning")
    args = parser.parse_args()
    if args.report_only:
        write_report(json.loads(RESULTS_JSON.read_text(encoding="utf-8")))
        print(f"written {REPORT_MD.relative_to(ROOT)}")
        return
    results = run(args.workers)
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(results, indent=1), encoding="utf-8")
    write_report(results)
    for label, summary in (("without overlap", results["summary_disjoint"]), ("all", results["summary_all"])):
        print(f"\n{label}:")
        for name, s in summary.items():
            print(f"  {name:34} Top-n {s['top_n']:5.1f} %   Top-(n+2) {s['top_n2']:5.1f} %   "
                  f"any {s['any_pocket']:5.1f} %   "
                  f"({s['proteins']} proteins, {s['ligands']} ligands, {s['mean_pockets']:.1f} pockets)")
    print(f"\nwritten {RESULTS_JSON.relative_to(ROOT)} and {REPORT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

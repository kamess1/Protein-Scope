"""Screen several structures and print a ranked pocket table.

Runs without an API key:

    python examples/batch_screen.py 1HSG 3ERT 2HYY --out output
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from proteinscope.pipeline import run_analysis
from proteinscope.rcsb import RCSBError
from proteinscope.report.builder import build_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", help="PDB IDs or file paths")
    parser.add_argument("--out", default=None, help="Write a full report per structure into this directory")
    parser.add_argument("--csv", default=None, help="Write the pocket table to this CSV file")
    args = parser.parse_args()

    rows = []
    for source in args.sources:
        try:
            result = run_analysis(source, probe_pockets=1)
        except (ValueError, FileNotFoundError, RCSBError) as exc:
            print(f"{source}: skipped ({exc})", file=sys.stderr)
            continue
        for pocket in result.pockets[:3]:
            ligands = [
                label for label, entries in result.ligand_pockets.items()
                if any(entry["pocket_id"] == pocket.pocket_id for entry in entries)
            ]
            rows.append({
                "structure": result.identifier,
                "resolution_A": result.audit.resolution_A,
                "quality": result.audit.quality_grade,
                "pocket": pocket.pocket_id,
                "volume_A3": round(pocket.volume),
                "apolar": round(pocket.apolar_fraction, 2),
                "score": round(pocket.druggability, 2),
                "class": pocket.druggability_class,
                "ligand": ", ".join(ligands),
            })
        if args.out:
            print(f"{result.identifier}: {build_report(result, args.out).pdf}", file=sys.stderr)

    if not rows:
        print("Nothing to report.", file=sys.stderr)
        return 1
    rows.sort(key=lambda row: -row["score"])
    widths = {key: max(len(key), *(len(str(row[key])) for row in rows)) for key in rows[0]}
    print("  ".join(key.ljust(widths[key]) for key in rows[0]))
    for row in rows:
        print("  ".join(str(value).ljust(widths[key]) for key, value in row.items()))
    if args.csv:
        with Path(args.csv).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

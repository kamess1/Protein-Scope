"""Command line interface.

`analyze` and `report` run entirely offline after the structure is downloaded; `agent` and
`ask` additionally need an Anthropic API key.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from typing import Any

from . import __version__
from .pockets import PocketParams
from .rcsb import RCSBError
from .report.builder import analysis_digest, build_report

_BOLD, _DIM, _RESET = "\033[1m", "\033[2m", "\033[0m"


def _color(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}" if sys.stdout.isatty() else text


def _params(args: argparse.Namespace) -> PocketParams:
    return PocketParams(
        grid_spacing=args.grid_spacing,
        psp_threshold=args.enclosure,
        min_volume=args.min_volume,
        max_pockets=args.max_pockets,
    )


def _print_overview(result: Any) -> None:
    audit = result.audit
    resolution = f"{audit.resolution_A:.2f} A" if audit.resolution_A else "resolution n/a"
    print(_color(f"\n{result.identifier}", _BOLD), f"- {audit.title or 'no title'}")
    print(f"  {audit.method or 'method n/a'}, {resolution}, quality {audit.quality_grade}, "
          f"{audit.residue_count} residues in {len(audit.chains)} chain(s)")
    for warning in audit.warnings:
        print(_color(f"  ! {warning}", _DIM))
    if result.pockets:
        print(_color("\n  Pockets", _BOLD))
        print(f"  {'#':>2}  {'Volume':>8}  {'Encl':>5}  {'Apolar':>6}  {'Score':>5}  Class     Ligand")
        for pocket in result.pockets:
            ligands = [
                label for label, entries in result.ligand_pockets.items()
                if any(entry["pocket_id"] == pocket.pocket_id for entry in entries)
            ]
            print(f"  {pocket.pocket_id:>2}  {pocket.volume:>7.0f}A3  {pocket.enclosure:>5.2f}  "
                  f"{pocket.apolar_fraction:>6.2f}  {pocket.druggability:>5.2f}  "
                  f"{pocket.druggability_class:<9} {', '.join(ligands) or '-'}")
    else:
        print("\n  No pockets passed the detection thresholds.")
    if result.ligand_profiles:
        print(_color("\n  Ligands", _BOLD))
        for label, profile in result.ligand_profiles.items():
            counts = ", ".join(f"{count} {kind}" for kind, count in sorted(profile.counts().items()))
            print(f"  {label}: {profile.atom_count} atoms, {profile.buried_fraction:.0%} buried; {counts}")


def _analysis(args: argparse.Namespace) -> Any:
    from .pipeline import run_analysis

    print(_color(f"Analysing {args.source} ...", _DIM), file=sys.stderr)
    return run_analysis(args.source, params=_params(args), probe_pockets=args.probe_pockets)


def command_analyze(args: argparse.Namespace) -> int:
    result = _analysis(args)
    if args.json:
        print(json.dumps(analysis_digest(result), indent=2, default=str))
    else:
        _print_overview(result)
    return 0


def command_report(args: argparse.Namespace) -> int:
    result = _analysis(args)
    if not args.quiet:
        _print_overview(result)
    paths = build_report(result, args.out)
    for warning in paths.warnings:
        print(_color(f"! {warning}", _DIM), file=sys.stderr)
    print(_color("\nWritten:", _BOLD))
    for label, path in [
        ("PDF", paths.pdf), ("Markdown", paths.markdown), ("Data", paths.data), ("Viewer", paths.viewer),
    ]:
        if path:
            print(f"  {label:<9} {path}")
    if args.open and paths.viewer:
        webbrowser.open(paths.viewer.resolve().as_uri())
    return 0


def _agent_listener(verbose: bool) -> Any:
    from .agents.runner import Event

    def listen(event: Event) -> None:
        agent = event.data.get("agent", "?")
        if event.kind == "tool":
            print(_color(f"  [{agent}] {event.text}", _DIM), file=sys.stderr)
        elif event.kind == "status" and verbose:
            print(_color(f"  [{agent}] {event.text.strip()[:160]}", _DIM), file=sys.stderr)
        elif event.kind == "text" and verbose:
            print(_color(f"  [{agent}] {event.text.strip()[:400]}", _DIM), file=sys.stderr)

    return listen


def _run_crew(request: str, out: str, verbose: bool) -> int:
    import anthropic

    from .agents.crew import Crew
    from .agents.runner import AgentError

    missing_credentials = (
        "No API credentials found. Set ANTHROPIC_API_KEY, or sign in with `ant auth login`."
    )
    try:
        crew = Crew(out_dir=out, listener=_agent_listener(verbose))
        result = crew.run(request)
    except TypeError as exc:  # The SDK raises TypeError when it cannot resolve credentials.
        if "authentication" not in str(exc):
            raise
        print(missing_credentials, file=sys.stderr)
        return 2
    except anthropic.AuthenticationError:
        print("The API rejected these credentials. Check ANTHROPIC_API_KEY.", file=sys.stderr)
        return 2
    except AgentError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    except anthropic.APIStatusError as exc:
        print(f"\nThe API returned {exc.status_code}: {exc.message}", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError as exc:
        print(f"\nCould not reach the API: {exc}", file=sys.stderr)
        return 1
    print("\n" + result.answer)
    usage = result.usage
    print(_color(
        f"\n[{usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out tokens, "
        f"{usage.get('cache_read_input_tokens', 0)} cached]", _DIM,
    ), file=sys.stderr)
    return 0


def command_agent(args: argparse.Namespace) -> int:
    request = f"Analyse the structure {args.source} and write the full report."
    if args.focus:
        request += f" Pay particular attention to: {args.focus}"
    return _run_crew(request, args.out, args.verbose)


def command_ask(args: argparse.Namespace) -> int:
    return _run_crew(" ".join(args.question), args.out, args.verbose)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="proteinscope",
        description="Protein structure, binding pocket and ligand analysis with Claude agents.",
    )
    parser.add_argument("--version", action="version", version=f"ProteinScope {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_analysis_options(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("source", help="PDB ID (e.g. 1HSG) or path to a .pdb/.cif file")
        sub.add_argument("--grid-spacing", type=float, default=1.0, metavar="A",
                         help="Pocket grid spacing in Angstrom (default 1.0; smaller is slower)")
        sub.add_argument("--enclosure", type=int, default=4, choices=range(1, 8), metavar="1-7",
                         help="Scan lines that must be buried for a pocket point (default 4)")
        sub.add_argument("--min-volume", type=float, default=60.0, metavar="A3",
                         help="Smallest pocket to keep (default 60)")
        sub.add_argument("--max-pockets", type=int, default=12, help="Pockets to keep (default 12)")
        sub.add_argument("--probe-pockets", type=int, default=3,
                         help="Top pockets to run the probe simulation on (default 3)")

    analyze = subparsers.add_parser("analyze", help="Analyse a structure and print the results")
    add_analysis_options(analyze)
    analyze.add_argument("--json", action="store_true", help="Print the full digest as JSON")
    analyze.set_defaults(func=command_analyze)

    report = subparsers.add_parser("report", help="Analyse and write report files, without an LLM")
    add_analysis_options(report)
    report.add_argument("-o", "--out", default="output", help="Output directory (default ./output)")
    report.add_argument("--open", action="store_true", help="Open the 3D viewer in a browser")
    report.add_argument("-q", "--quiet", action="store_true", help="Only print the written paths")
    report.set_defaults(func=command_report)

    agent = subparsers.add_parser("agent", help="Run the agent crew on one structure (needs an API key)")
    agent.add_argument("source", help="PDB ID or file path")
    agent.add_argument("--focus", default="", help="What the analysis should pay attention to")
    agent.add_argument("-o", "--out", default="output", help="Output directory (default ./output)")
    agent.add_argument("-v", "--verbose", action="store_true", help="Show agent reasoning and replies")
    agent.set_defaults(func=command_agent)

    ask = subparsers.add_parser("ask", help="Ask the crew a question in plain language (needs an API key)")
    ask.add_argument("question", nargs="+", help="e.g. 'find a KRAS G12C structure and analyse its pocket'")
    ask.add_argument("-o", "--out", default="output", help="Output directory (default ./output)")
    ask.add_argument("-v", "--verbose", action="store_true", help="Show agent reasoning and replies")
    ask.set_defaults(func=command_ask)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError, RCSBError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

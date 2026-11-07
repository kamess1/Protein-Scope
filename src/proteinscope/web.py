"""A local overview of the reports in the output directory, served in a browser.

`proteinscope serve` starts a small HTTP server on the loopback interface. Its index page
lists every analysis that has been written, and every other path is served straight from the
output directory, so the existing `report.html` (with its figures, viewer and downloads) works
unchanged. Nothing is analysed here: the page is a reader for what `proteinscope report` wrote.

The index is rendered per request, so a report written while the server runs shows up on
reload. Only the standard library is used, and the socket is bound to 127.0.0.1.
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from html import escape
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import __version__
from .report.html import CSS, THEME_JS

HOST = "127.0.0.1"  # Reports may contain unpublished work: never bind to every interface.


@dataclass
class Entry:
    """One analysed structure, as the overview needs it."""

    identifier: str
    directory: Path
    title: str
    written: datetime
    is_model: bool
    method: str | None
    resolution_A: float | None
    mean_plddt: float | None
    quality_grade: str | None
    residue_count: int
    ligands: list[str]
    pocket_count: int
    top_score: float | None
    top_class: str | None
    top_function: str | None
    genes: list[str]
    approved_drugs: int | None
    thumbnail: str | None
    files: dict[str, str]

    @property
    def href(self) -> str:
        return f"{self.directory.name}/report.html"


def _top_pocket(data: dict[str, Any]) -> dict[str, Any] | None:
    pockets = data.get("pockets") or []
    return pockets[0] if pockets else None


def read_entry(directory: Path) -> Entry | None:
    """Read one output directory, or None if it holds no readable analysis."""
    digest = directory / "analysis.json"
    try:
        data = json.loads(digest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    audit = data.get("audit") or {}
    context = data.get("context") or {}
    annotations = context.get("annotations") or {}
    top = _top_pocket(data)
    # analysis.json stores pocket ids as integers; JSON object keys arrive as strings.
    pocket_annotations = annotations.get("pockets") or {}
    top_row = pocket_annotations.get(str(top["pocket_id"])) if top else None

    files = {
        label: name for label, name in
        [("HTML", "report.html"), ("PDF", "report.pdf"), ("Markdown", "report.md"),
         ("JSON", "analysis.json"), ("3D viewer", "viewer.html")]
        if (directory / name).is_file()
    }
    overview = directory / "figures" / "overview.png"
    return Entry(
        identifier=data.get("identifier") or directory.name,
        directory=directory,
        title=audit.get("title") or "",
        written=datetime.fromtimestamp(digest.stat().st_mtime),
        is_model=(data.get("metadata") or {}).get("source") == "alphafold",
        method=audit.get("method"),
        resolution_A=audit.get("resolution_A"),
        mean_plddt=audit.get("mean_plddt"),
        quality_grade=audit.get("quality_grade"),
        residue_count=audit.get("residue_count") or 0,
        ligands=audit.get("ligands") or [],
        pocket_count=len(data.get("pockets") or []),
        top_score=top.get("druggability") if top else None,
        top_class=top.get("druggability_class") if top else None,
        top_function=_function_phrase(top_row),
        genes=[g for p in annotations.get("proteins") or [] for g in p.get("genes") or []],
        approved_drugs=sum(row.get("approved_drugs") or 0 for row in context.get("targets") or [])
        if context.get("targets") else None,
        thumbnail=f"{directory.name}/figures/overview.png" if overview.is_file() else None,
        files=files,
    )


def _function_phrase(row: dict[str, Any] | None) -> str | None:
    """The annotation of the top pocket, short enough for a card."""
    if not row:
        return None
    from .enrich import annotation_summary

    summary = annotation_summary(row)
    if summary == "-":
        return None
    parts = summary.split("; ")  # A whole annotation list would dwarf the card.
    return "; ".join(parts[:3]) + (f" (+{len(parts) - 3})" if len(parts) > 3 else "")


def collect_entries(out_dir: Path) -> list[Entry]:
    """Every analysed structure under `out_dir`, most recently written first."""
    entries = [
        entry for child in sorted(out_dir.iterdir()) if child.is_dir()
        for entry in [read_entry(child)] if entry is not None
    ] if out_dir.is_dir() else []
    return sorted(entries, key=lambda item: item.written, reverse=True)


def _e(value: Any) -> str:
    return escape(str(value), quote=True)


def _tone(entry: Entry) -> str:
    return {"high": "good", "moderate": "warn", "low": "muted"}.get(entry.top_class or "", "muted")


def _card(entry: Entry) -> str:
    thumb = (
        f'<a class="shot" href="{_e(entry.href)}"><img loading="lazy" src="{_e(entry.thumbnail)}" '
        f'alt="Overview of {_e(entry.identifier)}"></a>' if entry.thumbnail else ""
    )
    if entry.is_model:
        quality = f"predicted model · pLDDT {entry.mean_plddt:.0f}" if entry.mean_plddt else "predicted model"
    elif entry.resolution_A:
        quality = f"{entry.method or 'experimental'} · {entry.resolution_A:.2f} Å"
    else:
        quality = entry.method or "unknown method"

    facts = [
        f"{entry.residue_count} residues",
        f"{entry.pocket_count} pocket{'s' if entry.pocket_count != 1 else ''}",
    ]
    if entry.ligands:
        facts.append(f"ligands {', '.join(entry.ligands[:4])}")
    if entry.approved_drugs:
        facts.append(f"{entry.approved_drugs} approved drugs on the target")

    score = (
        f'<span class="badge tone-{_tone(entry)}">top pocket {entry.top_score:.2f} '
        f'{_e(entry.top_class or "")}</span>' if entry.top_score is not None else
        '<span class="badge tone-muted">no pocket</span>'
    )
    genes = f' <span class="chip">{_e(", ".join(sorted(set(entry.genes))[:3]))}</span>' if entry.genes else ""
    function = (f'<p class="muted">Top pocket: {_e(entry.top_function)}</p>' if entry.top_function else "")
    links = " ".join(
        f'<a href="{_e(entry.directory.name)}/{_e(name)}">{_e(label)}</a>'
        for label, name in entry.files.items() if name != "report.html"
    )
    return (
        f'<article class="card entry">{thumb}<div class="body">'
        f'<h3><a href="{_e(entry.href)}">{_e(entry.identifier)}</a>{genes}</h3>'
        f'<p class="subtitle">{_e(entry.title)}</p>'
        f'<p>{score} <span class="muted">{_e(quality)}</span></p>'
        f'<p class="muted">{_e(" · ".join(facts))}</p>'
        f'{function}'
        f'<p class="meta">Written {entry.written.strftime("%Y-%m-%d %H:%M")} · {links}</p>'
        "</div></article>"
    )


_INDEX_CSS = """
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(330px,1fr))}
.entry{display:flex;flex-direction:column;overflow:hidden;padding:0}
.entry .body{padding:14px 16px 16px;display:flex;flex-direction:column;gap:6px}
.entry .body p{margin:0}
.entry h3{margin:0}
.entry h3 a{text-decoration:none}
.entry .subtitle{font-size:13.5px;color:var(--muted);line-height:1.4}
.entry .meta{font-size:12px;color:var(--muted);margin-top:4px}
.entry .meta a{margin-right:8px}
.shot{display:block;background:var(--chip);line-height:0}
.shot img{width:100%;height:168px;object-fit:cover}
.empty{text-align:center;padding:48px 16px}
.empty code{background:var(--chip);padding:2px 7px;border-radius:5px}
"""


def render_index(entries: list[Entry], out_dir: Path) -> str:
    if entries:
        body = f'<div class="grid">{"".join(_card(entry) for entry in entries)}</div>'
        count = f"{len(entries)} analysed structure{'s' if len(entries) != 1 else ''}"
    else:
        body = (
            '<div class="card empty"><h3>No reports yet</h3>'
            '<p class="muted">Write one, then reload this page:</p>'
            "<p><code>proteinscope report 1HSG</code></p></div>"
        )
        count = "nothing analysed yet"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ProteinScope Reports</title>
<style>{CSS}{_INDEX_CSS}</style>
</head>
<body>
<header class="top"><div class="wrap">
  <h1>ProteinScope</h1>
  <p class="subtitle">{_e(count)} in {_e(out_dir)}</p>
  <div class="meta"><button id="theme" type="button">Theme: auto</button>
    <span class="byline">ProteinScope {_e(__version__)} · reload after writing a new report</span></div>
</div></header>
<main class="wrap">{body}</main>
<footer class="wrap">Served from your own machine. Every number in these reports is computed from the
coordinates by ProteinScope.</footer>
<script>{THEME_JS}</script>
</body>
</html>
"""


class _Handler(SimpleHTTPRequestHandler):
    """Serves the generated index at /, and the output directory everywhere else."""

    def do_GET(self) -> None:  # noqa: N802 - the base class defines the name.
        if self.path.split("?")[0] in ("/", "/index.html"):
            page = render_index(collect_entries(Path(self.directory)), Path(self.directory))
            body = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")  # The index changes as reports appear.
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, format: str, *args: Any) -> None:
        return  # One line per figure would bury the startup message.


def build_server(out_dir: Path, port: int = 8765) -> ThreadingHTTPServer:
    """An HTTP server on the loopback interface. Port 0 picks a free port."""
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    handler = partial(_Handler, directory=str(out_dir))
    try:
        return ThreadingHTTPServer((HOST, port), handler)
    except OSError as exc:
        raise OSError(f"Cannot serve on {HOST}:{port} ({exc}). Try --port 0 for any free port.") from exc


def serve(out_dir: Path, port: int = 8765, open_browser: bool = False) -> None:
    server = build_server(out_dir, port)
    url = f"http://{HOST}:{server.server_address[1]}/"
    print(f"ProteinScope reports: {url}")
    print(f"Serving {Path(out_dir).expanduser().resolve()}. Press Ctrl+C to stop.")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


def free_port() -> int:
    """An unused port, for tests and for callers that need the number in advance."""
    with socket.socket() as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])

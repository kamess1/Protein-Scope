"""Pick a protein in the browser and get its report: `proteinscope serve`.

The index page offers a catalogue of well-known drug targets, a field for any PDB or
AlphaFold ID, and every report already written. Analyses started there run one at a time in
a background thread (`proteinscope.jobs`) while the page shows their progress. Every other
path is served straight from the output directory, so `report.html` works unchanged.

Because the server can now act, not only read, it guards against other web pages in the
same browser, which can send requests to 127.0.0.1 too:

* it binds to the loopback interface only and answers only to Host 127.0.0.1 or localhost
  on its own port, which defeats DNS rebinding;
* starting an analysis needs a random token that exists only in the server's own page,
  a JSON body and a same-origin request, so another site can neither forge nor read it;
* the browser may submit PDB and AlphaFold IDs only, never a file path, and the pipeline is
  told not to read local files for these requests;
* errors shown on the page carry no traceback and no local path;
* nothing outside the output directory is served (symlinks included), directories are not
  listed, and the page never prints the full path of the output directory.

`--read-only` turns the analysis endpoint off and leaves a pure report browser.
Only the standard library is used.
"""

from __future__ import annotations

import hmac
import json
import secrets
import socket
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from html import escape
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import __version__
from .catalog import CatalogEntry, load_catalog, validate_identifier
from .jobs import JobQueue, QueueFull, Runner
from .report.html import CSS, THEME_JS

HOST = "127.0.0.1"  # Reports may contain unpublished work: never bind to every interface.
MAX_BODY = 2048  # An analysis request is one short identifier.
TOKEN_HEADER = "X-ProteinScope-Token"


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


def display_path(path: Path) -> str:
    """A path safe to show on screen: never the home directory, never a full absolute path.

    Relative to the working directory when it sits inside it, otherwise the folder name alone.
    Absolute paths carry the account name and the layout of someone's disk, and the page may
    be screenshotted or shared.
    """
    path = Path(path).expanduser().resolve()
    try:
        relative = path.relative_to(Path.cwd())
    except ValueError:
        return path.name or str(path)
    return f"./{relative}" if relative.parts else "."


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
.empty{text-align:center;padding:36px 16px}
.empty code,.hint code{background:var(--chip);padding:2px 7px;border-radius:5px}
.lookup{display:flex;gap:8px;flex-wrap:wrap;max-width:640px}
.lookup input{flex:1;min-width:220px;font:inherit;padding:8px 12px;border:1px solid var(--line);border-radius:8px;
background:var(--card);color:var(--ink)}
.lookup input:focus{outline:2px solid var(--accent);outline-offset:1px}
button.go{font:inherit;font-size:14px;padding:8px 16px;border-radius:8px;border:1px solid var(--accent);
background:var(--accent);color:#fff;cursor:pointer}
button.go:hover{filter:brightness(1.08)}
button.small{font:inherit;font-size:13px;padding:5px 12px;border-radius:7px;border:1px solid var(--line);
background:var(--card);color:var(--ink);cursor:pointer}
button.small:hover{border-color:var(--accent);color:var(--accent)}
.hint{font-size:13px;color:var(--muted);margin:8px 0 0}
.error{font-size:13px;color:var(--bad);margin:8px 0 0;min-height:1em}
.filters{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 14px}
.filters button{font:inherit;font-size:13px;padding:4px 12px;border-radius:999px;border:1px solid var(--line);
background:var(--card);color:var(--muted);cursor:pointer}
.filters button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
.catalog{display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(250px,1fr))}
.target{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;
display:flex;flex-direction:column;gap:6px}
.target h3{margin:0;font-size:15.5px}
.target h3 .muted{font-weight:400;font-size:13px}
.target p{margin:0;font-size:13.5px;line-height:1.45}
.target .actions{margin-top:auto;padding-top:4px;display:flex;gap:8px;align-items:center}
.state{align-self:flex-start;font-size:11.5px;padding:2px 9px;border-radius:999px;background:var(--chip);color:var(--muted)}
.state.ready{background:var(--good-bg);color:var(--good)}
.state.running,.state.queued{background:var(--warn-bg);color:var(--warn)}
.state.failed{background:var(--bad-bg);color:var(--bad)}
.progress{height:5px;border-radius:3px;background:var(--chip);overflow:hidden}
.progress i{display:block;height:100%;background:var(--warn);transition:width .4s}
.stage{font-size:12.5px;color:var(--muted)}
.jobs{display:flex;flex-direction:column;gap:8px;margin-top:14px;max-width:640px}
.job{display:grid;grid-template-columns:110px 1fr auto;gap:10px;align-items:center;background:var(--card);
border:1px solid var(--line);border-radius:8px;padding:8px 12px;font-size:13.5px}
.job b{font-weight:600}
.notice{display:flex;gap:10px;align-items:center;margin-top:12px;font-size:14px}
[hidden]{display:none!important}
"""


def _target_card(entry: CatalogEntry, has_report: bool, job: dict[str, Any] | None, read_only: bool) -> str:
    """One catalogue card. The script updates state, progress and actions in place."""
    state, label = ("ready", "Report ready") if has_report else ("new", "Not analysed yet")
    if job and job["state"] in ("queued", "running"):
        state, label = job["state"], "Waiting" if job["state"] == "queued" else "Analysing"
    elif job and job["state"] == "failed":
        state, label = "failed", "Failed"
    running = state in ("queued", "running")
    fraction = float(job["fraction"]) if job and running else 0.0
    report = f"{entry.id}/report.html"
    if read_only and not has_report:
        action = f'<span class="hint">Run <code>proteinscope report {_e(entry.id)}</code></span>'
    else:
        action = (
            f'<a class="open" href="{_e(report)}"{"" if has_report else " hidden"}>Open report</a>'
            + ("" if read_only else
               f'<button class="small" type="button" data-analyse="{_e(entry.id)}"'
               f'{" hidden" if has_report or running else ""}>Analyse</button>')
        )
    error = job["error"] if job and job["state"] == "failed" else ""
    return (
        f'<article class="target" data-id="{_e(entry.id)}" data-area="{_e(entry.area)}">'
        f'<span class="state {state}">{_e(label)}</span>'
        f'<h3>{_e(entry.name)} <span class="muted">{_e(entry.id)}</span></h3>'
        f"<p>{_e(entry.why)}</p>"
        f'<p class="muted">{_e(entry.gene)} · ligand {_e(entry.ligand)} · {_e(entry.area)}</p>'
        f'<div class="progress"{"" if running else " hidden"}><i style="width:{fraction * 100:.0f}%"></i></div>'
        f'<p class="stage"{"" if running or error else " hidden"}>{_e(error or (job or {}).get("stage", ""))}</p>'
        f'<div class="actions">{action}</div>'
        "</article>"
    )


def render_index(
    entries: list[Entry],
    out_dir: Path,
    token: str = "",
    jobs: list[dict[str, Any]] | None = None,
    read_only: bool = True,
) -> str:
    areas, catalog = load_catalog()
    analysed = {entry.identifier for entry in entries}
    by_id = {str(job["id"]): job for job in jobs or []}

    count = f"{len(entries)} report{'s' if len(entries) != 1 else ''} · {len(catalog)} targets in the catalogue"
    lookup = "" if read_only else """
<section id="analyse"><h2>Analyse a structure</h2>
<form class="lookup" id="lookup" autocomplete="off">
  <input id="identifier" name="identifier" maxlength="24" spellcheck="false"
    placeholder="PDB or AlphaFold ID, such as 6OIM or AF-P04637" aria-label="PDB or AlphaFold ID">
  <button class="go" type="submit">Analyse</button>
</form>
<p class="error" id="lookup-error" role="alert"></p>
<p class="hint">Downloads the public structure, analyses it here and writes the report. The first run of a
structure takes about half a minute; one analysis runs at a time.</p>
<div class="jobs" id="jobs" aria-live="polite"></div>
<div class="notice" id="fresh" hidden><span>New reports are ready.</span>
  <button class="small" type="button" id="reload">Show them</button></div>
</section>"""

    filters = "".join(
        f'<button type="button" data-area="{_e(area)}" aria-pressed="false">{_e(area)}</button>'
        for area in areas if any(e.area == area for e in catalog)
    )
    cards = "".join(
        _target_card(entry, entry.id in analysed, by_id.get(entry.id), read_only) for entry in catalog
    )
    catalog_html = f"""
<section id="catalogue"><h2>Catalogue</h2>
<p class="hint" style="margin:0 0 12px">Well-known drug targets, one structure each. Pick one to get its report.</p>
<div class="filters" role="group" aria-label="Filter by area">
  <button type="button" data-area="" aria-pressed="true">All</button>{filters}</div>
<div class="catalog">{cards}</div>
</section>"""

    if entries:
        reports = f'<div class="grid">{"".join(_card(entry) for entry in entries)}</div>'
    else:
        reports = (
            '<div class="card empty"><h3>No reports yet</h3>'
            '<p class="muted">Pick a target from the catalogue'
            + ("" if read_only else " or enter an ID above")
            + ", or run <code>proteinscope report 1HSG</code>.</p></div>"
        )
    reports_html = f'<section id="reports"><h2>Your reports</h2>{reports}</section>'

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="ps-token" content="{_e(token)}">
<title>ProteinScope Reports</title>
<style>{CSS}{_INDEX_CSS}</style>
</head>
<body>
<header class="top"><div class="wrap">
  <h1>ProteinScope</h1>
  <p class="subtitle">Pick a protein to see where a drug could bind, how suitable that site is and what is
  already known about it. {_e(count)} in {_e(display_path(out_dir))}.</p>
  <div class="meta"><button id="theme" type="button">Theme: auto</button>
    <span class="byline">ProteinScope {_e(__version__)}{" · read-only" if read_only else ""}</span></div>
</div></header>
<main class="wrap">{lookup}{catalog_html}{reports_html}</main>
<footer class="wrap">Served from your own machine to your own browser. Every number in these reports is computed
from the coordinates by ProteinScope.</footer>
<script>{THEME_JS}</script>
<script>{_INDEX_JS}</script>
</body>
</html>
"""


# All server data reaches the page through textContent, never innerHTML.
_INDEX_JS = """
(function(){
  const token=document.querySelector('meta[name="ps-token"]').content;
  const filters=[...document.querySelectorAll('.filters button')];
  filters.forEach(b=>b.addEventListener('click',()=>{
    filters.forEach(o=>o.setAttribute('aria-pressed',o===b));
    document.querySelectorAll('.target').forEach(c=>{c.hidden=!!b.dataset.area&&c.dataset.area!==b.dataset.area;});
  }));
  if(!token)return;

  const form=document.getElementById('lookup'),input=document.getElementById('identifier');
  const error=document.getElementById('lookup-error'),list=document.getElementById('jobs');
  const fresh=document.getElementById('fresh');
  const seen=new Map();let timer=null;

  async function api(path,body){
    const options={headers:{'X-ProteinScope-Token':token},cache:'no-store'};
    if(body){options.method='POST';options.headers['Content-Type']='application/json';options.body=JSON.stringify(body);}
    const response=await fetch(path,options);
    let data={};try{data=await response.json();}catch(e){}
    if(!response.ok)throw new Error(data.error||('The server answered '+response.status+'.'));
    return data;
  }

  async function analyse(id){
    error.textContent='';
    try{const data=await api('/api/analyze',{id:id});render([data.job]);poll();return true;}
    catch(e){error.textContent=e.message;return false;}
  }

  function label(job){return {queued:'Waiting',running:'Analysing',done:'Report ready',failed:'Failed'}[job.state];}

  function render(jobs){
    let finished=false;
    jobs.forEach(job=>{
      const before=seen.get(job.id);
      if(before&&before!=='done'&&job.state==='done')finished=true;
      seen.set(job.id,job.state);
      const card=document.querySelector('.target[data-id="'+CSS.escape(job.id)+'"]');
      if(card){
        const state=card.querySelector('.state');
        state.className='state '+(job.state==='done'?'ready':job.state);state.textContent=label(job);
        const running=job.state==='queued'||job.state==='running';
        const bar=card.querySelector('.progress');bar.hidden=!running;
        bar.firstElementChild.style.width=Math.round(job.fraction*100)+'%';
        const stage=card.querySelector('.stage');
        stage.hidden=!(running||job.error);stage.textContent=job.error||job.stage;
        const open=card.querySelector('.open'),button=card.querySelector('[data-analyse]');
        if(open)open.hidden=job.state!=='done';
        if(button){button.hidden=running||job.state==='done';button.textContent=job.state==='failed'?'Try again':'Analyse';}
      }
      let row=list.querySelector('[data-job="'+CSS.escape(job.id)+'"]');
      if(!row){
        row=document.createElement('div');row.className='job';row.dataset.job=job.id;
        row.append(document.createElement('b'),document.createElement('div'),document.createElement('span'));
        row.children[1].className='stage';list.append(row);
      }
      row.children[0].textContent=job.id;
      row.children[1].textContent=job.error||job.stage;
      const end=row.children[2];end.textContent='';
      if(job.state==='done'){const a=document.createElement('a');a.href=job.report;a.textContent='Open report';end.append(a);}
      else end.textContent=label(job)+(job.state==='running'?' '+Math.round(job.fraction*100)+'%':'');
    });
    if(finished)fresh.hidden=false;
    return jobs.some(j=>j.state==='queued'||j.state==='running');
  }

  async function tick(){
    timer=null;
    try{const data=await api('/api/jobs');if(render(data.jobs))timer=setTimeout(tick,1000);}
    catch(e){timer=setTimeout(tick,3000);}
  }
  function poll(){if(!timer)timer=setTimeout(tick,400);}

  form.addEventListener('submit',async e=>{
    e.preventDefault();
    const id=input.value.trim();
    if(!id){error.textContent='Enter a PDB ID such as 1HSG or an AlphaFold ID such as AF-P00533.';input.focus();return;}
    if(await analyse(id))input.value='';
  });
  input.addEventListener('input',()=>{error.textContent='';});
  document.querySelectorAll('[data-analyse]').forEach(b=>b.addEventListener('click',()=>analyse(b.dataset.analyse)));
  document.getElementById('reload').addEventListener('click',()=>location.reload());
  tick();
})();
"""


class ReportServer(ThreadingHTTPServer):
    """The HTTP server plus what every request needs: token, jobs and allowed hosts."""

    daemon_threads = True

    def __init__(self, out_dir: Path, port: int, read_only: bool, runner: Runner | None) -> None:
        self.out_dir = out_dir
        self.read_only = read_only
        self.token = secrets.token_urlsafe(32)  # New per start; lives only in the server's page.
        super().__init__((HOST, port), partial(_Handler, directory=str(out_dir)))
        bound = self.server_address[1]
        self.allowed_hosts = {f"{HOST}:{bound}", f"localhost:{bound}"}
        self.allowed_origins = {f"http://{host}" for host in self.allowed_hosts}
        self.jobs = None if read_only else JobQueue(out_dir, runner)

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.server_address[1]}/"


class _Handler(SimpleHTTPRequestHandler):
    """The index and API, plus the output directory served as static files.

    Everything outside the output directory is refused, and directories are never listed.
    """

    server: ReportServer

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        # viewer.html loads 3Dmol.js from a CDN; without this the local URL travels with it.
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "SAMEORIGIN")  # No other site may frame the controls.
        super().end_headers()

    # -- request gates ------------------------------------------------------------------

    def _host_allowed(self) -> bool:
        """Only our own address. A rebinding DNS name pointing at 127.0.0.1 is refused."""
        if self.headers.get("Host", "") in self.server.allowed_hosts:
            return True
        self._json(421, {"error": "This server answers only to its own address."})
        return False

    def _token_valid(self) -> bool:
        supplied = self.headers.get(TOKEN_HEADER, "")
        if supplied and hmac.compare_digest(supplied.encode(), self.server.token.encode()):
            return True
        self._json(403, {"error": "Reload the page; its session with the server has ended."})
        return False

    def _same_origin(self) -> bool:
        origin = self.headers.get("Origin")
        site = self.headers.get("Sec-Fetch-Site")
        if (origin is None or origin in self.server.allowed_origins) and site in (None, "same-origin"):
            return True
        self._json(403, {"error": "Requests from other sites are refused."})
        return False

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # -- static files ---------------------------------------------------------------------

    def translate_path(self, path: str) -> str:
        # Follow symlinks here so send_head sees where the request really points.
        return str(Path(super().translate_path(path)).resolve())

    def send_head(self):  # type: ignore[no-untyped-def] - the base class is untyped.
        """Refuse anything outside the output directory before opening it.

        The base class normalises `..` but resolves no symlinks, so a link inside the
        output directory would otherwise hand out a file from anywhere on disk.
        """
        target = Path(self.translate_path(self.path))
        root = Path(self.directory).resolve()
        if target != root and root not in target.parents:
            self.send_error(404, "Not found")
            return None
        return super().send_head()

    def list_directory(self, path: str) -> None:
        # A listing would expose file names of anything that happens to sit in the directory.
        self.send_error(404, "No listing")
        return None

    # -- routes ---------------------------------------------------------------------------

    def do_HEAD(self) -> None:  # noqa: N802 - the base class defines the name.
        if self._host_allowed():
            super().do_HEAD()

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_allowed():
            return
        route = self.path.split("?")[0]
        if route in ("/", "/index.html"):
            self._index()
        elif route == "/api/jobs":
            if self.server.jobs is None:
                self._json(200, {"jobs": []})
            elif self._token_valid():
                self._json(200, {"jobs": self.server.jobs.snapshot()})
        elif route.startswith("/api/"):
            self._json(404, {"error": "Unknown endpoint."})
        else:
            super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if not self._host_allowed():
            return
        if self.path.split("?")[0] != "/api/analyze":
            self._json(404, {"error": "Unknown endpoint."})
            return
        if self.server.jobs is None:
            self._json(405, {"error": "This server is read-only."})
            return
        if not (self._same_origin() and self._token_valid()):
            return
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            self._json(415, {"error": "Send the request as JSON."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if not 0 < length <= MAX_BODY:
            self._json(413, {"error": "The request is empty or too large."})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            identifier = validate_identifier(payload.get("id") if isinstance(payload, dict) else None)
        except (ValueError, UnicodeDecodeError) as exc:
            message = str(exc) if "PDB ID" in str(exc) else 'Send {"id": "1HSG"}.'
            self._json(400, {"error": message})
            return
        try:
            job = self.server.jobs.submit(identifier)
        except QueueFull as exc:
            self._json(429, {"error": str(exc)})
            return
        self._json(202, {"job": job.to_dict()})

    def _index(self) -> None:
        out_dir = Path(self.directory)
        jobs = self.server.jobs.snapshot() if self.server.jobs else []
        page = render_index(
            collect_entries(out_dir), out_dir, token=self.server.token, jobs=jobs,
            read_only=self.server.read_only,
        )
        body = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")  # The index changes as reports appear.
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return  # One line per figure would bury the startup message.


def build_server(
    out_dir: Path, port: int = 8765, read_only: bool = False, runner: Runner | None = None
) -> ReportServer:
    """A server on the loopback interface. Port 0 picks a free port."""
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        return ReportServer(out_dir, port, read_only, runner)
    except OSError as exc:
        raise OSError(f"Cannot serve on {HOST}:{port} ({exc}). Try --port 0 for any free port.") from exc


def serve(out_dir: Path, port: int = 8765, open_browser: bool = False, read_only: bool = False) -> None:
    server = build_server(out_dir, port, read_only)
    mode = "read-only" if read_only else "analyses enabled"
    print(f"ProteinScope: {server.url}", flush=True)
    print(f"Serving {display_path(Path(out_dir))} to this machine only ({mode}). Press Ctrl+C to stop.",
          flush=True)
    if open_browser:
        import webbrowser

        webbrowser.open(server.url)
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

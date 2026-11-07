"""Interactive and external 3D outputs: a standalone HTML page, PyMOL scripts and nglview."""

from __future__ import annotations

import html
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB import PDBIO, Select

from ..constants import MODIFIED_RESIDUES
from ..pipeline import AnalysisResult
from ..structure import residue_key

VIEWER_SCRIPT = "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"
_POCKET_COLORS = ["#e34948", "#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300", "#eb6834"]


class _AnalysedAtoms(Select):
    """Keep the first model, the protein, ligands and ions, and drop water and additives."""

    def __init__(self, result: AnalysisResult) -> None:
        structure = result.structure
        self.keep = {residue_key(r) for r in structure.protein_residues}
        self.keep.update(item.key for item in structure.ligands)
        self.keep.update(item.key for item in structure.ions)

    def accept_model(self, model: Any) -> bool:
        return model.get_id() == next(iter(model.get_parent())).get_id()

    def accept_residue(self, residue: Any) -> bool:
        return residue_key(residue) in self.keep


def structure_pdb_text(result: AnalysisResult) -> str:
    """Serialise the analysed atoms as PDB. Chain IDs must be single characters in this format."""
    writer = PDBIO()
    writer.set_structure(result.structure.biopython)
    buffer = io.StringIO()
    writer.save(buffer, select=_AnalysedAtoms(result))
    return buffer.getvalue()


def _pocket_payload(result: AnalysisResult, limit: int) -> list[dict[str, Any]]:
    payload = []
    for index, pocket in enumerate(result.pockets[:limit]):
        points = pocket.points
        if len(points) > 400:  # Thinning keeps the page responsive and hardly changes the shape.
            points = points[np.linspace(0, len(points) - 1, 400).astype(int)]
        payload.append({
            "id": pocket.pocket_id,
            "color": _POCKET_COLORS[index % len(_POCKET_COLORS)],
            "volume": round(pocket.volume),
            "score": round(pocket.druggability, 2),
            "cls": pocket.druggability_class,
            "center": pocket.center.round(2).tolist(),
            "points": points.round(2).tolist(),
            "residues": pocket.lining_residues[:14],
        })
    return payload


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --surface:#fcfcfb; --panel:#f3f2ee; --ink:#0b0b0b; --muted:#52514e; --line:#dcdbd5; --accent:#2a78d6; }
@media (prefers-color-scheme: dark) { :root { --surface:#1a1a19; --panel:#262625; --ink:#f0efec; --muted:#c3c2b7; --line:#3a3a38; --accent:#3987e5; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--surface); color:var(--ink); font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif; display:grid; grid-template-columns:320px 1fr; height:100vh; }
aside { padding:16px; overflow:auto; border-right:1px solid var(--line); }
h1 { font-size:18px; margin:0 0 2px; } h2 { font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin:18px 0 6px; }
p { margin:0; color:var(--muted); font-size:12.5px; }
#stage { position:relative; min-height:320px; }
#viewer { position:absolute; inset:0; }
.group { display:flex; flex-wrap:wrap; gap:4px; }
button { font:inherit; font-size:12.5px; padding:5px 9px; border:1px solid var(--line); background:var(--panel); color:var(--ink); border-radius:6px; cursor:pointer; }
button[aria-pressed="true"] { background:var(--accent); border-color:var(--accent); color:#fff; }
.pocket { display:block; width:100%; text-align:left; margin-bottom:6px; padding:8px 10px; }
.pocket b { display:flex; align-items:center; gap:7px; }
.swatch { width:10px; height:10px; border-radius:50%; flex:none; }
.pocket small { display:block; color:inherit; opacity:.8; margin-top:2px; }
#error { display:none; position:absolute; inset:16px; padding:16px; background:var(--panel); border-radius:8px; }
@media (max-width:760px) { body { grid-template-columns:1fr; grid-template-rows:60vh auto; height:auto; } aside { order:2; border-right:0; border-top:1px solid var(--line); } #stage { height:60vh; } }
</style>
</head>
<body>
<aside>
  <h1>__IDENTIFIER__</h1>
  <p>__SUBTITLE__</p>
  <h2>Protein</h2>
  <div class="group" id="styles">
    <button data-style="cartoon" aria-pressed="true">Cartoon</button>
    <button data-style="hydropathy" aria-pressed="false">Surface: hydropathy</button>
    <button data-style="charge" aria-pressed="false">Surface: charged groups</button>
  </div>
  <h2>Pockets</h2>
  <div class="group" style="margin-bottom:8px">
    <button id="toggle" aria-pressed="true">Show pockets</button>
    <button id="reset">Reset view</button>
  </div>
  <div id="pockets"></div>
  <h2>Ligands</h2>
  <p>__LIGANDS__</p>
</aside>
<main id="stage"><div id="viewer"></div><div id="error">The 3D library could not be loaded. This page needs internet access for 3Dmol.js.</div></main>
<script src="__SCRIPT__"></script>
<script>
const PDB = __PDB__;
const POCKETS = __POCKETS__;
const KD = {ALA:1.8,ARG:-4.5,ASN:-3.5,ASP:-3.5,CYS:2.5,GLN:-3.5,GLU:-3.5,GLY:-0.4,HIS:-3.2,ILE:4.5,LEU:3.8,LYS:-3.9,MET:1.9,PHE:2.8,PRO:-1.6,SER:-0.8,THR:-0.7,TRP:-0.9,TYR:-1.3,VAL:4.2};
const PROTEIN = {resn:Object.keys(KD).concat(__MODIFIED__)};
function mix(a,b,t){const p=(c,i)=>parseInt(c.substr(i,2),16);const v=i=>Math.round(p(a,i)+(p(b,i)-p(a,i))*t);return (v(1)<<16)|(v(3)<<8)|v(5);}
function hydropathy(atom){const k=KD[atom.resn];if(k===undefined)return 0xe8e7e3;return k>=0?mix('#e8e7e3','#d9892b',k/4.5):mix('#e8e7e3','#1f8a9e',-k/4.5);}
function charge(atom){if(['ASP','GLU'].includes(atom.resn))return 0xc8352f;if(['LYS','ARG'].includes(atom.resn))return 0x2a78d6;if(atom.resn==='HIS')return 0x9fc1ea;return 0xe8e7e3;}
if (typeof $3Dmol === 'undefined') { document.getElementById('error').style.display='block'; } else {
  const dark = matchMedia('(prefers-color-scheme: dark)').matches;
  const viewer = $3Dmol.createViewer('viewer',{backgroundColor: dark ? '#1a1a19' : '#fcfcfb'});
  viewer.addModel(PDB,'pdb');
  let surfaceId = null, showPockets = true;
  function ligands(){ viewer.setStyle({hetflag:true},{stick:{radius:.2,colorscheme:'greenCarbon'},sphere:{scale:.25,colorscheme:'greenCarbon'}}); }
  function drawPockets(){
    viewer.removeAllShapes();
    if(showPockets) POCKETS.forEach(p=>p.points.forEach(c=>viewer.addSphere({center:{x:c[0],y:c[1],z:c[2]},radius:.65,color:p.color,opacity:.55})));
    viewer.render();
  }
  function setProtein(mode){
    if(surfaceId!==null){viewer.removeSurface(surfaceId);surfaceId=null;}
    viewer.setStyle({},{});
    if(mode==='cartoon'){viewer.setStyle(PROTEIN,{cartoon:{color:'spectrum',opacity:.9}});}
    else{viewer.setStyle(PROTEIN,{cartoon:{color:'#bdbcb6',opacity:.2}});
      surfaceId=viewer.addSurface($3Dmol.SurfaceType.MS,{opacity:.88,colorfunc:mode==='hydropathy'?hydropathy:charge},PROTEIN);
    }
    ligands(); viewer.render();
  }
  document.querySelectorAll('#styles button').forEach(b=>b.onclick=()=>{document.querySelectorAll('#styles button').forEach(o=>o.setAttribute('aria-pressed',o===b));setProtein(b.dataset.style);});
  document.getElementById('toggle').onclick=e=>{showPockets=!showPockets;e.target.setAttribute('aria-pressed',showPockets);drawPockets();};
  document.getElementById('reset').onclick=()=>{viewer.zoomTo();viewer.render();};
  const list=document.getElementById('pockets');
  POCKETS.forEach(p=>{const b=document.createElement('button');b.className='pocket';
    const title=document.createElement('b');const sw=document.createElement('span');sw.className='swatch';sw.style.background=p.color;title.append(sw,`Pocket ${p.id}`);
    const line=document.createElement('small');line.textContent=`${p.volume} Å³ · score ${p.score} (${p.cls})`;
    const res=document.createElement('small');res.textContent=p.residues.join(' ');
    b.append(title,line,res);
    b.onclick=()=>focusPocket(p);
    list.append(b);});
  function focusPocket(p){
    const atoms=viewer.selectedAtoms(PROTEIN).filter(a=>Math.hypot(a.x-p.center[0],a.y-p.center[1],a.z-p.center[2])<14);
    viewer.zoomTo({serial:atoms.map(a=>a.serial)},400);
  }
  addEventListener('message',e=>{const d=e.data;if(!d||d.type!=='focusPocket')return;
    const p=POCKETS.find(x=>x.id===d.id);if(!p)return;
    if(!showPockets){showPockets=true;document.getElementById('toggle').setAttribute('aria-pressed',true);drawPockets();}
    focusPocket(p);});
  setProtein('cartoon'); drawPockets(); viewer.zoomTo(); viewer.render();
  addEventListener('resize',()=>viewer.resize());
  window.PROTEINSCOPE_READY=true;
}
</script>
</body>
</html>
"""


def write_html_viewer(result: AnalysisResult, path: Path, pocket_limit: int = 8) -> Path:
    title = result.audit.title or ""
    page = _PAGE
    ligand_text = ", ".join(html.escape(item.label) for item in result.structure.ligands) or "None bound"
    replacements = {
        "__TITLE__": f"{html.escape(result.identifier)} Pocket Viewer",
        "__IDENTIFIER__": html.escape(result.identifier),
        "__SUBTITLE__": html.escape(title[:180]),
        "__LIGANDS__": ligand_text,
        "__SCRIPT__": VIEWER_SCRIPT,
        # "</" must not appear in inline JSON, or a value could close the script element.
        "__PDB__": json.dumps(structure_pdb_text(result)).replace("</", "<\\/"),
        "__POCKETS__": json.dumps(_pocket_payload(result, pocket_limit)).replace("</", "<\\/"),
        "__MODIFIED__": json.dumps(sorted(MODIFIED_RESIDUES)),
    }
    for marker, value in replacements.items():
        page = page.replace(marker, value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path


def write_pymol_session(result: AnalysisResult, out_dir: Path, pocket_limit: int = 8) -> Path:
    """Write the structure and pocket points to files and a .pml script that loads both."""
    directory = Path(out_dir) / "pymol"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "structure.pdb").write_text(structure_pdb_text(result), encoding="utf-8")
    lines = [
        "load structure.pdb, protein", "hide everything", "show cartoon, polymer",
        "show sticks, organic", "color gray80, polymer", "util.cnc organic",
    ]
    for index, pocket in enumerate(result.pockets[:pocket_limit]):
        name = f"pocket_{pocket.pocket_id}"
        rows = [
            f"HETATM{i % 99999 + 1:5d}  C   PKT Z{pocket.pocket_id % 9999:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
            for i, (x, y, z) in enumerate(pocket.points)
        ]
        (directory / f"{name}.pdb").write_text("\n".join(rows) + "\nEND\n", encoding="utf-8")
        color = _POCKET_COLORS[index % len(_POCKET_COLORS)].replace("#", "0x")
        lines += [
            f"load {name}.pdb, {name}", f"hide everything, {name}", f"show spheres, {name}",
            f"set sphere_scale, 0.35, {name}", f"set sphere_transparency, 0.4, {name}",
            f"color {color}, {name}",
        ]
    lines += ["bg_color white", "orient polymer"]
    script = directory / "session.pml"
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script


def notebook_view(result: AnalysisResult, pocket_limit: int = 5) -> Any:
    """Return an nglview widget for Jupyter. Requires the `notebook` extra."""
    try:
        import nglview
    except ImportError as exc:
        raise ImportError('nglview is not installed. Run: pip install "proteinscope[notebook]"') from exc
    view = nglview.show_text(structure_pdb_text(result))
    view.clear_representations()
    view.add_representation("cartoon", selection="protein", color="chainindex")
    view.add_representation("ball+stick", selection="hetero and not water")
    for index, pocket in enumerate(result.pockets[:pocket_limit]):
        color = _POCKET_COLORS[index % len(_POCKET_COLORS)]
        rgb = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        for point in pocket.points[:: max(len(pocket.points) // 150, 1)]:
            view.shape.add_sphere(point.tolist(), rgb, 0.6)
    return view

# ProteinScope

Reads a protein structure (PDB entry, AlphaFold model or local file), finds its binding pockets,
profiles how ligands bind, puts the target into ChEMBL context, and writes an interactive HTML
report, a PDF and Markdown — on its own, from Claude Code over MCP (no API key), or driven by
a team of Claude agents.

Everything is computed from the coordinates by this package. The agents decide what to analyse
and write the prose; they never supply a number.

![A ProteinScope report page](docs/example-report.png)

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"        # or just -e . ; extras: mcp, chem (RDKit), notebook
```

Python 3.10 or newer. No PyMOL, APBS or fpocket needed: pockets, surfaces, electrostatics and
the 3D images are all computed and rendered in-process. Activate the venv in every new shell
(`source .venv/bin/activate`), otherwise `proteinscope` is not on the PATH.

Three ways to run it, and only the last needs an API key:

| Mode | Command | Who writes the text |
|---|---|---|
| Rule-based | `proteinscope report 1HSG` | Fixed rules |
| Claude Code over MCP | `proteinscope mcp` (registered in `.mcp.json`) | Claude in your Claude Code session |
| Built-in crew | `proteinscope agent 1HSG` | Claude via the API (`ANTHROPIC_API_KEY`) |

## Use

```bash
# Analyse and print the result. Downloads 1HSG from the PDB and caches it.
proteinscope analyze 1HSG

# The same, plus figures, a PDF, Markdown, JSON and an interactive 3D viewer.
proteinscope report 1HSG --open

# A local file works everywhere a PDB ID does, and so does an AlphaFold DB model.
proteinscope report ./my-model.cif
proteinscope report AF-P00533

# Skip the online context (ligand chemistry, ChEMBL).
proteinscope report 1HSG --offline

# Let the agents run it and write the interpretation.
proteinscope agent 1HSG --focus "whether the flap region opens a second site"

# Or state a goal and let them find the structure first.
proteinscope ask "find a high-resolution KRAS G12C structure with a covalent inhibitor and analyse its pocket"
```

From Python:

```python
from proteinscope.pipeline import run_analysis
from proteinscope.report.builder import build_report

result = run_analysis("1HSG")
print(result.pockets[0].summary())
paths = build_report(result, "output")   # paths.pdf, paths.markdown, paths.viewer, ...
```

## Claude Code as the agent (MCP, no API key)

`proteinscope mcp` serves the analysis tools over the Model Context Protocol. `.mcp.json` in this
repository registers it, so Claude Code started in this folder offers the server after you
approve it (`/mcp` shows its status). Then just ask in the chat, for example:

> Find a high-resolution KRAS G12C structure with a covalent inhibitor, analyse its pocket,
> compare it with the ChEMBL data and write the report.

Claude Code plans the steps, calls `search_pdb`, `analyze_structure`, `pocket_detail`,
`ligand_chemistry`, `target_bioactivity` and finally `write_report`, and the report says
"Written by: Claude via MCP, from the computed data". Reports land in `output/`. To register
the server outside this folder:

```bash
claude mcp add proteinscope -- "$PWD/.venv/bin/proteinscope" mcp --out "$PWD/output"
```

In a notebook, `proteinscope.viz.viewer.notebook_view(result)` returns an nglview widget
(`pip install "proteinscope[notebook]"`).

## What it computes

| Step | Method |
|---|---|
| Parsing | BioPython, mmCIF or PDB; splits protein, ligands, ions, crystallisation additives and water |
| Audit | Resolution, R factors, unobserved residues, chain breaks, incomplete side chains, clashes, B-factor outliers, assembly mismatch |
| Surface | Shrake–Rupley SASA per atom and residue, relative exposure, molecular dimensions, hydrophobic patch clustering |
| Electrostatics | Screened Coulomb with pH 7 formal charges, distance-dependent dielectric, Debye–Hückel screening |
| Pockets | LIGSITE-style grid scan on the protein alone, connected-component clustering, enclosure and depth |
| Druggability | Weighted heuristic over size, enclosure, apolarity, depth and buried charge, plus site tags (fragment-sized, shallow, polar, charged anchor) |
| Model confidence | For AlphaFold models: pLDDT from the B-factor column, per residue and per pocket |
| Ligand chemistry | CCD SMILES, RDKit descriptors (MW, cLogP, HBD/HBA, TPSA, QED), rule of five, Veber, rule of three, 2D depiction |
| Target context | UniProt mapping from RCSB; ChEMBL target, measured potencies, most potent compounds, drugs with a known mechanism |
| Ligands | Hydrogen bonds, salt bridges, hydrophobic contacts, π-stacking, halogen and metal contacts, buried surface, per-residue van der Waals energy |
| Probes | Six chemical probes scored on the pocket grid against the rigid protein |

Pocket detection is checked against known complexes: `tests/test_network.py` asserts that the
ligand of 1HSG, 2HYY, 3ERT, 1HCK and 2RH1 lands in the top-ranked pocket of its structure.

### What the numbers are not

- The **druggability score** is hand-weighted, not fitted. It ranks pockets within one
  structure; it is not a probability and does not compare across proteins. It favours large
  apolar pockets, so small charged sites rank low: trypsin's S1 pocket in 3PTB scores 0.20.
  The **site tags** flag such sites ("fragment-sized", "anionic anchor: favours cationic
  groups") instead of reweighting the score for one binding mode.
- **ChEMBL context** is matched by UniProt accession. Some proteins have their bioactivity
  under a different ChEMBL target (HIV-1 protease in 1HSG maps to a Gag-Pol entry with few
  records), so a low count is not proof that a target is undrugged.
- **Electrostatics** are a screened-Coulomb estimate, good for sign and rough size. For
  quantitative potentials use APBS.
- **Hydrogen bonds and salt bridges** come from heavy-atom geometry. Crystal structures have no
  hydrogens and ligand protonation is unknown, so these are candidates.
- **Probe energies** are rigid-grid estimates without desolvation or entropy. They compare
  probes and pockets; they are not binding free energies.
- A structure is **one conformation**. Absent loops and missing assembly partners can create or
  hide pockets, which is why the audit runs first and its warnings reach the report.

## The agents

`proteinscope agent` and `ask` start a crew of Claude Opus 5.5 roles over one shared workspace:

- **orchestrator** — plans the work and delegates; the only agent that talks to the user.
- **scout** — searches the PDB and picks entries (`search_pdb`, `entry_info`).
- **analyst** — runs and interprets the analysis (`analyze_structure`, `pocket_detail`,
  `ligand_detail`, `compare_structures`).
- **reporter** — writes the report files (`write_report`).

Each role is an `Agent` (`agents/runner.py`): a system prompt, its own tool subset, and a
streaming loop that executes tool calls until the model stops. Tool results are JSON, so the
agents reason over the same values the report prints. A report written by an agent says
"Written by: Claude, from the computed data"; one written by `proteinscope report` says
"rule-based summary".

## Output

`proteinscope report 1HSG` writes `output/1HSG/`:

```
report.html        Interactive report: summary tiles, embedded 3D viewer, sortable pocket table
                   with score bars and expandable details, ligand chemistry with 2D structures,
                   ChEMBL context, audit, figure gallery with lightbox, light/dark theme, print view
report.pdf         Typeset report with every figure and table
report.md          The same content as Markdown, figures linked
analysis.json      Every computed value, for your own scripts
viewer.html        Interactive 3D page: pockets, surfaces, ligands (needs internet for 3Dmol.js)
figures/           Overview, surface hydropathy and electrostatics, per-residue profile,
                   pocket ranking, pocket cut-aways, probe map, 2D ligand contact maps
pymol/session.pml  Loads the structure and pocket point clouds into PyMOL
```

## Development

```bash
pytest                                  # offline suite
PROTEINSCOPE_NETWORK_TESTS=1 pytest     # adds the RCSB, AlphaFold, ChEMBL and benchmark tests
```

Structures and API responses are cached in `~/.cache/proteinscope`, or `$PROTEINSCOPE_CACHE`.

To look at reports in a browser with working figures and 3D viewer, serve the folder:
`python3 -m http.server 8765 --directory output`.

## Layout

```
src/proteinscope/
  rcsb.py          Search, metadata and downloads from RCSB
  alphafold.py     AlphaFold DB identifiers, downloads and pLDDT
  webdata.py       Cached HTTP for AlphaFold DB, the CCD and ChEMBL
  chemistry.py     RDKit descriptors and 2D depictions of ligands
  targets.py       UniProt mapping and ChEMBL bioactivity
  enrich.py        Collects the context above into AnalysisResult.context
  structure.py     Parsing and classification into protein, ligands, ions, water
  audit.py         Structure quality audit
  surface.py       SASA, dimensions, hydrophobic patches
  electrostatics.py Charge model and screened-Coulomb potential
  pockets.py       Grid-based pocket detection
  druggability.py  Scoring and ranking
  interactions.py  Contact profiling and probe simulation
  pipeline.py      Runs all of the above into one AnalysisResult
  viz/             Software 3D renderer, matplotlib figures, HTML/PyMOL/nglview export
  report/          HTML, Markdown and PDF builders, rule-based narrative
  agents/          Tool layer, agent loop, crew
  mcp_server.py    The same tools over MCP for Claude Code
  cli.py           Command line interface
```

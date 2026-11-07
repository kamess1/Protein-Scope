# ProteinScope

Reads a protein structure (PDB entry, AlphaFold model or local file), finds its binding pockets,
profiles how ligands bind, puts the target into ChEMBL context, and writes an interactive HTML
report, a PDF and Markdown.

Every number is computed from the coordinates by this package. Claude decides what to analyse
and writes the prose; it never supplies a number.

![A ProteinScope report page](docs/example-report.png)

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"        # or just -e . ; extras: mcp, chem (RDKit), notebook
```

Python 3.10 or newer. No PyMOL, APBS or fpocket needed. Activate the venv in every new shell,
otherwise `proteinscope` is not on the PATH.

Three ways to run it, and only the last needs an API key:

| Mode | Command | Who writes the text |
|---|---|---|
| Rule-based | `proteinscope report 1HSG` | Fixed rules |
| Claude Code over MCP | `proteinscope mcp` (registered in `.mcp.json`) | Claude in your Claude Code session |
| Built-in crew | `proteinscope agent 1HSG` | Claude via the API (`ANTHROPIC_API_KEY`) |

## Use

```bash
proteinscope analyze 1HSG                  # analyse and print; caches the download
proteinscope report 1HSG --open            # plus figures, HTML, PDF, Markdown, JSON, 3D viewer
proteinscope serve                         # browse every written report in a browser
proteinscope report AF-P00533              # an AlphaFold model
proteinscope report ./my-model.cif         # a local file
proteinscope report 1HSG --offline         # skip ligand chemistry and ChEMBL
proteinscope ask "find a KRAS G12C structure with a covalent inhibitor and analyse its pocket"
```

From Python:

```python
from proteinscope.pipeline import run_analysis
from proteinscope.report.builder import build_report

result = run_analysis("1HSG")
paths = build_report(result, "output")   # paths.html, paths.pdf, paths.markdown, ...
```

## Browsing the reports

`proteinscope serve` opens an overview of everything in `output/` in your browser, so you pick a
structure from a list of cards instead of typing its ID. Each card shows the top pocket with its
score and function, the ligands, the quality and a thumbnail; clicking it opens the usual
interactive report, which now carries an "All reports" link back. The page is rebuilt on every
request, so a report written in another shell appears on reload.

```bash
proteinscope serve                    # http://127.0.0.1:8765, opens a browser window
proteinscope serve -o runs -p 9000 --no-open
```

It is a reader, not a launcher: analyses are still started with `proteinscope report`. It needs
no extra dependency and keeps to itself: the socket is bound to `127.0.0.1`, nothing outside the
output directory is served (symlinks out of it are refused), directories are never listed, and
responses carry `Referrer-Policy: no-referrer` so the local URL does not travel to the CDN that
`viewer.html` loads 3Dmol.js from. The page shows the output folder by name, not by full path.

## Claude Code as the agent (MCP, no API key)

`proteinscope mcp` serves the analysis tools over MCP, and `.mcp.json` registers it, so Claude
Code started in this folder offers the server after you approve it (`/mcp` shows its status).
Then just ask in the chat:

> Find a high-resolution KRAS G12C structure with a covalent inhibitor, analyse its pocket,
> compare it with the ChEMBL data and write the report.

Claude plans the steps and calls `search_pdb`, `analyze_structure`, `pocket_detail`,
`ligand_chemistry`, `functional_sites`, `target_bioactivity` and `write_report`. Reports land in
`output/`.

The paths in `.mcp.json` are relative, so the committed file names no directory of yours; the
client starts the server with the project root as its working directory. If you want the server
available outside this folder, register it in your own config instead of editing the committed
one — that writes an absolute path, but into your user config, not into the repository:

```bash
claude mcp add proteinscope -- "$PWD/.venv/bin/proteinscope" mcp --out "$PWD/output"
```

## What it computes

| Step | Method |
|---|---|
| Parsing | BioPython, mmCIF or PDB; splits protein, ligands, ions, additives and water |
| Audit | Resolution, R factors, missing residues, chain breaks, clashes, B-factor outliers |
| Surface | Shrake–Rupley SASA, relative exposure, dimensions, hydrophobic patches |
| Electrostatics | Screened Coulomb, pH 7 formal charges, distance-dependent dielectric |
| Pockets | LIGSITE-style grid scan on the protein alone, clustering, enclosure and depth |
| Pocket score | Logistic regression over eight descriptors, fitted on 72 complexes; plus site tags |
| Model confidence | For AlphaFold models: pLDDT per residue and per pocket |
| Ligand chemistry | CCD SMILES, RDKit descriptors (MW, cLogP, TPSA, QED), rule of five, 2D depiction |
| Function | UniProt active sites, cofactor sites, modifications and disease variants per pocket |
| Target context | UniProt mapping, then ChEMBL target, measured potencies and known drugs |
| Ligands | Hydrogen bonds, salt bridges, hydrophobic contacts, π-stacking, buried surface |
| Probes | Six chemical probes scored on the pocket grid against the rigid protein |

The **pocket score** is fitted, not guessed. `scripts/pocket_model.py` analyses a benchmark of
PDB entries with a ligand bound, labels each pocket by whether the ligand sits in it, and fits
a logistic regression over the pocket descriptors; the coefficients ship in
`src/proteinscope/data/ligandability.json`:

```bash
python scripts/pocket_model.py collect && python scripts/pocket_model.py fit
```

Cross-validated 5-fold, grouped by entry so pockets of one structure never span the split:
**ROC-AUC 0.93** and the top-ranked pocket correct in **90 %** of entries, against 0.91 and 83 %
for the hand-weighted heuristic it replaces (which stays in the output for comparison). The five
complexes `tests/test_network.py` checks are held out of training, so that test measures
generalisation.

### What the numbers are not

- The **pocket score** ranks the cavities of one structure. It says nothing about whether the
  target as a whole is tractable, and 72 training entries are few. Its known failure: no mostly
  polar entry is in the training set, so a high density of polar atoms counts against a pocket
  and a nucleotide site can rank below a smaller apolar one (CDK2/ATP in 1HCK does). Read it
  together with the function section.
- A small site can still be the real one. Trypsin's S1 pocket in 3PTB scores 0.10 because no
  drug-like molecule fits, yet it ranks first and the **site tags** say why ("fragment-sized",
  "anionic anchor").
- **Functional annotation** is curated UniProt evidence mapped by sequence alignment, so an
  unannotated pocket is not a pocket without function — only one nobody has annotated.
- **ChEMBL context** is matched by UniProt accession, and some proteins sit under a different
  target entry, so a low count is not proof that a target is undrugged.
- **Electrostatics** indicate sign and rough size only; use APBS for quantitative potentials.
- **Hydrogen bonds and salt bridges** come from heavy-atom geometry without hydrogens, so they
  are candidates. **Probe energies** are not binding free energies.
- A structure is **one conformation**: missing loops and assembly partners can create or hide
  pockets, which is why the audit runs first and its warnings reach the report.

## Output

`proteinscope report 1HSG` writes `output/1HSG/`:

```
report.html        Interactive: summary tiles, 3D viewer, sortable pocket table with expandable
                   details, ligand chemistry, UniProt function, ChEMBL context, figure gallery
report.pdf         Typeset report with every figure and table
report.md          The same content as Markdown
analysis.json      Every computed value, for your own scripts
viewer.html        3D page: pockets, surfaces, ligands (needs internet for 3Dmol.js)
figures/           Overview, surfaces, residue profile, pocket cut-aways, probe and contact maps
pymol/session.pml  Loads the structure and pocket point clouds into PyMOL
```

## Development

```bash
pytest                                  # offline suite
PROTEINSCOPE_NETWORK_TESTS=1 pytest     # adds the RCSB, AlphaFold, ChEMBL and benchmark tests
```

Structures and API responses are cached in `~/.cache/proteinscope`, or `$PROTEINSCOPE_CACHE`.

Code lives in `src/proteinscope/`: data sources (`rcsb`, `alphafold`, `chemistry`, `uniprot`,
`targets`, `enrich`), analysis (`structure`, `audit`, `surface`, `electrostatics`, `pockets`,
`druggability`, `interactions`, `pipeline`), and output (`viz/`, `report/`, `agents/`,
`mcp_server.py`, `web.py`, `cli.py`).

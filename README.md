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

## Claude Code as the agent (MCP, no API key)

`proteinscope mcp` serves the analysis tools over MCP, and `.mcp.json` registers it, so Claude
Code started in this folder offers the server after you approve it (`/mcp` shows its status).
Then just ask in the chat:

> Find a high-resolution KRAS G12C structure with a covalent inhibitor, analyse its pocket,
> compare it with the ChEMBL data and write the report.

Claude plans the steps and calls `search_pdb`, `analyze_structure`, `pocket_detail`,
`ligand_chemistry`, `target_bioactivity` and `write_report`. Reports land in `output/`.
To register the server outside this folder:

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
| Druggability | Weighted heuristic over size, enclosure, apolarity, depth and buried charge, plus site tags |
| Model confidence | For AlphaFold models: pLDDT per residue and per pocket |
| Ligand chemistry | CCD SMILES, RDKit descriptors (MW, cLogP, TPSA, QED), rule of five, 2D depiction |
| Target context | UniProt mapping, then ChEMBL target, measured potencies and known drugs |
| Ligands | Hydrogen bonds, salt bridges, hydrophobic contacts, π-stacking, buried surface |
| Probes | Six chemical probes scored on the pocket grid against the rigid protein |

Pocket detection is checked against known complexes: `tests/test_network.py` asserts that the
ligand of 1HSG, 2HYY, 3ERT, 1HCK and 2RH1 lands in the top-ranked pocket of its structure.

### What the numbers are not

- The **druggability score** is hand-weighted, not fitted. It ranks pockets within one structure
  and favours large apolar ones, so small charged sites rank low: trypsin's S1 pocket in 3PTB
  scores 0.20. The **site tags** flag those ("fragment-sized", "anionic anchor") instead of
  reweighting the score for one binding mode.
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
                   details, ligand chemistry, ChEMBL context, figure gallery, light/dark theme
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
To browse reports with working figures: `python3 -m http.server 8765 --directory output`.

Code lives in `src/proteinscope/`: data sources (`rcsb`, `alphafold`, `chemistry`, `targets`,
`enrich`), analysis (`structure`, `audit`, `surface`, `electrostatics`, `pockets`,
`druggability`, `interactions`, `pipeline`), and output (`viz/`, `report/`, `agents/`,
`mcp_server.py`, `cli.py`).

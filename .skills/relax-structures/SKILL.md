---
name: relax-structures
description: >
  Perform atomic structure relaxation (geometry optimisation) using MatterSim
  force-field models.
version: 1.0.0
triggers:
  - relax
  - relaxation
  - geometry optimisation
  - geometry optimization
  - structure optimisation
  - structure optimization
  - minimise energy
  - minimize energy
  - minimise forces
  - minimize forces
  - equilibrium geometry
  - optimize structure
  - optimise structure
  - optimize crystal
  - optimise crystal
tools:
  - python
  - bash
dependencies:
  - mattersim
  - ase
  - torch
  - numpy
  - atomate2
  - pymatgen
authors:
  - MatterSim Team
---

# Skill: Structure Relaxation

## Description

Perform atomic structure relaxation (geometry optimisation) using MatterSim
force-field models. Given one or more crystal or molecular structures, this
skill optimises atomic positions (and optionally lattice vectors) to minimise
the potential energy, then reports results to the terminal and saves them to
disk.

## Purpose

Use this skill when the user asks to:
- Relax / optimise a crystal or molecular structure
- Find the equilibrium geometry of an atomic system
- Minimise forces on atoms using MatterSim

## Inputs

See [`skill.json`](skill.json) for the full machine-readable schema.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `structures` | `List[ase.Atoms]` | *(required)* | Structures from files, builders, or inline coordinates |
| `model` | `string` | `mattersim-v1.0.0-1m` | Model checkpoint (`-1m` or `-5m`) |
| `device` | `string` | `auto` | `auto` (prefers CUDA), `cpu`, or `cuda` |
| `optimizer` | `string` | `FIRE` | `FIRE` or `BFGS` |
| `fmax` | `float` | `0.01` | Force convergence tolerance (eV/Å) |
| `steps` | `int` | `500` | Maximum optimisation steps |
| `filter` | `string \| null` | `null` | `null` (fixed cell), `FrechetCellFilter`, or `ExpCellFilter` |
| `pressure` | `float \| null` | `null` | External target pressure; implies `FrechetCellFilter` when filter is null |
| `pressure_unit` | `string` | `GPa` | Unit of pressure value: `GPa`, `kbar`, or `eV/A^3` |
| `constrain_symmetry` | `bool` | `false` | Preserve space-group symmetry via ASE `FixSymmetry` |

### Pressure unit conversion

The user may specify pressure in different units. **Always convert to eV/Å³**
before passing to the ASE cell filter's `scalar_pressure` parameter:

| Unit | Conversion to eV/Å³ |
|------|---------------------|
| `GPa` | `pressure * ase.units.GPa` (i.e., divide by 160.2177) |
| `kbar` | `pressure * ase.units.GPa / 10` (1 kbar = 0.1 GPa) |
| `eV/A^3` | `pressure` (no conversion needed) |

**Common pitfall**: ASE's `ExpCellFilter` / `FrechetCellFilter` expect
`scalar_pressure` in **eV/Å³**. Passing a value in GPa without conversion
will apply ~160× the intended pressure. Always convert first.

### Symmetry constraints

When `constrain_symmetry=True`, the `Relaxer` applies ASE's `FixSymmetry`
constraint, which preserves the space-group symmetry of the input structure
throughout the optimisation. Atomic displacements are projected onto the
symmetry-allowed subspace at each step.

## Outputs

All outputs are written to `mattersim_relax_results/<timestamp>/`:

| File | Description |
|------|-------------|
| `task_NNN_FORMULA.json` | `ForceFieldTaskDocument` JSON (atomate2 / Materials Project compatible) per structure |
| `relax_NNN_FORMULA.cif` | Relaxed structure (CIF for periodic, XYZ for non-periodic) |
| *(terminal)* | Formatted summary table |

### Output schema

Each `task_NNN_FORMULA.json` is a serialised
[`ForceFieldTaskDocument`](https://github.com/materialsproject/atomate2/blob/main/src/atomate2/forcefields/schemas.py)
(from `atomate2.forcefields.schemas`), which extends `AseStructureTaskDoc`
from the atomate2 / emmet-core ecosystem.

Key fields:

```
ForceFieldTaskDocument
├── output
│   ├── energy          — final total energy (eV)
│   ├── energy_per_atom — final energy per atom (eV/atom)
│   ├── forces          — final forces (eV/Å), shape (N, 3)
│   ├── stress          — final stress (kbar), shape (3, 3) symmetric
│   └── ionic_steps[]   — [initial, final] frames with energy and forces
├── input
│   ├── structure       — input pymatgen Structure / Molecule
│   ├── relax_cell      — whether cell vectors were relaxed
│   └── fix_symmetry    — whether symmetry was constrained
├── forcefield_name     — "MatterSim"
├── forcefield_version  — mattersim package version
└── converged           — True if fmax criterion was met
```

Stress units: **kbar** (3×3 symmetric matrix). The template passes
`relaxed.get_stress()` (eV/Å³ Voigt 6-vector) to the trajectory, and
`ForceFieldTaskDocument` converts via `convert_stress_from_voigt_to_symm`
automatically.

Non-periodic structures (molecules) produce a
`ForceFieldMoleculeTaskDocument` with no stress field.

---

## Procedure

Adapt and execute the reference template script at
[`scripts/relax_template.py`](scripts/relax_template.py). The script has five
clearly marked steps — modify only what the user's request requires:

| Step | What to adapt |
|------|---------------|
| **Step 1** — Build or load structures | Replace the placeholder with the user's structures (files, bulk builders, inline coordinates, pymatgen objects) |
| **Step 2** — Load calculator | Set `model_name` and `device` per user request |
| **Step 3** — Configure and run relaxation | Set `optimizer`, `fmax`, `steps`, `filter_name`, `pressure_in_GPa`, `constrain_symmetry` |
| **Step 4** — Print results | No changes needed |
| **Step 5** — Save results | No changes needed |

For **large batches** (> 50 structures), use
[`scripts/batch_relax_template.py`](scripts/batch_relax_template.py) instead.

### Supported structure input methods

| Method | Code |
|--------|------|
| Files on disk | `ase_read("file.cif", index=":")` |
| Multiple files | `for path in paths: atoms_list.extend(ase_read(path, index=":"))` |
| Bulk prototypes | `bulk("Si", "diamond", a=5.43, cubic=True)` |
| Inline coordinates | `Atoms(symbols=[...], positions=[...], cell=[...], pbc=True)` |
| Pymatgen | `AseAtomsAdaptor.get_atoms(Structure.from_file("POSCAR"))` |

> If the user's input is ambiguous, ask for clarification before proceeding.

---

## Rules

1. **Single calculator instance** — instantiate `MatterSimCalculator` once and
   share it across all structures.
2. **Auto-detect device** — use CUDA when available; fall back to CPU.
3. **No cell filter for non-periodic structures** — check `all(atoms.pbc)`
   before applying any cell filter.
4. **Isolate per-structure failures** — if one structure fails, log the error
   and continue with the rest.
5. **Never suppress errors** — always print and record them in the JSON output.
6. **Absolute imports only** — e.g. `from mattersim.applications.relax import Relaxer`.
7. **Cell relaxation is opt-in** — only apply a filter when the user explicitly
   requests cell / volume relaxation.
8. **Respect user parameters** — if the user specifies model, device, optimizer,
   fmax, steps, filter, or pressure, use those values instead of defaults.
9. **Print the output path** at the end so the user can find their results.
10. **Report results in the dialogue** — after the script finishes, summarise
    the key results directly in the conversation UI. Include for each structure:
    formula, convergence status, total energy, energy per atom, max force, and
    the output directory path. If any structure failed, highlight the error.
    Do not just say "done" — the user should be able to read the results without
    opening files or scrolling through terminal output.

---

## Examples

- [`examples/relax_bulk_si.py`](examples/relax_bulk_si.py) — Relax a bulk
  silicon cell (runnable with CLI arguments).

## Integration

| Agent | How to reference |
|-------|-----------------|
| **GitHub Copilot** | Discoverable from `.skills/`; can symlink into `.github/prompts/` |
| **OpenAI Codex** | Reference from `AGENTS.md` or include in agent context |
| **Claude Code** | Reference from `CLAUDE.md` or include in agent context |
| **Cursor / others** | Point agent instructions at this `SKILL.md` |

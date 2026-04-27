---
name: phonon-calculation
description: >
  Calculate phonon properties (band structure, DOS, force constants) for
  periodic structures using MatterSim and phonopy.
version: 1.0.0
triggers:
  - phonon
  - phonon calculation
  - phonon band structure
  - phonon dispersion
  - phonon DOS
  - phonon density of states
  - force constants
  - lattice dynamics
  - vibrational properties
  - imaginary frequency
  - dynamical stability
tools:
  - python
  - bash
dependencies:
  - mattersim
  - ase
  - torch
  - numpy
  - phonopy
  - matplotlib
authors:
  - MatterSim Team
related_skills:
  - relax-structures
---

# Skill: Phonon Calculation

## Description

Calculate phonon properties — band structure, density of states (DOS), and
force constants — for periodic crystal structures using MatterSim as the
force-field backend and phonopy for lattice dynamics. Optionally relaxes the
structure first (recommended).

## Purpose

Use this skill when the user asks to:
- Calculate phonon band structure or phonon dispersion
- Compute phonon density of states (DOS)
- Check for imaginary frequencies / dynamical stability
- Obtain force constants for a crystal structure
- Perform lattice dynamics calculations with MatterSim

## Inputs

See [`skill.json`](skill.json) for the full machine-readable schema.

### Core parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `structures` | `List[ase.Atoms]` | *(required)* | Fully periodic structures (`pbc=True` in all directions) |
| `model` | `string` | `mattersim-v1.0.0-1m` | Model checkpoint (`-1m` or `-5m`) |
| `device` | `string` | `auto` | `auto` (prefers CUDA), `cpu`, or `cuda` |

### Relaxation parameters (pre-phonon)

Phonon calculations require well-relaxed structures. By default, relaxation
runs before phonon. Set `relax_first=false` only if the structure is already
relaxed.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `relax_first` | `bool` | `true` | Relax before phonon calculation |
| `relax_optimizer` | `string` | `FIRE` | `FIRE` or `BFGS` |
| `relax_fmax` | `float` | `0.01` | Force convergence (eV/Å) |
| `relax_steps` | `int` | `500` | Max relaxation steps |
| `relax_filter` | `string \| null` | `FrechetCellFilter` | Cell filter (`null` = fixed cell) |
| `relax_pressure` | `float \| null` | `null` | External pressure |
| `relax_pressure_unit` | `string` | `GPa` | `GPa`, `kbar`, or `eV/A^3` |
| `relax_constrain_symmetry` | `bool` | `true` | Preserve symmetry during relaxation |

> **Note on defaults**: For phonon calculations, the defaults differ from the
> standalone relaxation skill: `relax_filter` defaults to `FrechetCellFilter`
> (cell relaxation on) and `relax_constrain_symmetry` defaults to `true`,
> because phonon calculations are sensitive to residual stress and symmetry
> breaking.

### Phonon parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `find_prim` | `bool` | `false` | Find and use the primitive cell |
| `amplitude` | `float` | `0.01` | Displacement magnitude (Å) for finite differences |
| `supercell_matrix` | `[int,int,int] \| null` | `null` | Supercell size; `null` for auto |
| `qpoints_mesh` | `[int,int,int] \| null` | `null` | Q-point mesh; `null` for auto |
| `max_atoms` | `int \| null` | `null` | Max atoms for auto supercell; `null` for auto |

## Outputs

All outputs are written to `mattersim_phonon_results/<timestamp>/`, with
per-structure subdirectories:

```
mattersim_phonon_results/<timestamp>/
├── phonon_results.json                  # Combined results for all structures
└── 000_Si8/
    ├── relax/                           # (if relax_first=true)
    │   └── relax_results.json
    ├── phonon/                          # phonopy output
    │   ├── Si8_phonon_band.png
    │   ├── Si8_phonon_dos.png
    │   ├── band.yaml
    │   ├── total_dos.dat
    │   └── phonopy_params.yaml
    └── relaxed.cif                      # (if relax_first=true)
```

### JSON schema

```
phonon_results.json
├── schema_version: "1.0"
├── task: "phonon_calculation"
├── metadata: { model, device, relax_first, phonon_params, timestamp,
│               n_structures }
├── units: { energy: "eV", forces: "eV/Å", stress: "GPa",
│            positions: "Å", cell: "Å", frequency: "THz" }
└── structures[]:
    ├── index
    ├── input: { formula, n_atoms, pbc, cell, symbols, positions }
    ├── relaxation: { ran, converged, energy_eV, max_force, relaxed_cell,
    │                 relaxed_positions } | null
    └── phonon: { has_imaginary, supercell_matrix, qpoints_mesh,
                  n_displacements, work_dir, output_files, error? }
```

---

## Procedure

Adapt and execute the reference template script at
[`scripts/phonon_template.py`](scripts/phonon_template.py). The script has
six clearly marked steps — modify only what the user's request requires:

| Step | What to adapt |
|------|---------------|
| **Step 1** — Build or load structures | Replace the placeholder with the user's structures |
| **Step 2** — Load calculator | Set `model_name` and `device` per user request |
| **Step 3** — (Optional) Relax structures | Set relaxation parameters; set `relax_first=False` to skip |
| **Step 4** — Run phonon calculations | Set `find_prim`, `amplitude`, `supercell_matrix`, `qpoints_mesh`, `max_atoms` |
| **Step 5** — Print results | No changes needed |
| **Step 6** — Save results | No changes needed |

### Supported structure input methods

Same as the [relax-structures](../relax-structures/SKILL.md) skill:

| Method | Code |
|--------|------|
| Files on disk | `ase_read("file.cif", index=":")` |
| Multiple files | `for path in paths: atoms_list.extend(ase_read(path, index=":"))` |
| Bulk prototypes | `bulk("Si", "diamond", a=5.43, cubic=True)` |
| Inline coordinates | `Atoms(symbols=[...], positions=[...], cell=[...], pbc=True)` |
| Pymatgen | `AseAtomsAdaptor.get_atoms(Structure.from_file("POSCAR"))` |

> If the user's input is ambiguous, ask for clarification before proceeding.

### Relationship to relax-structures skill

This skill **reuses the same `Relaxer` class** from
`mattersim.applications.relax` for the optional pre-phonon relaxation step.
It does NOT invoke the relax-structures skill as a sub-skill — the phonon
script is fully self-contained.

When `relax_first=true` (default), the script:
1. Relaxes the structure with cell relaxation and symmetry constraints
2. Records relaxation provenance (convergence, energy, forces) in the JSON
3. Uses the relaxed structure for phonon calculation

---

## Rules

1. **Periodic structures only** — require `all(atoms.pbc)` and fail early
   with a clear message if any structure is non-periodic.
2. **Single calculator instance** — instantiate `MatterSimCalculator` once and
   share it across all structures and both relaxation + phonon steps.
3. **Auto-detect device** — use CUDA when available; fall back to CPU.
4. **Isolate per-structure failures** — if one structure fails, log the error
   and continue with the rest.
5. **Per-structure subdirectories** — each structure gets its own directory
   with `relax/` and `phonon/` subdirs. Use absolute paths.
6. **Validate array inputs** — coerce `supercell_matrix` and `qpoints_mesh`
   to `np.array` and validate shapes: `(3,)` or `(3,3)` for supercell,
   `(3,)` for qpoints.
7. **Set matplotlib backend** — use `matplotlib.use("Agg")` at the top of the
   script for headless compatibility.
8. **Relax by default** — run relaxation before phonon unless the user
   explicitly says to skip. Default relaxation uses cell relaxation + symmetry
   constraints (different from the standalone relax skill defaults).
9. **Absolute imports only** — e.g. `from mattersim.applications.phonon import PhononWorkflow`.
10. **Respect user parameters** — if the user specifies model, device, supercell
    matrix, q-mesh, amplitude, or relaxation settings, use those values.
11. **Report results in the dialogue** — after the script finishes, summarise
    the key results directly in the conversation UI. Include for each structure:
    formula, whether relaxation converged, whether imaginary frequencies were
    found, the supercell matrix used, and the output directory path. Do not
    just say "done".

---

## Examples

- [`examples/phonon_bulk_si.py`](examples/phonon_bulk_si.py) — Calculate
  phonon properties for bulk silicon (runnable with CLI arguments).

## Integration

| Agent | How to reference |
|-------|-----------------|
| **GitHub Copilot** | Discoverable from `.skills/`; can symlink into `.github/prompts/` |
| **OpenAI Codex** | Reference from `AGENTS.md` or include in agent context |
| **Claude Code** | Reference from `CLAUDE.md` or include in agent context |
| **Cursor / others** | Point agent instructions at this `SKILL.md` |

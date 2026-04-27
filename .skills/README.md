# Skills

This directory contains skill definitions for AI coding agents (GitHub Copilot,
OpenAI Codex, Claude Code, Cursor, etc.). Each subdirectory defines a
self-contained skill with instructions that an agent can follow to perform a
specific task.

## Available Skills

| Skill | Description |
|-------|-------------|
| [relax-structures](./relax-structures/) | Run structure relaxation with MatterSim |

## Structure

Each skill lives in its own directory with the following layout:

```
.skills/
├── README.md                          # This file
└── <skill-name>/
    ├── SKILL.md                       # Instructions for AI agents (required)
    ├── skill.json                     # Machine-readable metadata (inputs, outputs, defaults)
    └── examples/                      # Runnable examples
        └── *.py
```

## Integration

- **GitHub Copilot** — skills are auto-discoverable from `.skills/` or can
  be symlinked into `.github/prompts/`.
- **OpenAI Codex** — reference skill files from `AGENTS.md`.
- **Claude Code** — reference skill files from `CLAUDE.md`.
- **Cursor / others** — point agent instructions at the relevant `SKILL.md`.


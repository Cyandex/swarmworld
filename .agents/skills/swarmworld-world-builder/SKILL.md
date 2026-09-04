---
name: swarmworld-world-builder
description: Create, revise, review, or debug declarative 2-D SwarmWorld scenario packages under worlds/, including their geometry, resources, technology operations, facilities, hazards, prompts, documentation, tests, runs, replay, and analysis. Do not use for ordinary work on the legacy biological world.
---

# SwarmWorld World Builder

Build an opt-in world package that runs through the existing SwarmWorld engine without changing the biological default or another completed world.

## Establish the scope

Work from the SwarmWorld repository root. Read the repository instructions, then read `worlds/ashen_realms/README.md`, `worlds/ashen_realms/scenario.yaml`, and every document referenced by that manifest. Treat Ashen Realms as a structural example, not as content to overwrite or retain accidentally.

Classify the request before editing:

- For review or diagnosis, inspect and report; do not implement unrelated improvements.
- For a new world, create `worlds/<new_id>/` and keep all existing worlds unchanged.
- For a change to an existing world, preserve its identity and bump its semantic version when runtime-defining package content changes.
- If the requested behavior exceeds format version 1, explain the engine/protocol boundary before changing Python, TypeScript, action spaces, replay formats, or fixed catalog sizes.

Read [the format-v1 contract](references/format-v1-contract.md) before authoring or changing package data. Read [the verification workflow](references/verification-workflow.md) before running tests or producing a handoff.

## Author the package

For a new world, copy `worlds/ashen_realms/` into a new sibling directory and replace all Ashen-specific content. Do not leave its identity, configuration paths, prompts, README claims, or reference results in the new package.

Design the world as a coherent system:

1. Give `scenario.yaml` a new lowercase ID, name, description, version, agent prompt, and complete internal document routing.
2. Map every required stable slot exactly once to a unique public scenario ID. Stable slots are compatibility addresses, not inherited meanings.
3. Compose ordered normalized 2-D geometry with reachable walkable regions, resources, facilities, and workspaces.
4. Define required compatibility fields plus domain fields with bounded initial and update behavior.
5. Define six-component resource compositions, deposits, capacities, and intentional regrowth.
6. Define all ten public process operations and bounded process effects.
7. Define facility capabilities and placements so the default recipe is legal and reachable.
8. Define disturbances, mission, milestones, default recipe, artifact terminology, service aliases, rendering metadata, and analysis aliases.
9. Rewrite agent instructions using only the new scenario vocabulary. Keep researcher guidance separate from runtime claims.
10. Add a standalone README, configs, tests, and isolated run/analysis paths for the new world.

Package YAML is data only. Never add executable Python, shell commands, imports, network access, or self-modifying behavior to a world package. Do not let in-world agents mutate the active package during an episode.

## Preserve scientific meaning

The generic scripted policy reads `missions.yaml` and must not gain world-specific names in Python. LLM agents receive the resolved scenario descriptor and scenario-specific resource and operation enums; physics and action legality remain authoritative.

Distinguish shared metrics from scenario semantics. Movement, action counts, coverage, mass flow, facility use, provenance, and determinism are comparable when experimental designs match. Composition dimensions, material properties, and services must be interpreted through the active scenario aliases and must not be pooled across worlds solely because they share internal numeric slots.

Do not claim mechanics the engine does not implement. In format version 1, a road-like terrain can be walkable, visible, and alter fields, but does not intrinsically change movement speed or cost. Renderer colors and heights are declarative; arbitrary custom meshes are not.

## Verify and hand off

Validate early, then run proportionate tests. A completed new world should normally have:

- successful `world validate`, `world describe`, and `doctor` checks;
- deterministic scenario tests based on `tests/test_scenarios.py`;
- a short scripted smoke trace and deterministic replay;
- a trace report with scenario analysis in a world-specific output directory;
- Python and web regression checks when the affected scope warrants them;
- a documented reference run whose claims match the generated trace;
- no edits to existing worlds or unrelated dirty-worktree files.

Use `runs/<world_id>/...` for all new traces, reports, figures, and studies. Never reuse the biological, Ashen Realms, paper-study, dossier, or technology-atlas output directories.

In the final handoff, summarize the world definition, changed files, commands run, validation results, reference-run behavior, limitations, and any format-v1 requirement the world could not express.

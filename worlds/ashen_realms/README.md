# Ashen Realms

Ashen Realms is a separate, declarative SwarmWorld example for volcanic terrain,
rocks, ores, metallurgy, hazardous fields, ancient roads, mines, and forge enclaves.
It is an original epic-fantasy setting. It uses broad mythic and volcanic motifs but
does not copy named locations, characters, lore, or artwork from Tolkien.

The existing biological world remains the default. Nothing in this folder is loaded
unless a configuration explicitly sets:

```yaml
world:
  scenario_package: worlds/ashen_realms
```

This document is both a human runbook and the authoring contract for an agent that
wants to construct another world.

## Quick start: 24 agents for 100 ticks

Run these commands from the `SwarmWorld` repository root:

```bash
conda activate PyTorch
python -m pip install -e ".[dev,analysis]"

biofoundry world validate worlds/ashen_realms
biofoundry doctor --config worlds/ashen_realms/configs/demo.yaml

biofoundry simulate \
  --config worlds/ashen_realms/configs/demo.yaml \
  --agents 24 \
  --ticks 100 \
  --policy scripted \
  --output runs/ashen-realms-24x100.jsonl
```

For a declarative scenario, `--policy scripted` automatically selects the generic
catalog-aware industry policy. It reads the package's default recipe; the policy does
not contain Ashen Realms resource or operation names.

The run prints a common SwarmWorld summary plus `scenario` and `scenario_analysis`.
The scenario analysis reports extraction mass by material, remaining deposits,
disturbance exposure, field histories, operation and facility use, process yield,
aliased material properties, microbatches, tests, artifacts, services, and spatial
coverage.

### Verified reference result

The command above was run on 2026-08-20 with seed 1701. It completed all 2,400 agent
action outcomes with zero rejections. The society harvested 25.181141 iron ore,
10.796587 carbon fuel, and 7.149538 limestone flux; fabricated and tested 21
microbatches with a 100% test-completion and pass rate; used the Great Forge 21 times;
and built six volcanic load-and-heat barriers. One eruption and one ashfall occurred.
Deterministic replay verified all 12 snapshots and ended at digest
`c568eb9a47aef76cf51b9cdf37c3baf516e3213f5420d3c523159477be06edf3`.

The complete compact result, including package hash and regression-check counts, is
machine-readable at
[`examples/reference_24x100_summary.json`](examples/reference_24x100_summary.json).
Exact values are a regression reference for this package version, engine revision,
seed, and policy—not a scientific claim about real metallurgy.

## Analyze and replay the run

### Isolated single-run analysis

Run the following from the `SwarmWorld` repository root. Every generated analysis
file is written beneath `runs/ashen-realms-analysis/24-agent-100-tick-seed-1701/`.
The commands only read the source trace and do not write to any biological-world
study, report, figure, dossier, or technology-atlas directory.

```bash
mkdir -p runs/ashen-realms-analysis/24-agent-100-tick-seed-1701

biofoundry replay \
  runs/ashen-realms-24x100.jsonl \
  > runs/ashen-realms-analysis/24-agent-100-tick-seed-1701/replay-verification.json

biofoundry trace-report \
  runs/ashen-realms-24x100.jsonl \
  --output runs/ashen-realms-analysis/24-agent-100-tick-seed-1701/trace-report.json \
  --figures-dir runs/ashen-realms-analysis/24-agent-100-tick-seed-1701/trace-figures

biofoundry ecosystem-report \
  runs/ashen-realms-24x100.jsonl \
  --output-dir runs/ashen-realms-analysis/24-agent-100-tick-seed-1701/ecosystem \
  --horizon 100 \
  --evaluation-seeds 2701 2702 2703 2704 2705 2706 2707 2708
```

This creates:

- `replay-verification.json`, containing the deterministic replay result and final
  digest;
- `trace-report.json`, containing common action, mobility, experimental-cycle,
  artifact, causal, model, and execution diagnostics;
- the report's `scenario_analysis`, containing named ore extraction and remaining
  deposits, eruption exposure, field trajectories, operation and facility usage,
  process yield, aliased metallurgy properties, test outcomes, and aliased final
  artifact services;
- `trace-figures/`, containing PNG and PDF provenance/lineage figures;
- `ecosystem/ecosystem-report.json`, containing the frozen artifact-system assay and
  held-out disturbance schedules; and
- `ecosystem/` PNG and PDF overview, map, lineage, and interaction figures.

`trace-report` is the principal analysis for this scenario. `ecosystem-report` is an
optional agent-free assay of the six final artifacts. Its historical common service
slots remain stable for compatibility; interpret them using the scenario service
aliases embedded in the trace descriptor. The held-out seeds above evaluate the
frozen final society and do not continue agent learning.

Do not use `analyze-study` on this one trace: that command estimates across independent
simulation seeds and belongs to the multi-seed workflow below.

### Browser playback

Start the offline playback server:

```bash
biofoundry playback runs/ashen-realms-24x100.jsonl
```

Then start the browser client from a second terminal:

```bash
cd web
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`.

`replay` verifies snapshot integrity and deterministic action replay. It also rejects
a trace when the installed package hash differs from the hash recorded in the trace.
`trace-report` retains all common behavioral and causal sections and adds a
`scenario_analysis` section. `playback` exposes the completed society to the same web
observatory without model calls.

To view the live world instead, run:

```bash
biofoundry serve \
  --config worlds/ashen_realms/configs/demo.yaml \
  --record runs/ashen-realms-live.jsonl
```

Then, in a second terminal:

```bash
cd web
npm ci
npm run dev
```

Open the URL printed by Vite. The observatory reads scenario-provided terrain and
resource colors, heights, names, facilities, field layers, and legends. All extra
fields are selectable and appear in cell inspection.

## Inspect the resolved world contract

```bash
biofoundry world describe worlds/ashen_realms
```

The description is intended for humans and agents. It includes the scenario identity,
version, package hash, finite catalog limits, catalogs, fields, services, mission, and
property aliases.

## What the example defines

Ashen Realms includes:

- a 2-D rectangular world with an active caldera, magma sea, lava channels, obsidian
  waste, ash plains, sulfur marsh, iron mountains, ancient roads, a forge enclave, and
  a proving ground;
- deterministic elevation, temperature, lava depth, ash load, ground stability,
  toxic gas, water availability, and solar exposure fields;
- basalt, obsidian, iron ore, copper ore, sulfur, limestone flux, water, and carbon
  fuel deposits;
- wash, crush, roast, smelt, refine, alloy, cast, forge, quench, and anneal operations;
- ore washers, roasting hearths, furnaces, forges, assay halls, and lore archives;
- eruption, ashfall, and earthquake disturbances;
- a safe package-defined metallurgy surrogate tracking composition, purity, porosity,
  grain refinement, work hardening, thermal control, oxidation, and process quality;
- scenario aliases for artifact geometry, material properties, and persistent services;
- a mission, default recipe, scripted baseline, LLM instructions, renderer metadata,
  and analysis definitions.

Package files are data. They cannot import Python, execute shell commands, access the
network, or mutate physics during an episode.

## Package layout

```text
worlds/ashen_realms/
  README.md
  scenario.yaml             package identity and document routing
  terrain.yaml              terrain catalog
  geometry.yaml             ordered 2-D geometry layers
  fields.yaml               environmental state and update operators
  resources.yaml            matter catalog and spatial deposits
  operations.yaml           technology vocabulary and process effects
  facilities.yaml           facility catalog and placements
  artifacts.yaml            artifact and geometry vocabulary
  missions.yaml             objective, milestones, and default recipe
  disturbances.yaml         deterministic hazard sequence
  rendering.yaml            colors, services, and field overlays
  analysis.yaml             composition/property names and metrics
  configs/
    demo.yaml                24 agents, 100 ticks
    study.yaml               longer scripted study profile
    llm_agent.yaml           structured-output LLM profile template
  prompts/
    agent_instructions.md
    researcher_instructions.md
  examples/
    minimal_world.yaml
    reference_24x100_summary.json
```

`scenario.yaml` is the only entry point. Every referenced YAML file and the agent
prompt contributes to the SHA-256 package hash recorded in traces.

## Run other compatible workflows

### Scaling benchmark

```bash
biofoundry benchmark \
  --config worlds/ashen_realms/configs/demo.yaml \
  --agents 8 24 64 \
  --ticks 100
```

### Multi-seed study

```bash
biofoundry research-study \
  --config worlds/ashen_realms/configs/study.yaml \
  --policy scripted \
  --conditions full \
  --seeds 1701 1702 1703 1704 \
  --ticks 256 \
  --agents 24 \
  --output-dir runs/ashen-realms-study

biofoundry analyze-study \
  runs/ashen-realms-study/study-summary.json \
  --output-dir runs/ashen-realms-study/analysis
```

The seed is the unit of replication. Common mobility, action, specialization,
communication, provenance, and outcome columns remain compatible. World-specific
metallurgy and hazard results must be interpreted from each trace's
`scenario_analysis`; do not silently compare them to biological metrics.

### LLM agents

First edit `configs/llm_agent.yaml` to supply the provider model and base URL expected
by your environment. Then run:

```bash
biofoundry simulate \
  --config worlds/ashen_realms/configs/llm_agent.yaml \
  --policy llm \
  --ticks 100 \
  --output runs/ashen-realms-llm.jsonl
```

The strict JSON schema advertises Ashen Realms resource and operation IDs rather than
the biological vocabulary. The system prompt uses `prompts/agent_instructions.md`,
and every semantic observation includes the complete resolved scenario descriptor.
Physics and action legality remain authoritative even when an agent proposes an
invalid recipe or unsupported claim.

## How to make an entirely new world

Copy the whole directory so the existing example remains immutable:

```bash
cp -R worlds/ashen_realms worlds/my_world
```

Then perform the following sequence.

### 1. Give the package a new identity

Edit `scenario.yaml`:

- choose a globally meaningful lowercase `id`;
- choose a human-readable `name`;
- start a new semantic `version`;
- describe the scientific setting;
- retain `format_version: 1`;
- keep every referenced document inside the package directory.

Never reuse a version after changing package content. The hash detects changes, but a
meaningful version makes reports interpretable.

### 2. Redefine terrain slots

Format version 1 preserves the established numeric wire shape. It therefore has nine
terrain slots. Every slot must be defined exactly once:

```text
DEEP_WATER, TIDAL, MEADOW, FUNGAL_GROVE, CHITIN_GARDEN,
CELLULOSE_FIELD, MINERAL_SPRING, FOUNDRY, TEST_FIELD
```

The slot is an internal compatibility address, not its meaning. Ashen Realms maps
`FUNGAL_GROVE` to `OBSIDIAN_WASTE` and exposes only the latter ID to scenario agents.
Each record defines at least `slot`, `id`, `name`, `walkable`, `color`, and `height`.
Legacy agents never see these aliases because they use the unchanged default world.

### 3. Compose the 2-D geometry

`geometry.yaml` assigns a base terrain and then applies features in file order. Later
features overwrite earlier ones. Supported deterministic shapes are:

- `circle`: normalized `center` and scalar `radius`;
- `ring`: normalized `center`, `inner`, and `outer` radii;
- `ellipse`: normalized `center` and two-axis `radius`;
- `rectangle`: normalized `[x0, y0, x1, y1]` bounds;
- `ridge`: normalized `start`, `end`, and `width`;
- `noise`: seeded per-cell `probability`.

Coordinates range from zero to one, so the same definition scales to different grid
sizes. The runtime relocates a facility to the nearest walkable cell when necessary.
The validator rejects unknown shapes and catalog references.

### 4. Define environmental fields

Each field record supports:

- `range: [minimum, maximum]`;
- a stable `id` and display `name`;
- initial constant, x/y gradients, radial sources, seeded noise, and terrain offsets;
- bounded four-neighbor diffusion;
- decay;
- per-terrain sources;
- an optional deterministic day cycle.

Format version 1 requires five compatibility fields so shared simulation and artifact
interfaces remain stable:

```text
temperature
water_availability   -> shared moisture channel
ground_stability     -> shared nutrients channel
toxic_gas            -> shared contamination channel
solar
```

These aliases do not claim that stability is a nutrient. They preserve the numeric
observation and replay contract while the scenario descriptor supplies the real
meaning. Add any number of extra fields, such as `elevation`, `lava_depth`, or
`ash_load`; they are serialized, analyzed, rendered, and available in semantic
observations.

### 5. Define resources and deposits

Format version 1 has `NONE` plus eight matter slots. Define every slot exactly once.
Each resource has a public `id`, display metadata, and six normalized composition
values. `analysis.yaml` names those six dimensions for the new technology domain.

A deposit selects a resource, geometry shape, capacity, variation, initial-fill range,
and optional regrowth. A non-regenerating ore should omit `regrowth` or set it to zero.
Deposits are applied in file order, so a later rare deposit can replace a common rock
deposit on overlapping cells.

### 6. Define technology operations

Format version 1 has ten stable process slots. Each operation maps one slot to a public
ID and declares bounded additive effects on:

```text
purity
porosity
grain_refinement
work_hardening
thermal_control
oxidation
quality
```

Recipe intensity is bounded to `[0, 1]`; a recipe contains one to twelve steps. The
engine computes deterministic mass yield and common property channels. `analysis.yaml`
renames those channels for the domain—for example, shared `stiffness` is reported as
`strength` in Ashen Realms. Effects are transparent game-level surrogates, not
quantitative metallurgical predictions.

### 7. Define facilities

Map all seven stable facility slots, including `NONE`, to scenario IDs. Placements use
normalized coordinates. Declare each facility's `operations`; the runtime rejects a
recipe when the current terrain/facility capability union does not cover every step.
A workspace terrain can also declare operations. Ashen Realms gives the Great Forge
the complete integrated process chain while individual stations expose narrower
capabilities.

### 8. Define hazards

Each disturbance has an ID, name, normalized radius, and a mapping of field deltas.
It may also transform sufficiently affected cells to a declared terrain. The run
configuration controls the interval and intensity. Disturbances execute in listed
order and repeat deterministically.

### 9. Define mission, artifacts, presentation, and analysis

- `missions.yaml` gives humans and agents the objective, milestones, and a complete
  default recipe used by the generic scripted policy.
- `artifacts.yaml` assigns scenario language to the stable material-system and geometry
  channels.
- `rendering.yaml` supplies service aliases, colors, and field-overlay metadata.
- `analysis.yaml` names composition and property channels and documents expected
  scenario metrics.
- `prompts/agent_instructions.md` gives agents domain context without executable code.

Common artifact arrays and service slots remain stable for replay and analysis. Use
the descriptor aliases when interpreting them. Do not pool a scenario-specific service
with a biological service merely because they occupy the same stable numeric slot.

### 10. Validate before running

```bash
biofoundry world validate worlds/my_world
biofoundry world describe worlds/my_world
biofoundry doctor --config worlds/my_world/configs/demo.yaml
```

Validation checks package containment, required documents, catalog completeness,
duplicate IDs, cross-references, required compatibility fields, facility positions,
geometry shapes, and the default recipe. It prints the package hash on success.

### 11. Add deterministic tests

A serious new world should test:

- identical seeds produce identical initial and final state digests;
- all required resources and facilities are reachable;
- every declared resource appears in at least one valid configuration;
- field values remain inside declared ranges;
- disturbances occur at expected ticks;
- the default recipe validates and can execute;
- scenario IDs appear in structured-output schemas and semantic observations;
- replay rejects a changed package hash;
- web and optional Godot renderers handle all catalog entries;
- legacy SwarmWorld tests remain unchanged.

Use `tests/test_scenarios.py` as the starting pattern.

## Reading outputs correctly

Every new trace records:

- engine revision and package backend;
- scenario ID, name, semantic version, and SHA-256 package hash;
- complete resolved run configuration;
- prompt, schema, and configuration hashes;
- authoritative snapshots with all scenario fields and catalogs;
- ordinary actions and events.

Common analysis is safe for movement, coverage, inventory flow, action counts,
communication, facility use, technology-cycle completion, artifact provenance, and
determinism. Scenario analysis is safe only under its recorded package meaning.

If two traces have different package hashes, treat them as different treatments even
when their scenario IDs and versions match. Do not pool them without an explicit,
documented migration decision.

## Machine-readable handoff for an agent

An authoring agent should follow this exact order:

1. Read `scenario.yaml` and every referenced document.
2. Copy the package; never alter Ashen Realms in place for a different study.
3. Change identity and version.
4. Fill every stable catalog slot once, using unique public IDs.
5. Define ordered geometry and reachable workspaces.
6. Define required compatibility fields plus domain fields.
7. Define resource compositions and spatial deposits.
8. Define operation effects and a valid default recipe.
9. Define facilities, disturbances, artifact aliases, mission, rendering, and analysis.
10. Write human and agent prompts using only the new scenario vocabulary.
11. Run `world validate`, `world describe`, and `doctor`.
12. Run a short scripted trace, deterministic replay, and `trace-report`.
13. Run Python tests and web tests before handing the package to another researcher.

At no point should an authoring agent add executable code to YAML or allow an in-world
agent to rewrite the active package during an episode.

## Current format limits

The first scenario format intentionally favors compatibility over unbounded catalogs:

- 9 terrain slots;
- 8 non-empty resource slots;
- 6 non-empty facility slots;
- 10 operation slots;
- one stable material-system artifact type;
- rectangular 2-D cardinal movement.

These are explicit validation limits. They keep current commands, observations,
PettingZoo spaces, renderer messages, and old replays compatible. A future format can
expand them through a versioned observation and replay protocol; it should not change
format version 1 silently.

## Troubleshooting

`scenario package not found` means the `world.scenario_package` path cannot be resolved
from the current directory or repository root. Run commands from `SwarmWorld`, or use
an absolute package path.

`must define every stable slot` means a catalog omitted an internal compatibility
slot. Redefine its public meaning even when the new scenario does not actively place
that terrain, resource, or facility.

`scenario package differs from the recorded trace` means files changed after the run.
Restore the recorded package version or analyze snapshot integrity without claiming
current-code deterministic replay.

An empty technology cycle usually means agents cannot reach one default-recipe
resource or a workspace within the chosen horizon. Use a smaller map, longer horizon,
or revise deposit geometry; do not hide the failure in analysis.

Material equations and artifact services are normalized research surrogates. They are
appropriate for controlled SwarmWorld experiments, not engineering certification,
mine planning, hazard forecasting, or physical metallurgy design.

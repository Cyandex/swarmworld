# SwarmWorld scenario format version 1

Use this reference when creating or changing a declarative world. Confirm the current implementation against `src/biofoundry/scenarios.py` and the complete human contract in `worlds/ashen_realms/README.md` before relying on it.

## Package entry point

`scenario.yaml` is the only package entry point. It declares package identity and routes to data documents inside the same directory:

```yaml
format_version: 1
id: my_world
name: My World
version: 0.1.0
description: >-
  A concise scientific description.
agent_prompt: prompts/agent_instructions.md
documents:
  terrains: terrain.yaml
  geometry: geometry.yaml
  fields: fields.yaml
  resources: resources.yaml
  operations: operations.yaml
  facilities: facilities.yaml
  artifacts: artifacts.yaml
  missions: missions.yaml
  disturbances: disturbances.yaml
  rendering: rendering.yaml
  analysis: analysis.yaml
```

All referenced files and the agent prompt must remain inside the package. The package hash covers `scenario.yaml`, referenced documents, and the referenced agent prompt. It does not cover the package README, run configs, unreferenced researcher guidance, or generated reference results; configuration and other provenance are recorded separately in traces.

## Fixed catalog slots

Define every stable slot exactly once and give every public `id` a unique value.

```text
Terrains (9):
DEEP_WATER, TIDAL, MEADOW, FUNGAL_GROVE, CHITIN_GARDEN,
CELLULOSE_FIELD, MINERAL_SPRING, FOUNDRY, TEST_FIELD

Resources (9 including NONE):
NONE, KELP, SHELL, FUNGUS, CHITIN, CELLULOSE, MINERAL, WATER, CATALYST

Facilities (7 including NONE):
NONE, FERMENTER, WASHER, PRESS, ALIGNER, TESTER, ARCHIVE

Operations (10):
WASH, GRIND, FERMENT, ALKALINE_TREAT, MINERALIZE,
ALIGN, WEAVE, PRESS, DRY, COAT
```

Public IDs may have entirely different meanings. The stable slots preserve numeric observations, action schemas, traces, replay, and current clients.

Terrain records require `slot`, `id`, `name`, `walkable`, `color`, and `height`. Terrain or facility records may declare supported public operation IDs.

## Geometry

`geometry.yaml` declares a public base terrain and ordered features. Later features overwrite earlier terrain. Coordinates are normalized from zero to one and scale to the configured grid size.

Supported shapes:

```text
circle:    center, radius
ring:      center, inner, outer
ellipse:   center, radius [rx, ry]
rectangle: bounds [x0, y0, x1, y1]
ridge:     start, end, width
noise:     probability
```

Use only deterministic shape definitions. Ensure that walkable space is connected enough for the mission, required deposits are reachable, and facilities do not depend on automatic relocation to repair a poor layout.

## Environmental fields

Every scenario must define these exact compatibility field IDs:

```text
temperature
water_availability
ground_stability
toxic_gas
solar
```

The shared wire channels interpret `water_availability` as moisture, `ground_stability` as nutrients, and `toxic_gas` as contamination. These are compatibility aliases, not scientific equivalences. Add any number of extra fields such as elevation, wind, radiation, pressure, lava depth, or ash load.

Each field can define a bounded range, initial constant, x/y gradients, radial sources, seeded noise, terrain offsets, four-neighbor diffusion, decay, per-terrain sources, and an optional deterministic day cycle. The package cannot supply arbitrary update code.

## Resources and processes

Each resource has six normalized composition values. Give those dimensions domain-specific names in `analysis.yaml`. Deposits use supported geometry plus capacity, capacity variation, initial fill, and optional regrowth. Ordered deposits may overwrite resource identity where they overlap.

Every operation maps one fixed operation slot to a public ID and bounded additive effects on:

```text
purity
porosity
grain_refinement
work_hardening
thermal_control
oxidation
quality
```

Recipes contain one to twelve steps with intensity in `[0, 1]`. They use public resource and operation IDs at the package boundary. The engine normalizes them to stable slots and computes shared surrogate mass yield and material properties. These equations are controlled game-level research surrogates, not physical engineering predictions.

The current artifact catalog has one stable `MATERIAL_SYSTEM` type. Artifact geometry aliases can rename:

```text
layers, surface_area, channel_density, anisotropy,
branching, connectivity, curvature, modularity
```

The six stable service slots are:

```text
water_capture, remediation, structural_support,
adaptive_regulation, self_maintenance, ecological_support
```

Map all six to public IDs in `rendering.yaml`. Some common analysis figures may retain the stable labels; interpret results through the scenario descriptor.

## Facilities, mission, and hazards

Map all facility slots, including `NONE`. Placements use normalized coordinates. A recipe is legal only when the union of current terrain and facility capabilities supports all recipe steps.

`missions.yaml` defines the human/agent objective, milestones, and a complete default recipe. The generic `ScenarioIndustryPolicy` follows that recipe without hard-coded scenario vocabulary. Make every required input reachable within the intended horizon and provide a reachable workspace capable of every default step.

Each disturbance defines a public ID, name, normalized radius, field deltas, and optional terrain transformation with a threshold. The run configuration controls disturbance interval and intensity. Disturbances repeat deterministically in declared order during the live run; held-out evaluation may permute their schedule by evaluation seed.

## Boundaries that require engine work

Format version 1 does not express:

- additional terrain, resource, facility, operation, artifact, or service slots;
- non-rectangular topology, diagonal or non-cardinal movement;
- terrain-dependent movement cost or speed;
- arbitrary field equations or package-authored Python;
- multiple fundamentally different artifact physics backends;
- arbitrary browser meshes or executable rendering code;
- changing observation/action/replay wire shapes.

Treat these as a versioned engine/protocol design task rather than silently approximating them in YAML.

# Game design

Current implementation: engine revision 9.

## Player fantasy

The viewer watches a society invent a material culture from biological evidence.
There is no assigned chief scientist and no fixed role prompt. Spatial experience,
failed experiments, communication, and persistent artifacts create differentiation.

## World loop

1. Observe organisms, materials, environmental gradients, and existing artifacts.
2. Record a local observation or publish an insight.
3. Harvest and trade biological feedstocks.
4. Compose a typed processing recipe.
5. Fabricate and test a material batch.
6. Build an artifact and install a behavior program.
7. Let the artifact operate, fail, grow, heal, or alter the environment.
8. Inspect outcomes, fork successful designs, and repair or replace failures.

## Actions

| Action | Status | Executable semantics |
|---|---|---|
| `WAIT` | complete | no mutation |
| `MOVE` | complete | simultaneous cardinal movement |
| `INSPECT` | complete | structured local measurement and citable private record |
| `HARVEST` | complete | sample present local matter subject to capacity; no named target |
| `DEPOSIT` | complete | transfer typed mass to shared station inventory |
| `OPERATE` | complete | consume a scaled recipe and stage a microbatch |
| `TEST` | complete | reveal deterministic surrogate properties and utility |
| `PROPOSE_RECIPE` | complete | validate and retain a candidate recipe |
| `BUILD` | complete | consume inputs, create batch and programmed artifact |
| `REPAIR` | complete | bounded local artifact repair |
| `DISMANTLE` | complete | retire once and recover bounded, health-adjusted feedstock mass when enabled |
| `COMMUNICATE` | complete | local broadcast or addressed delivery with stable reply threads |
| `PUBLISH` | complete | append to cultural archive |
| `DEPOSIT_INSIGHT` | complete | persistent spatial notebook marker |
| `WRITE_PROGRAM` | complete | validate and install bounded artifact DSL |
| `FORK_PROGRAM` | complete | require a known content-addressed parent and record exact instruction diff lineage |
| `CLAIM_TASK` | complete | persistent task board claim |
| `TEACH` | complete | transfer cited records/program skills to broadcast or addressed nearby recipients |
| `TRADE` | complete | transfer material to an addressed adjacent agent and optionally fulfill a request |
| `COMBINE_DESIGN` | complete | require distinct authors, retain lineage, and inherit explicitly published empirical material grounding |
| `METABOLIZE` | optional treatment | consume personal organic feedstock to restore bounded action energy |

The experiment uses deterministic measurements. Instrument noise and calibration are
reserved for a later robustness study so the first causal mechanism remains auditable.

## Material programs

Recipes transform feedstocks into a `MaterialBatch`. Example:

```yaml
inputs:
  - resource: KELP
    mass: 1.2
  - resource: SHELL
    mass: 0.4
steps:
  - operation: WASH
    intensity: 0.7
  - operation: ALIGN
    intensity: 0.75
  - operation: PRESS
    intensity: 0.45
output_form: patterned_membrane
design_principles:
  - directional_wettability
```

Composition and structure determine normalized properties. These equations are
transparent game surrogates, not materials-prediction claims.

## Open invention specification

There are no predefined artifact technologies. `artifact=1` means only “build a
material system.” The agent supplies an unrestricted name, falsifiable claimed
function, architecture description, biological inspirations, predicted effects, and a
continuous geometry vector covering layers, surface area, channels, anisotropy,
branching, connectivity, curvature, and modularity. The recipe decides composition and
processing; the program decides bounded later-tick actions. Text labels never alter
physics.

The browser procedurally derives each visible object from the same geometry vector, so
agent decisions produce different shapes without a renderer-side catalog.

## Artifact behavior programs

Artifact programs run every tick. Example:

```json
{
  "name": "humidity_gated_collector",
  "instructions": [
    {"op": "gt", "dest": "r0", "a": "moisture", "b": 0.55},
    {"op": "mul", "dest": "r1", "a": "r0", "b": "permeability"},
    {"op": "mul", "dest": "r2", "a": "r1", "b": 0.02},
    {"op": "collect_water", "value": "r2"}
  ]
}
```

Supported sensors currently include moisture, nutrients, temperature, solar exposure,
contamination, artifact state, and relevant material properties. Supported
actuators include water collection, growth, healing, opening, contamination
reduction, and bounded signal emission.

## Memory and knowledge

- Working memory is short and private.
- Episodic memory is bounded and private.
- Notebook records are persistent private scientific artifacts.
- Deposited insights are spatially persistent.
- Published insights enter a shared append-only archive.
- Material batches, programs, and artifacts retain contributor provenance.
- Programs have immutable instruction hashes, explicit parent-child diffs, and private
  skill indices that distinguish passive observation from measured verification.
- Retrieval uses deterministic normalized BM25-style lexical scoring and logs selected
  and subsequently cited record IDs; no embedding model is required at current memory
  capacities.
- Each agent has a sparse empirical map of cells it has actually seen, grounded material
  sites, and its own extraction outcomes.

The empirical map stores evidence rather than beliefs: it does not label unseen matter
as absent, rank candidate materials, recommend actions, or expose authoritative world
state. An executable recipe can use only matter the acting agent has observed, possesses,
or can access in the conserved shared depot. Agents may still publish speculative
hypotheses in prose and decide for themselves how much negative evidence is persuasive.

Experiments can independently remove these layers.

Science configurations hide global resource landmarks by default. Resources become
visible only within the local observation radius, and inspection records can preserve
coordinates for later communication. Laboratory visibility is a separate switch:
`public_infrastructure_map: true` exposes only a neutral station base map, while the
50-agent technology-ecology profile sets it to `false` so agents must discover its
distributed laboratories.

## Visual grammar

- Teal pulses: delivered information.
- Amber markers: deposited insights.
- Visible fluids: actual resource or process flows.
- Artifact animation: actual program outputs and physical state.
- Scientific overlay: moisture and contamination fields.
- Intelligence overlay: communication and knowledge deposits.

Visual effects must never imply a causal interaction absent from the event log.

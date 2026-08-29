# Fifty-agent technological ecosystem experiment

Status: current live-society and representative-run mechanism protocol. The stronger
paper comparison is the population-scaling design in
[FLAGSHIP_EXPERIMENT.md](FLAGSHIP_EXPERIMENT.md).

## Scientific question

Can a decentralized society of homogeneous LLM agents create a cumulative,
functionally coupled ecology of material technologies through exploration, experiment,
inheritance, and environmental niche construction?

The evolving units are recipes, physical artifacts, public evidence, and executable
artifact programs. LLM weights do not change. “Co-evolution” therefore means cultural
and technological co-evolution, not genetic or gradient-based learning by the agents.

## Open-ended society

The treatment contains 50 identically prompted `gpt-5.6-luna` agents. No agent receives
an assigned role, target recipe, technology catalog, biological analogy, route, or
division of labor. Each agent sees local environmental measurements, nearby agents and
artifacts, its private empirical notebook, and a bounded retrieval of public evidence.

Agents may explore, harvest, trade, test recipes, publish evidence, build arbitrarily
named material systems, author bounded executable programs, fork programs they have
learned, repair or dismantle artifacts, and respond to requests. Artifacts keep
executing after their authors move away and change conserved environmental fields.

The engine defines available matter, unit operations, sensor/actuator primitives, and
conservation laws. It does not define technologies. An invention is an agent-created
composition of recipe, processing sequence, architecture, geometry, and executable
behavior.

## Why agents should move

The earlier fixed world concentrated every laboratory in one foundry, making central
clustering rational. The 50-agent profile changes the opportunity landscape without
controlling behavior:

- six neutral laboratory stations are distributed across the procedural landscape;
- raw materials remain spatially localized and exhaustible;
- no global resource or infrastructure map is shown;
- disturbances are spatially heterogeneous;
- communication, teaching, trade, inspection, and program manipulation remain local;
- agents choose every movement action themselves.

No dispersal reward, patrol route, forced teleport, or anti-crowding rule is used.
Movement is measured by successful path length, cells visited, interval relocation,
regional crowding, and the time evolution of technological sites. Regional crowding is
the probability that two agents occupy the same one of 48 map-scale neighborhoods; it
detects a population packed around one facility even when every agent occupies a
different cell. Transient gathering is allowed—the question is whether agents relocate
as material, processing, evidence, and artifact opportunities change.

## Causal definition of collaboration

An artifact is collaborative only if at least two agents enter its recorded causal
construction through material contribution, cited evidence ancestry, or authored
program history. Dialogue volume and artifact prose do not count.

Program descent is content-addressed. A `FORK_PROGRAM` edge records the exact parent,
child, instruction difference, author, artifact, and tick. A cross-agent fork requires
an agent-authored parent and a child authored by a different simulation agent. Built-in
or otherwise non-agent-authored starter programs are excluded from both the numerator
and denominator. If the same program has several independently recorded authors, the
metric conservatively requires the child author to differ from all of them.

## Causal definition of ecological co-evolution

After the discovery episode, agents are frozen. The intact technological society then
runs autonomously for 288 ticks—one complete cycle of the world's native contamination,
drought, and storm disturbances. The evaluator records balanced service coverage:

\[
R(t)=\overline{s(t)}\left(\frac{1}{2}+
\frac{1}{2}\frac{\min s(t)}{\overline{s(t)}}\right),
\qquad
\mathrm{AUC}=\frac{1}{T}\sum_t R(t).
\]

The assay is repeated from the exact same final state after removing each artifact in
turn. If removing artifact A changes the autonomous performance of artifact B, the
engine records a directed ecological interaction `A -> B`. Positive effects indicate
support; negative effects indicate inhibition. Two positive edges in opposite
directions constitute reciprocal technological dependence. The same deterministic
future weather is used in every knockout.

This is simpler than inventing a separate collection of evaluator stresses: it asks
whether the society survives the world it actually inhabited, using the disturbances
already present in the substrate.

## Experimental conditions

The primary treatment is the full cultural society. The primary mechanism control is
`no-communication`, which removes public communication, publishing, and teaching while
preserving physical stigmergy through the shared environment and locally observable
artifacts. This is a stringent control: both populations contain 50 agents and can
still affect one another physically.

Use identical paired procedural seeds and the same 1,200-tick horizon. The production
profile leaves both model calls and embodied action attempts unlimited so that agents
remain capable of acting throughout the episode. Report realized model calls, tokens,
and action attempts for every condition. If a later confirmatory study introduces a
matched-compute cap, preregister it above the maximum observed pilot demand and treat
any exhausted episode as censored rather than as an ordinary completed run.

Primary endpoint:

- agent-free intact ecosystem resilience AUC.

Mechanism endpoints:

- collaborative-artifact fraction;
- cross-agent program-fork fraction;
- mean absolute artifact-knockout coupling;
- reciprocal-support pair count;
- maximum habitat-level knockout effect;
- cumulative performance improvements;
- mobility, interval relocation, and regional crowding.

The independent simulation seed is the statistical replicate. Agents and ticks are
not replicates.

## Run one live 50-agent society

```bash
conda activate PyTorch
export OPENAI_API_KEY=...

biofoundry serve \
  --config configs/openai-gpt-5.6-luna-technology-ecology-50.yaml \
  --record runs/technology-ecology-live-seed-3001.jsonl.gz
```

In a second terminal:

```bash
cd web
npm ci
npm run dev
```

Open the exact URL printed by Vite. The browser reconnects to the BioFoundry server at
`ws://127.0.0.1:8765/ws`.

The authoritative server retains an adaptively sampled, bounded set of world
checkpoints and streams them incrementally to every newly connected visualizer. It
always retains tick zero while reducing sampling density for very long episodes, and
keeps the complete compact society-dynamics history. To revisit every exact transition
after the run, open the completed trace with `biofoundry playback`.

## Paired pilot: separate commands

These commands remain useful for a focused N=50 mechanism pilot. For the current paper
pilot—including N=4/N=16 scaling and the independent-search envelope—use the single
command in [FLAGSHIP_EXPERIMENT.md](FLAGSHIP_EXPERIMENT.md).

Full cultural society:

```bash
biofoundry research-study \
  --config configs/openai-gpt-5.6-luna-technology-ecology-50.yaml \
  --policy llm --conditions full \
  --seeds 3001 3002 3003 3004 3005 \
  --ticks 1200 \
  --output-dir runs/technology-ecology-pilot-full
```

Stigmergy-only communication ablation:

```bash
biofoundry research-study \
  --config configs/openai-gpt-5.6-luna-technology-ecology-50.yaml \
  --policy llm --conditions no-communication \
  --seeds 3001 3002 3003 3004 3005 \
  --ticks 1200 \
  --output-dir runs/technology-ecology-pilot-no-communication
```

Do not select confirmatory seeds until the pilot has frozen prompts, budgets, horizon,
figures, exclusion rules, and the prospective sample size.

## Analysis and figures

Analyze the paired study:

```bash
biofoundry analyze-study \
  runs/technology-ecology-pilot-full/study-summary.json \
  runs/technology-ecology-pilot-no-communication/study-summary.json \
  --labels full no-communication \
  --output-dir runs/technology-ecology-pilot-analysis
```

Run the complete causal assay and render figures for one recorded society:

```bash
biofoundry ecosystem-report \
  runs/technology-ecology-pilot-full/llm-full-seed-3001.jsonl.gz \
  --output-dir runs/technology-ecology-seed-3001-report
```

The report produces JSON plus PNG/PDF figures for:

1. technological succession, movement, autonomous resilience, and service niches;
2. executable program phylogeny and differential cultural adoption;
3. the artifact-by-artifact knockout interaction matrix and habitat-level importance;
4. a spatial map of laboratories, agents, artifacts, and strongest causal couplings.

Paired seed-level endpoint estimates and confidence intervals come from
`biofoundry analyze-study`, not from the single-trace ecosystem report.

## Interpretation

Evidence for a cumulative technological society requires measured physical function
and attributable inheritance. Evidence for co-evolution additionally requires causal
ecological interactions under artifact removal. A dense citation graph, frequent
messages, agent clustering, or imaginative artifact names alone is insufficient.

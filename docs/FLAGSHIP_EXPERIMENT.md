# Flagship experiment: cultural gain in open-ended materials invention

## Question

Does decentralized cultural interaction let a population of homogeneous LLM agents
discover better executable material technologies than the best result from the same
number of isolated LLM searches?

This is deliberately one experiment, not a benchmark suite. Agents receive no catalog
of biological analogies, recipes, roles, or preferred designs. They use pretrained
knowledge, local measurements, private memory, and the action affordances exposed by
the treatment. All performance and resilience endpoints come from simulator state,
never an LLM judge.

## Falsifiable claim

At matched population size, duration, model, and total decision opportunity, the full
society has positive **cultural gain**:

```
discovery-frontier AUC(full society)
    - discovery-frontier AUC(best of N isolated searches) > 0
```

The evidence becomes substantially stronger if the gain:

1. increases with population size;
2. is reduced by removing communication or executable program inheritance;
3. survives held-out disturbance schedules after every agent is frozen; and
4. is accompanied by causally closed lineages linking multi-agent evidence,
   executable programs, physical artifacts, and measured frontier gains.

A null or negative cultural-gain curve is a valid result. More artifacts, messages,
or elaborate prose do not rescue the claim.

## Treatments

| Condition | Scientific meaning |
|---|---|
| `full` | Local observation, public culture, stigmergy, trade, and executable program descent |
| `no-communication` | Removes messaging, publishing, teaching, task claims, trade, and publication-dependent composition from both the prompt schema and engine |
| `no-program-forking` | Preserves communication but removes executable descent-with-modification |
| `independent-search` | Runs N isolated one-agent worlds and takes the predeclared endpoint-wise maximum |

The independent-search control is intentionally strong: every isolated agent receives
an entire copy of the world. Isolated member i begins at exactly the same position and
receives decisions at exactly the same ticks as swarm agent i. Each reported primary
endpoint is the maximum over those N members, making this a conservative control for
the collective. The retained
`selected_member` is specifically the discovery-frontier-AUC winner; endpoint winners
are recorded separately. If explicit budgets are configured, the runner partitions
their totals across isolated members so the aggregate cannot exceed the swarm budget.
With unlimited budgets, both sides receive the same physical-action capacity and fixed
macroturn opportunities per agent. Event-triggered replanning is disabled in the
confirmatory runner so communication cannot silently purchase additional model calls.
Use nonzero temperature for this control so repeated model calls are genuine
independent samples.

For fixed-size scaling, initial positions are nested: the N=4 coordinates are the
first four coordinates of N=16 and N=50 for the same seed. The full permutation of
walkable cells is drawn during every initialization, so changing N does not alter the
downstream simulator RNG state. The checked-in profile still uses a seed-dependent
procedural layout; `--world-scaling fixed` means that its 72 × 54 dimensions do not
grow with N. These are experimental controls, not agent hints.

## Primary design

- Population sizes: 4, 16, and 50.
- Discovery horizon: 1,200 ticks.
- Confirmatory world seeds: 4101–4108. Do not tune on these seeds.
- Held-out evaluator seeds: 9101–9116. These affect only post-discovery stress
  locations and stress order.
- Primary world scaling: fixed 72 × 54 dimensions with a paired procedural layout for
  each seed. This asks whether a larger society uses the same finite ecology more
  effectively.
- Robustness design: repeat with `constant-area-per-agent`; the runner records the
  realized dimensions.
- Statistical unit: one independently generated episode seed, never an agent or tick.
- Paired inference: compare conditions only within the same population size and world
  seed.

### Primary endpoints

1. Time-normalized immutable lifetime discovery-frontier AUC.
2. Best final-state artifact performance under the program installed at the end of
   discovery.
3. Mean frozen-society resilience AUC over held-out disturbance schedules.

The analysis also reports lifetime best performance and the peak attained by the
currently installed program. Installing a new program closes the prior program epoch
but cannot erase historical performance. This prevents reprogramming from making a
discovery disappear while keeping historical and final-state claims distinct.

### Mechanism endpoints

1. Fraction of frontier gain with causally closed multi-agent evidence and executable
   program ancestry.
2. Cross-agent program-fork fraction and maximum lineage depth.
3. Collaborative artifact fraction.
4. Ecological coupling and reciprocal support in artifact knockout assays.

The action schema, exact condition capabilities, provider response IDs, model usage,
configuration hash, prompt hash, schema hash, engine revision, Git commit, and dirty
state are written to the trace or study manifest.

## Commands

Activate the existing environment and install the checked-out code:

```bash
conda activate PyTorch
cd SwarmWorld
python -m pip install -e ".[dev,analysis]"
export OPENAI_API_KEY="..."
```

Run the small end-to-end pilot first. These are development seeds and must not be
included in confirmatory statistics:

```bash
biofoundry research-study \
  --config configs/openai-gpt-5.6-luna-technology-ecology-50.yaml \
  --policy llm \
  --conditions full no-communication no-program-forking independent-search \
  --population-sizes 4 16 \
  --world-scaling fixed \
  --decision-schedule fixed \
  --seeds 3001 3002 \
  --ticks 400 \
  --held-out-evaluation-seeds 9001 9002 9003 9004 \
  --output-dir runs/flagship-validity-pilot
```

Run the frozen confirmatory experiment only after inspecting the pilot for mechanical
failures:

```bash
biofoundry research-study \
  --config configs/openai-gpt-5.6-luna-technology-ecology-50.yaml \
  --policy llm \
  --conditions full no-communication no-program-forking independent-search \
  --population-sizes 4 16 50 \
  --world-scaling fixed \
  --decision-schedule fixed \
  --seeds 4101 4102 4103 4104 4105 4106 4107 4108 \
  --ticks 1200 \
  --held-out-evaluation-seeds \
    9101 9102 9103 9104 9105 9106 9107 9108 \
    9109 9110 9111 9112 9113 9114 9115 9116 \
  --output-dir runs/flagship-confirmatory
```

Generate JSON, seed-level CSV files, and publication PNG/PDF figures:

```bash
biofoundry analyze-study \
  runs/flagship-confirmatory/study-summary.json \
  --output-dir runs/flagship-confirmatory/analysis
```

The principal scaling figure is `analysis/population-scaling.pdf`. The numerical
cultural-gain table is `analysis/cultural-gain.csv`. Standard condition endpoints are
in `analysis/condition-endpoints.csv`, and population-resolved endpoints are in
`analysis/population-scaling.csv`. `analysis/scaling-slopes.csv` reports bootstrap
confidence intervals for the change in each endpoint—and in cultural gain—per doubling
of population size. The figure overlays the independent seed observations beneath the
means and 95% intervals. `analysis/compute-audit.csv` reports decision opportunities,
provider calls, tokens, non-WAIT actions, and admitted action attempts for every run.
`analysis/design-audit.csv` verifies decision opportunity, initial positions, decision
phases, and engine revision within every population-by-seed cell. Token use remains a
separately reported cost because social information can make prompts longer even when
model-call opportunity is identical. For legacy summaries created before explicit
initial/final position fields, the analyzer verifies starting positions against each
trace's authoritative tick-zero snapshot and reports that provenance in the audit.

For one predeclared representative seed, render the expensive artifact-by-artifact
knockout and lineage figures:

```bash
biofoundry ecosystem-report \
  runs/flagship-confirmatory/llm-full-n-50-seed-4101.jsonl.gz \
  --output-dir runs/flagship-confirmatory/representative-4101 \
  --evaluation-seeds 9101 9102 9103 9104 9105 9106 9107 9108
```

## Interpretation guardrails

- Positive scaling relative to one agent is not swarm advantage; it may be parallel
  search. The independent-search envelope is the key baseline.
- A full-versus-no-communication difference identifies the contribution of explicit
  culture, while residual no-communication performance may reflect physical
  stigmergy.
- A full-versus-no-program-forking difference tests executable inheritance, not all
  communication.
- The held-out assay measures persistence and ecological function after agent activity
  stops. It is not additional discovery time.
- Do not change prompts, thresholds, physics, or endpoints after viewing confirmatory
  seeds. Any change begins a new experiment version and new seed block.
- Do not combine engine-revision-8 pilot estimates with engine-revision-9 results;
  revision 9 repairs spatial matching, decision scheduling, and lifetime accounting.

## Why this remains compatible with the bitter lesson

The experiment adds compute, memory, interaction, and objective feedback—not a
hand-authored materials strategy. Capability-aware schemas remove impossible actions
but do not recommend possible ones. Scaling success must come from the model's general
knowledge and search over experience. The held-out evaluator is causally downstream
of discovery and cannot teach the agents. Domain equations define the environment and
measurement apparatus; they do not encode which invention the society should produce.
Matched spawns and fixed decision times remove experimental confounds without adding
roles, recipes, reward shaping, search heuristics, resource hints, or success-directed
fallback behavior.

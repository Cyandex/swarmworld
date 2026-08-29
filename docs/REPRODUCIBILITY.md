# Reproducibility

Current paper-study boundary: engine revision 9. See
[FLAGSHIP_EXPERIMENT.md](FLAGSHIP_EXPERIMENT.md) for the active protocol and
[QUICKSTART.md](QUICKSTART.md) for copy-paste commands.

## Existing development environment

```bash
conda activate PyTorch
git clone https://github.com/lamm-mit/SwarmWorld.git
cd SwarmWorld
python -m pip install -e ".[dev,analysis]"
```

For a clean independent environment:

```bash
conda env create -f environment.yml
conda activate biofoundry-world
```

## Integrity checks

```bash
biofoundry doctor --config configs/demo.yaml
python -m ruff check src tests game
python -m pytest tests game/tests
```

## Deterministic episode

```bash
biofoundry simulate \
  --config configs/demo.yaml \
  --seed 17 \
  --ticks 256 \
  --output runs/demo-seed17.jsonl

biofoundry replay runs/demo-seed17.jsonl
```

The trace header records an explicit authoritative `engine_revision`. The final state
digest should repeat on the same engine revision and dependency set. Legacy or
mismatched traces receive snapshot-integrity verification only; counterfactual replay
is disabled because changed dynamics would invalidate the intervention. Floating
point changes across platforms can still matter for very long field simulations;
confirmatory runs should pin environment manifests and platform.

The fixed generator and procedural generator are both reset-deterministic. Procedural
snapshots include the sampled coast, biome, and laboratory manifest plus the accepted
generation-attempt number. Compare the manifest as well as the seed when diagnosing a
run.

LLM traces also record the exact agents granted each macroturn. Replay applies those
policy-side scheduling acknowledgements before the corresponding world actions; older
traces fall back to the deterministic scheduler encoded in their header. This is needed
because event-triggered replanning is part of authoritative state even though it is not
a physical action. Recorded private `research_state` notebook writes are likewise
reapplied before the action step; they are policy-side authoritative memory mutations,
not reconstructible from the action payload alone.

## Live renderer

```bash
biofoundry serve --config configs/demo.yaml --record runs/live.jsonl
```

For the primary web observatory, start a second terminal:

```bash
cd web
npm ci
npm run dev
```

Open the local URL printed by Vite. The client connects to the same authoritative
WebSocket and never changes simulation equations locally.

Open `godot/project.godot` with Godot 4 and run. Keys `1`, `2`, and `3` select
world, scientific, and collective-intelligence views. `-` and `+` control zoom.
Click an agent to inspect it. `Space` pauses, `N` single-steps, `,` and `.` change
speed, arrow keys move the selected agent, and `H`, `E`, `R`, and `X` submit
harvest, inspect, repair, and dismantle actions.

For automated visual regression, set a project-specific capture path before a
bounded renderer launch. Do not add `--headless`: Godot's dummy renderer cannot
produce the viewport image used by the capture hook.

```bash
BIOFOUNDRY_CAPTURE="$PWD/runs/godot-smoke.png" \
  godot --path godot --quit-after 120
```

The current project was parsed and run end-to-end with Godot 4.7.1 on Apple Silicon.
The client connected to the Python WebSocket, received snapshots, rendered through
OpenGL-on-Metal compatibility mode, and produced
`docs/assets/biofoundry-procedural-mvp.png` through the capture hook.

## LLM service

Set `llm.enabled: true` only after starting a compatible endpoint and verifying the
model name. API keys are read from the environment variable named by
`llm.api_key_env`; they must never appear in YAML, prompts, logs, or commits.

No LLM is downloaded or contacted by the default configuration.

For the optional procedural/metabolism condition:

```bash
biofoundry simulate \
  --config configs/openai-gpt-5.6-luna-procedural-metabolism.yaml \
  --policy llm --ticks 512 --model-call-budget 256 \
  --output runs/luna-procedural-metabolism-seed17.jsonl.gz

biofoundry replay runs/luna-procedural-metabolism-seed17.jsonl.gz
```

The compressed trace is lossless. Repeated prompt fragments are restored by
`read_records`, and mortality/replacement plus inherited program IDs enter the state
digest. Do not pool these runs with the default non-metabolic condition.

## Collective-science study

The current workflow, seed split, conditions, endpoints, and commands are in
[FLAGSHIP_EXPERIMENT.md](FLAGSHIP_EXPERIMENT.md). A study writes:

- `study-manifest.json` before the first episode;
- one exact JSONL action/event/model trace per seed and condition;
- `study-summary.json` with functional, mechanism, specialization, and token metrics;
- analysis JSON/CSV plus PNG/PDF figures after `biofoundry analyze-study`.
- optional contributor-removal reports from `biofoundry counterfactual-replay`.
- per-run planning, rejection, exploration, task-allocation, batch/test completion,
  and citation diagnostics from `biofoundry trace-report`.

For a citation-to-artifact mechanism audit of any downloaded or newly produced trace:

```bash
biofoundry trace-report PATH/TO/TRACE.jsonl.gz \
  --output runs/trace-report.json \
  --figures-dir runs/trace-figures
```

The figure marks publications that enter artifact ancestry, links cited evidence into
built artifacts, and places the artifact's measured peak field performance beside the
evaluator target. It does not infer quality from citation depth.

For a budgeted mechanism study, set the same `--model-call-budget` and
`--action-attempt-budget` in paired conditions. The current flagship instead leaves
both unset and uses `--decision-schedule fixed`, giving every agent
treatment-invariant opportunities while reporting realized calls, tokens, and actions
in the compute audit. Event-driven studies must be labeled separately because salient
events can change when calls occur. Retryable provider outages advance neither world
time nor action queues; `provider_outage` records let analysis verify that rule.

The oracle policy may be run for mechanical validation. The LLM study commands never
run during installation or tests and require an explicitly configured provider.

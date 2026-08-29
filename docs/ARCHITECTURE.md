# Architecture

Current implementation: engine revision 9. Revision-specific documents elsewhere in
`docs/` are historical design and audit records unless they explicitly say otherwise.

## Authority boundary

`BioFoundrySimulation` is the only component allowed to mutate scientific state.
PettingZoo, policies, model servers, FastAPI, and Godot submit actions or consume
state; none defines physical outcomes.

```text
policy or LLM server
        |
        v
AgentAction validation
        |
        v
BioFoundrySimulation.step
  |       |         |
  v       v         v
world  materials  artifact VM
        |
        v
events + snapshot + state digest
       / \
      v   v
  JSONL   WebSocket -> Three.js / Godot
```

This boundary makes a recorded action trace replayable without the original model
and prevents presentation timing from changing an experimental result.

## Tick transaction

One call to `step` performs the following deterministic transaction:

1. Replace absent and invalid actions with `WAIT`.
2. Resolve all movement proposals from the same pre-movement state.
3. Apply local actions such as harvesting, communication, publication, building,
   trading, repair, and program installation.
4. Diffuse environmental fields, renew biological resources, and apply any
   seed-deterministic contamination, drought, or storm event.
5. Execute every installed artifact behavior program under a fixed instruction and
   actuation budget.
6. Apply physical artifact growth, collection, damage, healing, and response.
7. In the optional economy treatment, apply passive metabolism, mortality, and
   deterministic replacement with bounded cultural inheritance.
8. Emit tick-stamped events, rewards, metrics, and optional snapshots.

World RNG is created from the episode seed. Rendering and model providers have no
access to it.

## State layout

The population uses structure-of-arrays storage:

```text
x[N], y[N], energy[N], active[N], generation[N], inventory[N, resources]
```

World fields are dense arrays with shape `[height, width]`. Artifact physics uses
preallocated arrays up to the configured artifact limit. Natural-language memory
is bounded and separate from dense physical state.

`EmpiricalKnowledge` is a sparse, deterministic evidence store maintained separately for
each agent. Passive local sensing and extraction outcomes update it during authoritative
simulation steps, so exact action replay reconstructs the same evidence. It stores no
inferred beliefs. The LLM receives a compact view listing only empirically grounded
materials and is free to interpret incomplete coverage itself.

Each agent also receives a private bench-state view. It lists the IDs, authored
recipes, creation ticks, and masses of that agent's pending microbatches. It never
reveals latent properties or utility before `TEST`; afterward, the measured result is
retained in the same private view and notebook. This keeps the experimental loop
observable without turning the evaluator into an oracle.

Action feedback retains both the world result and the agent's own free-form `message`
as private intent. The engine does not interpret that intent or turn it into a role;
preserving it simply prevents a multi-turn agent from forgetting why it moved, sampled,
or approached a station when only the terse action outcome would otherwise survive.

Internal IDs are integers. Human-readable strings are produced at API and log
boundaries. This matters at thousands of agents.

## Agent timescales

An LLM does not issue every step. A macroturn may produce a plan of up to 16 atomic
actions, held in the policy's bounded queue. Between macroturns, the simulator
executes that queue without another model request.

In the scalable LLM profiles, failed preconditions and decision-relevant measured
state transitions—testing, building, salient physical discoveries, and disturbances—
trigger an early macroturn. Fabricating a pending sample does not interrupt the queued
`TEST`, and an unmeasured publication does not erase subsequent work. Incoming messages
enter a structured inbox but wait for the recipient's next scheduled macroturn. The
legacy eager behavior remains available through `science.selective_replanning: false`
and `science.message_interrupts: true`.
The `no-feedback` ablation removes action-result observation and early replanning.
Scheduling remains deterministic, the resulting requests are logged, and an optional
episode-level call budget can prevent a feedback condition from receiving more
inference. The flagship study instead sets `science.decision_schedule: fixed`, which
gives treatment-invariant opportunities and disables event-triggered extra decisions.

Every executed action also preserves the agent-authored `message` as private intent
alongside the authoritative result. This gives a stateless model continuity across
macroturns without assigning a role, route, recipe, or goal in engine code: an agent
can remember that its previous move was meant to reach a station, while failed and
successful consequences remain separately observable.

The semantic observation includes a compact `local_affordances` object: walkable
directions, whether matter is collectable on the current tile, whether the current
tile is a fabrication/depot workspace, whether a private microbatch awaits testing,
and whether local exchange or artifact manipulation is physically possible. These are
facts about action legality, not recommendations. This separation is important for
small local models and for fair model comparisons: scientific choices remain with the
agent, while success no longer depends on inferring simulator rules from prose labels.

Each structured macroturn also contains a private, model-authored `research_state` with
a goal, hypothesis, progress assessment, next observable checkpoint, and collaboration
need. The latest update is retained in the agent's notebook and shown on its next
macroturn. The simulator neither supplies nor rewards this content. It provides the
planning/reflection continuity used by persistent-agent environments while leaving
role formation, scientific direction, stopping decisions, and collaboration endogenous.

The state may retain up to sixteen stable `evidence_ids`. These are agent-selected
pointers, not engine-written conclusions. Private retrieval observes which selected
records later appear in these pointers or action provenance and uses that empirical
reuse signal, with bounded exploration, to rank future memories.

With `llm.experience_attention`, the latest self-authored research state is also the
query for bounded public archive, skill, and private-memory retrieval. Full local
physical affordances remain visible, but repeated artifact measurements and cultural
records are rendered through fixed context budgets. This moves scaling from transcript
accumulation to experience selection without embedding a material ontology or workflow.

Sparse empirical map memory treats terrain, resources, and stations symmetrically:
for every directly observed category it retains the nearest seen coordinate and
last-seen tick. In particular, a foundry is not forgotten merely because it is encoded
as terrain rather than as a station. These records contain observations only; they do
not identify which landmark an agent ought to use.

The observation also summarizes the bounded recent action-result window as counts,
successes, failures, and the current same-action streak. This is descriptive
metacognitive input, not an anti-repetition penalty or workflow rule: an agent may keep
sampling when its own hypothesis warrants replication, or revise its research state
when it judges the loop uninformative. `INSPECT` is explicitly a site/environment
measurement; objective processed-material properties enter memory only through `TEST`.

When `science.public_infrastructure_map` is enabled, fixed foundries and stations
appear in a `public_infrastructure` base map. This can prevent laboratory search from
dominating a materials-science experiment while preserving partial observability where
it matters: the map contains no biological resources, environmental measurements,
artifacts, agent locations, hypotheses, or recommended actions. The 50-agent
technology-ecology profile deliberately disables this map so distributed laboratory
discovery remains part of the task. `no-infrastructure-map` is the corresponding
navigation control; `global-landmarks` is the stronger diagnostic that also reveals
distant resources.

## Executable artifacts

The build enum contains only `NONE` and `MATERIAL_SYSTEM`. Scientific identity is not
an engine-side class. Agents author a free-form name, claimed function, architecture,
biological inspirations, predicted effects, continuous geometry, processing recipe,
and behavior program. Prose is provenance only; numeric architecture, measured batch
properties, environment, and program actuation determine physics.

An `ArtifactProgram` contains at most 64 straight-line instructions and uses 16
fixed float registers. Inputs are named sensors; outputs are capability-scoped
actuators. There are no jumps, calls, imports, strings interpreted as code, or
access to the host.

The VM clips each register to `[-4, 4]` and each extensive actuation to a configured
per-tick cap. A program can only request named world actuators; it cannot mutate
arrays directly. With `physics.closed_artifact_fluxes: true`, moisture and nutrients
are normalized per-cell inventories with explicit capacities: `collect_water` removes
exactly the amount added to artifact storage; growth, autonomous healing, manual repair,
and nutrient signaling share a bounded artifact reserve; and remediation removes no
more contamination than is present. Natural spring recharge, resource regrowth,
disturbances, metabolism, and artifact transfers are separately accumulated in a flux
ledger. These remain transparent game-level surrogate units, not SI-calibrated material
physics. The checked-in OpenAI experiment profiles enable closure explicitly; the
dataclass default is off for backward-compatible programmatic runs.

Programs have a content-derived `program_id` computed only from canonical instructions.
`FORK_PROGRAM` requires a program the acting agent authored, observed, was taught, or
inherited, and at least one changed instruction. The registry stores exact parent/child
IDs and diffs. A private skill-library entry becomes `verified` only after the agent
performs an `INSPECT` that records measured artifact services; verification is evidence
presence, not an evaluator pass label. Teaching can transfer those records without
granting matter or bypassing grounding.

## World generation and economy treatments

`world.generator: fixed` retains the controlled analytic layout used for paired causal
ablations. `procedural` samples coast, biome geometry, laboratory position, resources,
and disturbance centers only during reset. A deterministic generate-and-retry loop
enforces nonempty biomes; resource/station reachability is guaranteed by construction.
The sampled manifest is recorded in snapshots and authoritative state. Fixed-world and
procedural-world results must be reported separately.

The economy is disabled by default. When enabled, every attempted action has an
explicit energy cost and agents can convert personally held organic feedstock through
`METABOLIZE`. Mortality and replacement are independent switches. Replacement preserves
the stable population slot, increments `generation`, resets private memory, and may
inherit a bounded set of measured program records. This is cultural turnover—not a
claim of biological evolution or reproduction of LLM weights.

Addressed communication, program forking, the verified skill library, retrieval
diagnostics, closed artifact fluxes, dismantle recovery, and trace prompt deduplication
also default off in raw `GameConfig()`. Checked-in treatment profiles opt into each
mechanism explicitly, which makes their controls reconstructible from every trace
header.

## Provider boundary

`OpenAICompatibleProvider` sends non-streaming Responses API requests to
`/v1/responses` on a configured endpoint. The same client can target OpenAI, vLLM,
mistral.rs, or another Responses-compatible service. Swarm memory remains explicit
in each request, so provider-side response storage is disabled. The game never
imports a serving backend, and no provider is contacted when `llm.enabled` is
false.

The LLM policy accepts only the bounded root object
`{"research_state": {...}, "plan": [...]}`. Malformed
output becomes a logged model error and a safe `WAIT`; it never reaches `eval`,
`exec`, a shell, or the artifact VM unchecked. With `llm.structured_output: true`,
the Responses API request uses `text.format.type: json_schema` and `strict: true`.
The closed schema covers every action field, stable artifact target ID, material recipe,
invention specification,
insight, and valid artifact-VM instruction form, and host code validates the decoded response against
the same schema again. State-dependent preconditions are still validated
authoritatively by the simulator.

Material names present in the JSON Schema form a representational vocabulary, not a
global availability catalog. Runtime grounding is deliberately uniform across all
materials: independent recipe inputs require direct observation or conserved personal
possession; combined designs may inherit only material evidence in their exact cited
publication lineage. Shared-depot mass satisfies fabrication, not epistemic grounding.
`HARVEST` uses the neutral resource field and samples local matter, preventing a
requested label from overriding physical state.

`TEACH` resolves records from the teacher's own bounded memory as well as the public
archive. It can therefore transfer a private observation or test result to nearby
notebooks without globally publishing it; it does not confer direct empirical grounding
or bypass recipe validation.

`COMMUNICATE`, `TEACH`, and `TRADE` accept a locally validated `target_agent_id`.
Messages receive stable IDs; `reply_to` forms response edges, while teaching and trade
record fulfillment edges. Empty targets preserve local broadcast semantics. The
`no-addressing` treatment disables targeted delivery without disabling communication.

With `science.request_tracking`, any successful non-`WAIT` action may cite a visible
message in `reply_to`. The exact action then closes that request with a causal
fulfillment edge. The engine validates reachability and success but never interprets
the request prose or decides which action should satisfy it. A bounded fulfillment
receipt and any resulting notebook record are returned to the requester; physical
grounding is not transferred implicitly.

Retryable transport and service failures are outside the scientific world. With
`llm.freeze_on_provider_outage`, planning is transactional: a transient failure in any
scheduled request preserves every queued action and scheduling bit, records a
`provider_outage`, and retries without advancing a tick. Invalid model output and
non-retryable provider errors remain logged failed decisions and cannot deadlock a run.

## PettingZoo boundary

`BioFoundryParallelEnv` wraps the simulation. Its fixed action space includes a
bounded text payload that can carry a JSON recipe, insight, message, or program.
Numeric MARL policies can ignore that payload. PettingZoo is an adapter for multi-agent
research compatibility; it is not the communication layer or renderer.

## Renderer boundary

The live server broadcasts versioned snapshots and recent event deltas. Godot draws
the newest authoritative state and may interpolate between frames. Dropping a
visual frame does not drop a simulator tick. Renderer clients can submit bounded
pause, step, speed, and one-tick agent commands; only the Python simulation validates
and applies them.

## Collective-science authority

`ResearchMission` maintains a shared depot, task claims, candidate recipes,
microbatches, objective test results, and milestone state. It evaluates proposal
utility authoritatively but never exposes the utility equation to agents. Epistemic
authors, physical feedstock contributors, recipe lineage, test identity, and artifact
program history are stored separately; citing another agent cannot fabricate a material
contribution.

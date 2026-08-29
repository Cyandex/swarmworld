# Human play: an embedded human agent in a live society

- Status: design accepted; implementation phased below
- Location: all game code lives in the standalone [`game/`](../game/README.md) package
- Constraint: **strictly additive** — no existing engine, server, client, config, or
  test file is modified. The game imports `biofoundry` as a library.

This document specifies the human-playable game mode of SwarmWorld. A human joins a
live society as one embodied agent among LLM agents, under the same physics, the same
action contract, and the same information limits. The Python engine remains the sole
authority; the game client is a renderer plus an input source, exactly like any policy.

## 1. Design principles

1. **The human is an agent, not a god.** The player controls exactly one agent slot,
   receives that agent's `semantic_observation`, and submits `AgentAction` values that
   pass the identical validation, capability gating, preconditions, and energy costs as
   model-authored actions. No global visibility, no extra affordances.
2. **Existing code is never changed.** `BioFoundrySimulation.step` already accepts
   partial action mappings and defaults absent agents to `WAIT`, so a mixed human/AI
   population needs no engine change. Everything new — server, pacing, client,
   configs, tests — lives under `game/`, which imports engine classes and wraps them
   by composition (a delegation proxy and a small policy subclass; no monkey-patching).
3. **Human sessions are honest data.** The canonical trace stays byte-compatible with
   every existing replay and analysis tool; human provenance is recorded in a sidecar
   JSONL next to the trace. Human sessions must never be pooled with autonomous study
   conditions. In exchange, they inherit exact deterministic replay, playback, and
   counterfactual artifact knockouts for free.
4. **LLM latency must never read as lag.** The pacing model below makes model latency
   visible as other agents "thinking", never as unresponsive controls.

## 2. Pacing: decision windows with a timeout

One mechanism covers real-time, prompted, and turn-based play.

Each tick, the game loop checks the human input queue before stepping:

- If an action is queued, it is applied this tick.
- If the queue is empty and a decision window is open, the loop waits up to
  `input_timeout_seconds` for input. If the timer lapses, the human agent `WAIT`s and
  the world continues.
- If `require_response: true`, the wait is indefinite: the world blocks until the
  human acts (turn-based play).

Game configuration (a `game:` section in the game's own YAML profiles under
`game/configs/`; the remainder of the file is a standard biofoundry configuration and
is passed to `biofoundry.config.load_config` untouched):

```yaml
game:
  human_agent: agent_000000   # the slot a joining player controls
  decision_interval: 1        # open a decision window every k ticks
  input_timeout_seconds: 0    # 0 = never wait; T > 0 = wait up to T per window
  require_response: false     # true = block until the human acts (turn-based)
  queue_limit: 4              # buffered inputs applied on subsequent ticks
```

Notes:

- Wall-clock waiting never enters the engine; replay determinism is unaffected because
  only the recorded per-tick `actions` matter.
- With `input_timeout_seconds: 0` the game is fully real-time: the human acts whenever
  they like and their agent waits otherwise, symmetric with an LLM agent whose plan
  queue is empty.
- A future "matched schedule" fairness mode can restrict human decision windows to the
  same staggered macroturn cadence as LLM agents and require plans rather than single
  actions; this is deliberately out of scope for v1.

## 3. Game server (Phase 0) — `game/server/`

A standalone FastAPI/WebSocket server that reuses engine components by import:
`BioFoundrySimulation`, `EventRecorder`, the policy classes, and the broadcast hub.
Wire protocol stays compatible with protocol v1, so the existing Three.js Observatory
can attach to a game session as a spectator.

1. **Sticky possession.** A `HumanSlot` holds a bounded input queue for the configured
   agent. WebSocket commands: `join` (claim the slot; returns the agent id),
   `release`, and `human_action` (queued, not last-write-wins). One possessing
   connection per slot; other clients remain observers.
2. **Policy wrapping, not patching.** The human slot is removed from LLM scheduling by
   passing the policy a lightweight simulation *view* whose `scheduled_macro_agents()`
   filters the human agent (everything else delegates to the real simulation), so the
   slot consumes no model calls and receives no machine-authored research state. A
   small `LLMPolicy` subclass skips the plan-queue pop for the human slot so a human
   action never discards a model plan. `src/biofoundry` is not edited.
3. **Decision windows.** The game loop implements section 2. New outbound packets
   (additive; unknown types are ignored by existing clients):
   - `decision_prompt {agent, tick, deadline_ms, affordances}` where `affordances` is
     the `local_affordances` block of the agent's observation;
   - `action_ack {request_id, accepted, detail}` for submitted human actions;
   - each `frame` carries the possessed agent's full `semantic_observation` under
     `human_observation` while a player is joined.
4. **Provenance sidecar.** The canonical trace is recorded exactly as `serve` records
   it. A sidecar `<trace>.human.jsonl` records the session settings and one line per
   human action `{tick, agent, action, request_id, wall_time}` plus control commands
   (join, pause, speed). Analysis can join sidecar and trace by tick; replay tooling
   never sees a schema change.
5. **Input robustness.** The game server validates command payload shapes before
   constructing `AgentAction`, acks malformed input with an error instead of dropping
   the connection, and rejects actions for inactive agents with a clear detail.
6. **Tests.** `game/tests/` covers: join/release lifecycle, exclusion from macroturn
   scheduling, queue semantics, timeout and require-response pacing (with the scripted
   policy), malformed payloads, sidecar contents, and unchanged canonical-trace
   replayability of a human session. Run with `python -m pytest game/tests`.

Known defects observed in the existing `src/biofoundry/server.py` input path (a
malformed `manual_action` payload can drop the connection; inactive agents ack like
success; control commands are unrecorded) are deliberately **not** fixed there under
the additive-only constraint; the game server simply does not inherit them.

## 4. The game client (Phase 1) — `game/client/`

A dependency-free, no-build vanilla-JavaScript canvas client (ES modules) served
statically by the game server itself: start one Python process, open the printed URL,
play. No npm step. The reconnecting-WebSocket logic and palette constants are vendored
from `web/src` as JavaScript copies with provenance headers; `web/` itself is not
modified.

v1 surface:

- top-down tile renderer with camera follow, zoom, and interpolated agent motion
  (presentation-only, per `docs/PROTOCOL.md`);
- keyboard control: arrows/WASD move, `E` inspect, `H` harvest, `Space` wait/pass,
  plus a contextual action menu populated from `local_affordances`;
- HUD: energy, inventory total, current tile readout, decision-window countdown,
  "agent thinking" indicators during macroturns;
- nearby-events ticker (filtered to the player's information radius) and a chat panel:
  `COMMUNICATE` composer, received messages, and public-archive reader;
- join flow: connect, claim the configured slot, or spectate.

Verbs with structured payloads (`BUILD`, `OPERATE`, `WRITE_PROGRAM`,
`COMBINE_DESIGN`, `PROPOSE_RECIPE`, `TEACH`, `TRADE`, `PUBLISH`) are Phase 2:

- recipe builder that greys out empirically ungrounded materials (the grounding rule
  surfaces as a discovery gate rather than a rejection string);
- geometry sliders with a live procedural artifact preview;
- a block-style editor for the bounded program DSL (sense, arithmetic, actuators),
  mirroring the same enums as `structured_output.py`;
- publish/teach/trade dialogs and the task board as a quest log.

## 5. Scoring and the research reading (Phase 3)

The game surfaces measurements the engine already makes; none of them feed back into
physics, prompts, or validation:

- personal discovery frontier versus the society's frontier;
- cultural influence: forks of the player's programs, citations of the player's
  published evidence, adoption of the player's recipes, teach/trade counts;
- mission milestones and disturbance survival as quest/wave framing.

Research use: a recorded human session is a human-in-the-swarm condition. Exact
replay, held-out frozen-society evaluation, and artifact knockouts apply unchanged, so
"what did the human contribute causally" is answerable with existing tooling. Human
sessions are labeled by the sidecar and must be excluded from autonomous study pools.

## 6. Milestones

1. **M0 — player slot and pacing** (`game/server/` + configs + tests; zero changes to
   existing files). Verifiable headlessly with the scripted policy.
2. **M1 — canvas client MVP**: move, inspect, harvest, deposit, wait, chat, camera
   follow, decision countdown; playable end-to-end with scripted agents (no API key)
   and with LLM societies in prompted or turn-based mode.
3. **M2 — deep-verb UIs**: recipes, geometry, programs, social verbs; the full 21-verb
   contract reachable by a human.
4. **M3 — polish and science**: background LLM planning for low-timeout real-time
   play, influence HUD, matched-schedule fairness mode, human-influence analysis
   script, optional purpose-built world package.

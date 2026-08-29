# Web observatory architecture

## Scientific boundary

The web application is a renderer, instrument panel, and bounded command client. It
does not implement world equations, validate actions, execute artifact programs, or
call agent models. All scientific state transitions occur in the Python simulator.

```text
LLM or scripted policies
          |
          v
authoritative Python simulation -----> JSONL scientific record
          |
          | protocol-v1 snapshots, events, control state
          v
FastAPI WebSocket
          |
          v
React state adapter
    |          |             |
    v          v             v
Three.js    ECharts      Cytoscape.js
world       dynamics     provenance
```

The client sends only `toggle_pause`, `set_paused`, `step`, `set_speed`, and
`manual_action`. A human action passes through the same `AgentAction` validation as
other policies and overrides the selected agent for one tick.

## Rendering model

The browser treats integer grid locations as simulation coordinates. When a new
snapshot arrives, agent meshes interpolate toward their authoritative target
locations; interpolation is never sent back to the server.

One instanced mesh is used for each high-cardinality layer:

- terrain tiles;
- visible biological resources;
- agent bodies;
- agent visors.

Stations and artifacts remain individual interactive objects because their counts
are smaller. Artifact shapes are generated from agent-authored continuous architecture
parameters rather than semantic types. The default orthographic camera
creates a readable 2.5D scientific map while retaining 3D lighting, geometry,
occlusion, and orbit controls. The landscape is a continuous seeded heightfield
derived from the authoritative biome grid. Animated water, biome-specific vegetation,
minerals, spores, lighting, and shadows provide depth without introducing unlogged
scientific state.

World-space selection resolves clicks back to authoritative integer grid cells. When
several artifacts share a cell, the renderer applies deterministic, presentation-only
radial offsets so each remains selectable; inspectors always report the unmodified
simulation coordinate. A searchable object index provides a second, camera-independent
selection path.

## Observability

Every displayed scientific quantity is derived from the snapshot or event stream:

- environmental overlays use flattened world fields;
- positions, energy, inventory totals, and actions use agent structure-of-arrays;
- artifact state uses identity, authorship, geometry, health, maturity, current and
  peak services, performance, storage, program, and causal-parent fields;
- communication lines use `message_delivered` events;
- short-lived ground rings encode each agent's current non-WAIT action family, while
  recent build, program-install, repair, and dismantle events produce typed agent-to-
  artifact arcs; these marks fade with event age and never persist as world state;
- the provenance graph uses insight causal parents, artifacts, recent messages,
  content-addressed program nodes, and compact metadata for private observations or
  tests that a public outcome actually cites. It renders distinct `observed`, `tested`,
  `authored`, `built`, `forked`, and `installed` relations. Repeated recent messages
  are aggregated by directed pair rather than drawn as parallel lines. Selecting a
  node reveals the corresponding recorded evidence, instructions, authors, tick, or
  artifact state;
  its topology remains stable across ticks while selected-node measurements update in
  place, preventing live scientific values from forcing a graph re-layout;
- the report uses transparent summary statistics and performs no model call.

Every principal view also has a paper-export path. The world and the active Inspect,
Activity, Lineage, or Report panel are regenerated directly from the displayed
authoritative snapshot as standard SVG geometry on white, then rasterized into a
matching 2× PNG. The lineage export uses deterministic semantic lanes—actors,
evidence, insights, programs, and artifacts—and a separate recent-communication
summary rather than capturing the interactive Cytoscape canvas. A second lineage
export reads the current Cytoscape force-layout coordinates and regenerates the same
cluster/component organization as white-background vector SVG plus PNG. Its positions
are explicitly presentation-only and may vary across browser layout runs; scientific
nodes and typed relations remain authoritative. Export generation is presentation-only:
it does not call an agent model, query hidden state, or modify the simulation.

`causal_evidence` is an observer projection, not a new shared-memory channel. The
server emits compact metadata only for private observations, tests, or recipes whose
stable IDs are already cited by a public record, deposited insight, or artifact
provenance. Uncited private memory remains absent from the presentation snapshot and
therefore unavailable to both the browser and other agents.

The server also maintains an online, compact society-dynamics series. Its points
contain no world arrays or private agent state. The initial WebSocket packet hydrates
the complete series and each live frame appends one point, so a browser restart does
not erase aggregate history. Outcome, normalized society, and action-composition
views are different projections of this same authoritative series.

Ambient spores, lighting, interpolation, shadows, and camera effects are clearly
presentation-only and do not imply an unlogged state variable.

## Failure recovery

The server emits two-second heartbeats independently of simulation ticks, including
while an LLM macroturn is awaiting a provider. The browser uses each frame or heartbeat
as evidence of liveness. Failed handshakes, socket errors without a close event,
connection timeouts, and silent twelve-second stalls all enter a bounded exponential-
backoff reconnect loop. Generation guards prevent an obsolete socket callback from
closing a newer connection.

In development, `npm run dev` supervises the Vite child process and restarts it one
second after an unexpected exit. `npm run dev:once` is available when process
supervision is not wanted. Ctrl-C shuts the supervisor down normally.

## Replay behavior

The live world timeline retains an adaptively sampled, bounded presentation history in
server and browser memory. It is intended for immediate spatial inspection and is
distinct from the complete compact society-dynamics series.

For a completed trace, `biofoundry playback TRACE` runs a separate offline server with
the same HTTP and WebSocket surface. It reads the recorded configuration, model-side
memory commits, macroturn schedule, and executed actions; reconstructs every state
transition without model calls; verifies every stored authoritative digest; and sends
compact keyframes plus the complete dynamics history to each connecting browser. The
transport exposes exact-tick seek, pause, step, restart, and rates up to 64×. Playback
is refused when the trace and installed simulator have different engine revisions.

The video control uses the browser's screen-capture permission rather than adding an
unlogged render loop. The user selects the current tab, preserving Three.js animation,
activity overlays, charts, and panel interactions in one recording. Offline recording
restarts the playhead at tick zero and stops automatically at completion; the browser
downloads MP4 when H.264 MediaRecorder support is available and otherwise WebM. Video
pixels are a presentation artifact, while the JSONL trace remains the scientific
record. `biofoundry replay TRACE` remains the nonvisual integrity-check command.

## Scaling roadmap

Protocol v1 caps presentation agents at 2,048 and sends full snapshots. The renderer
already uses instancing, but scaling beyond this limit should add:

1. periodic keyframes plus binary state deltas;
2. server-side spatial aggregation and level-of-detail packets;
3. selected-agent trajectory queries instead of universal histories;
4. density fields for distant cohorts;
5. Web Worker decoding and optional OffscreenCanvas rendering;
6. server-computed graph summaries for large provenance networks.

These are transport and presentation changes. They must not modify agent decisions or
the authoritative simulation update order.

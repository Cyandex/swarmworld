# Renderer and replay protocol

- Protocol version: `1`
- Current authoritative engine revision: `9`

## Live connection

The Python server exposes:

- `GET /health`
- `GET /snapshot`
- `GET /history`
- `WS /ws`

The default address is `http://127.0.0.1:8765`. Scientific state remains
server-authoritative, but the WebSocket accepts bounded human control commands.

## Snapshot packet

```json
{
  "type": "snapshot",
  "snapshot": {
    "protocol": 1,
    "tick": 0,
    "seed": 17,
    "world": {},
    "agents": {},
    "artifacts": {},
    "archive": [],
    "insights": [],
    "causal_evidence": [
      {
        "record_id": "observation_0000000042",
        "kind": "observation",
        "author": "agent_000007",
        "tick": 18,
        "summary": "Observed 0.84 cellulose at (12, 9)",
        "causal_parents": []
      }
    ],
    "metrics": {}
  }
}
```

World arrays are flattened row-major with index `y * width + x`. Agent and artifact
fields are structure-of-arrays. `display_count` may be lower than the true `count`.
`causal_evidence` contains compact observer metadata only for otherwise-private records
that are causally cited by public outcomes. It is not part of any agent observation and
does not reveal uncited private memory. Clients must tolerate the field being absent in
older protocol-v1 recordings.

## Frame packet

```json
{
  "type": "frame",
  "snapshot": {},
  "events": [
    {"tick": 18, "kind": "artifact_built", "payload": {}}
  ]
}
```

Version 1 sends a presentation snapshot at each configured render interval. Later
versions may send deltas between periodic snapshots; clients must branch on
`protocol` rather than guessing.

## Society-dynamics history

A newly connected WebSocket client receives a `dynamics_history` object alongside
its initial snapshot. Later frame packets contain only the newest `dynamics_point`.
`GET /history` returns the same compact history object for analysis clients.

```json
{
  "version": 1,
  "sample_interval": 1,
  "role_window": 64,
  "action_names": ["WAIT", "MOVE", "INSPECT"],
  "points": [
    {
      "tick": 0,
      "artifact_utility": 0.0,
      "mean_energy": 1.0,
      "spatial_concentration": 0.0,
      "research_score": 0.0,
      "artifact_count": 0,
      "communication_rate": 0.0,
      "specialization": 0.0,
      "behavioral_diversity": 0.0,
      "action_distribution": [1.0, 0.0, 0.0]
    }
  ]
}
```

The actual action vector contains one position for every protocol action, ordered by
`action_names`. Metrics have the following population-normalized definitions:

- `spatial_concentration`: occupancy Herfindahl index rescaled so unique cells are
  zero and complete co-location is one;
- `communication_rate`: rolling share of communicate, teach, and trade actions;
- `behavioral_diversity`: rolling action entropy divided by the maximum entropy over
  the action vocabulary;
- `specialization`: rolling mutual information between agent identity and action,
  divided by action entropy.

Dynamics are observer measurements only. They do not enter simulation state, reward,
prompts, action validation, or replay digests. Full snapshots are not retained for
this series, so its memory and connection cost scales with ticks and action types,
not world area or population size.

## Control packets

The Godot client can send `toggle_pause`, `set_paused`, `step`, `set_speed`, and
`manual_action` commands. A manual action contains an agent ID and the same
validated action mapping used by PettingZoo. It overrides that agent for one tick;
it never grants direct array, program-VM, filesystem, or model access.

```json
{
  "command": "manual_action",
  "agent": "agent_000003",
  "action": {"verb": 1, "direction": 2}
}
```

The server answers with a `control` packet reporting pause state, speed, model
identity, request count, model-error count, provider-attempt count, and whether a
retryable provider outage currently holds world time fixed.

An offline `biofoundry playback` server uses the same endpoints and adds these optional
control fields: `playback_mode`, `playback_complete`, `max_tick`, and `trace_name`.
It accepts `toggle_pause`, `set_paused`, `step`, `set_speed`, `seek`, and
`play_from_start`; it rejects all world-mutating commands. A seek packet carries
`playback_seek: true`, telling clients to replace the displayed event window rather
than append duplicate events. `GET /replay/manifest` describes keyframe coverage,
exact-seek support, and the number of verified authoritative snapshots.

## JSONL replay

The first record is a header. Action, model-trace, event, and snapshot records follow:

```json
{"type":"header","protocol":1,"metadata":{"engine_revision":9}}
{"type":"actions","tick":0,"macroturn_agents":[],"actions":{"agent_000000":{"verb":1}}}
{"type":"model_trace","tick":0,"agent":"agent_000000","messages":[],"response":"..."}
{"type":"event","tick":0,"kind":"agents_moved","payload":{}}
{"type":"snapshot","digest":"sha256...","state_digest":"sha256...","snapshot":{}}
```

`digest` protects the renderer snapshot. `state_digest` hashes the full authoritative
state, including exact arrays, inventories, memories, provenance, and installed
program bodies. `biofoundry replay` reconstructs the configuration, re-applies every
recorded action without an LLM, and verifies both hashes at every stored snapshot when
the header's explicit `engine_revision` matches the installed engine. Older logs and
logs from a different engine revision remain readable and receive integrity-only
snapshot verification; they are never silently reinterpreted using changed physics.
Counterfactual action removal is refused across an unknown or mismatched revision.
`biofoundry playback` has the same strict revision requirement because its displayed
intermediate states are reconstructed by the installed simulator. It never calls the
recorded model; prompts and full responses are not retained in its in-memory playback
index.

Engine revision 4 adds authoritative field-flux ledgers, program/skill lineage,
addressed message threads, optional economy generations, and procedural generation
manifests. Older revision-3 actions remain parseable, but their state transitions must
not be counterfactually replayed under revision 4.

Engine revision 5 adds agent-directed bounded experience retrieval, durable tested
recipes, generic request-to-action fulfillment, selective replanning, and transactional
provider-outage records. A `provider_outage` record has `world_advanced: false`; it has
no corresponding action transaction and therefore consumes neither a tick nor a queued
motor action.

Engine revision 9 is the current paper-study boundary. It adds nested population spawns,
matched isolated-agent macroturn phases, fixed treatment-invariant study schedules,
immutable lifetime discovery accounting, distinct current-program and final-state
metrics, and per-cell compute/design audits. Revision-8 results remain readable but
must not be pooled with revision-9 estimates.

## Lossless trace compaction

`trace.deduplicate_prompts: true` writes each repeated system/action-manual fragment as
one `trace_template` record and stores references inside later model traces. The shared
`read_records` API hydrates those references, so replay, reports, and prompt inspection
receive the original exact strings. No prompt content is discarded.

`trace.compression: gzip` compresses the complete JSONL byte stream. Readers detect the
gzip header rather than relying on a suffix, although `.jsonl.gz` is recommended.
Snapshots remain full protocol-v1 snapshots: measured traces showed repeated LLM prompt
text, not snapshots, to be the dominant storage cost, so delta snapshots were not
introduced.

## Compatibility policy

- Adding optional fields does not increment the version.
- Changing array meaning, enum values, coordinate convention, or tick ordering does.
- Unknown event kinds must be ignored by renderers but retained by analysis tools.
- A renderer may submit versioned commands but must never mutate scientific state
  locally or imply that a command succeeded before the authoritative frame arrives.

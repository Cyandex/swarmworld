# SwarmWorld human play

Join a live SwarmWorld society as one embodied agent among model-driven agents —
same physics, same 21-verb action contract, same validation, same information
limits. The design is specified in [docs/HUMAN_PLAY.md](../docs/HUMAN_PLAY.md).

**This package is strictly additive.** It imports the authoritative
`biofoundry` engine as a library and modifies nothing under `src/`, `web/`,
`godot/`, `configs/`, or `tests/`. Traces recorded here are byte-compatible
with `biofoundry replay`, `biofoundry playback`, and all existing analysis.

## Play now (no API key)

```bash
conda activate PyTorch
cd SwarmWorld
python game/serve.py --config game/configs/arcade-scripted.yaml
```

Open http://127.0.0.1:8765/play/ and press **Join**. You are `agent_000000` in
a scripted research society. Move with WASD/arrows, `E` inspect, `H` harvest,
`G` deposit, `T` test, `Space` wait, `C` chat, `P` pause, `F` field overlays,
mouse wheel zoom, drag to pan, `M` to toggle camera follow.

## Play inside a live LLM society

```bash
export OPENAI_API_KEY="..."
python game/serve.py --config game/configs/llm-prompted.yaml    # 4 s decision windows
python game/serve.py --config game/configs/llm-turn-based.yaml  # world waits for you
```

Pacing is one mechanism with two knobs (`game:` section of the profile):

- `input_timeout_seconds: 0` — real time; the world never waits, your agent
  simply WAITs when you have not acted (symmetric with an agent whose plan
  queue is empty).
- `input_timeout_seconds: T` — prompted; each decision window waits up to `T`
  seconds for you, shows a countdown, then continues.
- `require_response: true` — turn-based; the world blocks until you act.

Other clients on the same port are spectators: the Three.js Observatory
(`cd web && npm run dev`) attaches to the same WebSocket and shows the full
analyst view of the society you are playing in.

## What a session records

With recording enabled (the default; disable with `--no-record`):

- `runs/game/<profile>-<timestamp>.jsonl[.gz]` — a **standard** trace. Your
  actions pass the same `AgentAction` validation as model actions, are recorded
  in the same per-tick `actions` records, and replay deterministically:
  `biofoundry playback runs/game/<trace>` works unchanged.
- `runs/game/<trace>.human.jsonl` — the provenance sidecar: session settings,
  join/release, every human action with its tick and request id, and control
  commands. Analysis joins sidecar and trace by tick.

Human sessions are a human-in-the-swarm condition; never pool them with
autonomous study runs.

## Layout

```text
game/
  serve.py        entry point (python game/serve.py --config ...)
  settings.py     game profile loader (game: section + standard config)
  session.py      GameSession: LiveGame subclass with decision-window pacing
  policy.py       HumanAwareLLMPolicy + schedule-filtering simulation view
  human.py        possession slot, input queue, provenance sidecar
  app.py          FastAPI app: /play client, /ws protocol, game commands
  configs/        run profiles (scripted arcade, LLM prompted, LLM turn-based)
  client/         no-build vanilla-JS canvas client served at /play/
  tests/          python -m pytest game/tests
```

## Protocol additions (all additive; unknown types are ignored by old clients)

- commands: `join`, `release`, `human_action {request_id, action}`
- packets: `join_ack`, `release_ack`, `action_ack`, `game_state`,
  `decision_prompt {agent, tick, deadline_ms, affordances}`; `frame` gains
  `game` and `human_observation` keys.

## Verify

```bash
python -m pytest game/tests
```

The suite covers possession, LLM-schedule exclusion (a joined human consumes no
model calls and never discards a model plan), timeout and turn-based pacing,
malformed-input guards, sidecar contents, and digest-verified deterministic
replay of a human session with the stock playback tools.

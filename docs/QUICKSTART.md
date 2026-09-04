# Quickstart and experiment runbook

This is the operational entry point for a fresh SwarmWorld source checkout. The
checked-in profiles are indexed in [configs/README.md](../configs/README.md), and the
separate paper dataset is described in [DATA.md](../DATA.md).

## 1. Install and validate

```bash
git clone https://github.com/lamm-mit/SwarmWorld.git
cd SwarmWorld
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,analysis]"

biofoundry doctor --config configs/demo.yaml
python -m ruff check src tests game
python -m pytest tests game/tests
```

Or create the Conda environment in `environment.yml`. The no-key demo does not
download or contact a model.

For the Observatory:

```bash
cd web
npm ci
npm test
npm run build
cd ..
```

## 2. Choose a run mode

| Goal | Command | Main output |
|---|---|---|
| Mechanics smoke test | `biofoundry simulate` | exact trace |
| Watch/interact with a society | `biofoundry serve` | exact trace plus WebSocket |
| Run paired conditions | `biofoundry research-study` | study directory |
| Watch a completed trace | `biofoundry playback` | no new scientific data |
| Verify an existing trace | `biofoundry replay` | integrity result |
| Diagnose one trace | `biofoundry trace-report` | JSON and optional figures |
| Analyze a completed study | `biofoundry analyze-study` | CSV/JSON/figures |

`research-study` is intentionally headless. To observe an exact completed replicate,
open its trace later with `playback`.

## 3. Run and replay without an API key

```bash
biofoundry simulate \
  --config configs/demo.yaml \
  --policy scripted \
  --seed 17 \
  --ticks 128 \
  --output runs/demo-seed17.jsonl

biofoundry replay runs/demo-seed17.jsonl
```

## 4. Watch a live society

Terminal 1:

```bash
biofoundry serve \
  --config configs/demo.yaml \
  --record runs/live-demo.jsonl
```

Terminal 2:

```bash
cd web
npm run dev
```

Open the URL printed by Vite. The Observatory displays authoritative Python state; it
does not determine simulation outcomes.

## 5. Run language-model agents

SwarmWorld accepts an OpenAI-compatible Responses endpoint. For the hosted example:

```bash
export OPENAI_API_KEY="..."

biofoundry serve \
  --config configs/openai-gpt-5.6-luna-technology-ecology-50.yaml \
  --record runs/technology-ecology-live.jsonl.gz
```

For open-weight serving, see
[LOCAL_MODEL_SERVING.md](LOCAL_MODEL_SERVING.md). Credentials must remain in
environment variables, never YAML files, prompts, traces, or commits.

## 6. Run a controlled mechanics study

```bash
biofoundry research-study \
  --config configs/demo.yaml \
  --policy scripted \
  --conditions full no-communication independent-search \
  --population-sizes 4 16 \
  --seeds 3001 3002 \
  --ticks 128 \
  --held-out-evaluation-seeds 9001 9002 \
  --output-dir runs/example-study

biofoundry analyze-study \
  runs/example-study/study-summary.json \
  --output-dir runs/example-study/analysis
```

This command checks mechanics; it is not a reproduction of the paper's complete
compute-intensive protocol. Protocol guardrails are in
[FLAGSHIP_EXPERIMENT.md](FLAGSHIP_EXPERIMENT.md). Exact completed paper traces,
manifests, tables, and figures are in the external dataset.

For a worked analysis of this study—including verified example values, trace
diagnostics, mobility, counterfactuals, dossiers, and the paper-data workflow—continue
with [ANALYSIS.md](ANALYSIS.md).

## 7. Replay released paper data

Download the dataset described in [DATA.md](../DATA.md), then run:

```bash
biofoundry replay PATH/TO/TRACE.jsonl.gz
biofoundry playback PATH/TO/TRACE.jsonl.gz
```

Playback starts paused and makes no model or API calls. Start the Observatory in a
second terminal to inspect and seek through the trace.

## 8. Human play

```bash
python game/serve.py --config game/configs/arcade-scripted.yaml
```

Open `http://127.0.0.1:8765/play/`. Human actions pass through the same authoritative
validation and trace format as agent actions. See [game/README.md](../game/README.md).

## Reproducibility rules

- The independent simulation seed is the statistical unit.
- Save every seed, including provider failures and zero-artifact episodes.
- Freeze code, configuration, prompt, schemas, endpoints, and exclusion rules before
  confirmatory seeds are opened.
- Do not edit or pull code while a publication run is active.
- Report model/provider identity and platform with the manifest.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for trace and engine details.

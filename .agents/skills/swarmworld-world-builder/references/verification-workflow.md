# Verification and handoff workflow

Use commands from the SwarmWorld repository root. Replace `<world_id>`, config paths, seed lists, and horizons with the requested design. Keep outputs isolated under `runs/<world_id>/`.

## Static validation

```bash
python -m pip install -e ".[dev,analysis]"

biofoundry world validate worlds/<world_id>
biofoundry world describe worlds/<world_id>
biofoundry doctor --config worlds/<world_id>/configs/demo.yaml
```

Inspect the resolved descriptor. Confirm identity/version, package hash, complete catalogs, public IDs, fields, service aliases, property aliases, mission, and default recipe.

## Required world tests

Base a new test module on `tests/test_scenarios.py`. Test meaningful behavior rather than copied text:

- absence of `scenario_package` still selects the unchanged legacy world;
- package validation and stable catalogs;
- same seed gives the same initial and post-step state digests;
- every required resource and facility is reachable;
- declared fields stay in range;
- disturbances occur at expected ticks and use public scenario meanings;
- the default recipe validates and executes at a capable workspace;
- public resource and operation IDs appear in semantic observations and structured-output enums;
- live/manual actions normalize public IDs correctly;
- the trace records scenario identity and package hash;
- replay rejects a runtime-defining package change;
- the renderer accepts all catalogs and extra fields.

Run the focused tests first, then proportionate regressions:

```bash
pytest -q tests/test_scenarios.py tests/test_<world_id>.py
pytest -q
ruff check src tests

cd web
npm ci
npm test
npm run build
cd ..
```

If the new world uses no engine or web changes, the focused scenario tests plus existing regressions may be sufficient during iteration; run the full relevant suite before final handoff.

## Smoke run and replay

Use a small run to detect unreachable inputs, invalid recipes, empty technology cycles, or broken field dynamics quickly:

```bash
mkdir -p runs/<world_id>/smoke/analysis

biofoundry simulate \
  --config worlds/<world_id>/configs/demo.yaml \
  --agents 4 \
  --ticks 32 \
  --policy scripted \
  --output runs/<world_id>/smoke/trace.jsonl

biofoundry replay \
  runs/<world_id>/smoke/trace.jsonl \
  > runs/<world_id>/smoke/analysis/replay-verification.json

biofoundry trace-report \
  runs/<world_id>/smoke/trace.jsonl \
  --output runs/<world_id>/smoke/analysis/trace-report.json \
  --figures-dir runs/<world_id>/smoke/analysis/trace-figures
```

Do not hide an empty technology cycle in analysis. Determine whether the horizon is too short, the map is too large, a resource is unreachable, the workspace lacks capabilities, or the scripted policy cannot complete the default recipe.

## Reference run and optional ecosystem assay

After the smoke run passes, run the requested reference scale. A conventional baseline is 24 agents for 100 ticks, but do not impose it when the user specifies another design.

```bash
mkdir -p runs/<world_id>/reference/analysis

biofoundry simulate \
  --config worlds/<world_id>/configs/demo.yaml \
  --agents 24 \
  --ticks 100 \
  --policy scripted \
  --output runs/<world_id>/reference/trace.jsonl

biofoundry replay \
  runs/<world_id>/reference/trace.jsonl \
  > runs/<world_id>/reference/analysis/replay-verification.json

biofoundry trace-report \
  runs/<world_id>/reference/trace.jsonl \
  --output runs/<world_id>/reference/analysis/trace-report.json \
  --figures-dir runs/<world_id>/reference/analysis/trace-figures

biofoundry ecosystem-report \
  runs/<world_id>/reference/trace.jsonl \
  --output-dir runs/<world_id>/reference/analysis/ecosystem \
  --horizon 100 \
  --evaluation-seeds 2701 2702 2703 2704 2705 2706 2707 2708
```

`trace-report` is the principal single-run analysis. The ecosystem report freezes final artifacts and evaluates them without continued agent learning. Interpret stable service slots using the scenario aliases.

For independent replicates, use `research-study` and analyze its study summary; do not use `analyze-study` on one trace. The seed is the unit of replication.

## Browser verification

Replay a completed trace without model calls:

```bash
biofoundry playback runs/<world_id>/reference/trace.jsonl
```

In another terminal:

```bash
cd web
npm ci
npm run dev
```

Open the URL printed by Vite. Inspect terrain names/colors/heights, resources, facilities, extra field selectors, cell details, events, artifacts, and timeline seeking. Run `biofoundry serve --config worlds/<world_id>/configs/demo.yaml` instead when live interaction is requested.

## Handoff standard

The new package README should include:

- world concept and scientific objective;
- terrain, resources, fields, operations, facilities, hazards, and services;
- exact install, validate, simulate, replay, playback, and analysis commands;
- isolated output layout;
- a reference result tied to package version, package hash, engine revision, config, and seed;
- scenario-specific interpretation rules and limitations;
- instructions for creating another world;
- troubleshooting for package paths, missing slots, hash mismatch, and empty technology cycles.

Do not retain copied Ashen Realms reference results or claims. Report what the new trace actually did, including failures, low performance, lack of diversity, or lack of collaboration.

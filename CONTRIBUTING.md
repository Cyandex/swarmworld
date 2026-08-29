# Contributing to SwarmWorld

Contributions that improve correctness, reproducibility, documentation, tests, or
clearly scoped research capabilities are welcome.

## Before opening a change

1. Open an issue for substantial behavioral or protocol changes so scientific
   compatibility can be discussed first.
2. Keep the authoritative simulator deterministic for a fixed configuration, seed,
   engine revision, and dependency environment.
3. Treat changes to dynamics, action schemas, scheduling, prompts, metrics, and trace
   semantics as scientific changes. Update the engine revision and reproducibility
   documentation when compatibility changes.
4. Do not commit study traces, downloaded datasets, generated figures, model weights,
   credentials, caches, or local environment files.

## Local checks

Install the development dependencies and run:

```bash
python -m ruff check src tests game
python -m pytest tests game/tests

cd web
npm ci
npm test
npm run build
cd ..

python scripts/verify_release.py
```

Add or update regression tests for behavioral changes. A pull request should explain
the scientific effect, compatibility implications, test coverage, and any change to
trace or configuration formats.

## Code and documentation style

- Keep the Python command and package name `biofoundry` for compatibility.
- Keep physical outcomes in the authoritative Python simulator; renderers are views,
  not alternate simulation engines.
- Prefer explicit configuration and recorded provenance over hidden defaults.
- Use environment-variable references for credentials and never include secret values.
- Link large reproducibility artifacts from the companion dataset rather than adding
  them to Git.

By contributing, you agree that your contribution is licensed under the repository's
[Apache License 2.0](LICENSE).

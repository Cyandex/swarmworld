.PHONY: install test lint doctor smoke benchmark serve web-install web web-test web-build release-check

install:
	python -m pip install -e ".[dev,analysis]"

test:
	python -m pytest tests game/tests

lint:
	python -m ruff check src tests game

doctor:
	biofoundry doctor --config configs/demo.yaml

smoke:
	biofoundry simulate --config configs/demo.yaml --ticks 128 --output runs/smoke.jsonl

benchmark:
	biofoundry benchmark --config configs/demo.yaml --ticks 100

serve:
	biofoundry serve --config configs/demo.yaml

web-install:
	cd web && npm ci

web:
	cd web && npm run dev

web-test:
	cd web && npm test

web-build:
	cd web && npm run build

release-check: lint test web-test web-build
	python scripts/verify_release.py

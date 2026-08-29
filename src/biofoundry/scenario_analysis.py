"""Scenario-aware analysis layered on top of the common trace contract."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .events import read_records


def analyze_scenario_trace(path: str | Path) -> dict[str, Any]:
    """Summarize package-defined matter, fields, hazards, and process outcomes."""

    source = Path(path)
    header: dict[str, Any] = {}
    descriptor: dict[str, Any] = {}
    extraction: Counter[str] = Counter()
    disturbances: Counter[str] = Counter()
    disturbance_events: list[dict[str, Any]] = []
    operation_usage: Counter[str] = Counter()
    facility_usage: Counter[str] = Counter()
    tested_properties: dict[str, list[float]] = {}
    test_utilities: list[float] = []
    passed_tests = 0
    batch_input_mass = 0.0
    batch_output_mass = 0.0
    field_history: list[dict[str, Any]] = []
    final_world: dict[str, Any] | None = None
    final_artifacts: dict[str, Any] = {}
    event_counts: Counter[str] = Counter()
    final_agents: dict[str, Any] = {}
    for record in read_records(source):
        kind = str(record.get("type", ""))
        if kind == "header":
            header = dict(record.get("metadata", {}))
        elif kind == "event":
            event_kind = str(record.get("kind", "unknown"))
            event_counts[event_kind] += 1
            payload = record.get("payload", {})
            if event_kind == "resource_harvested" and isinstance(payload, dict):
                extraction[str(payload.get("resource", "unknown"))] += float(
                    payload.get("amount", 0.0)
                )
            elif event_kind == "environmental_disturbance" and isinstance(payload, dict):
                disturbance_id = str(payload.get("kind", "unknown"))
                disturbances[disturbance_id] += 1
                disturbance_events.append(
                    {
                        "tick": int(record.get("tick", 0)),
                        "id": disturbance_id,
                        "name": str(payload.get("name", disturbance_id)),
                        "intensity": round(float(payload.get("intensity", 0.0)), 6),
                        "affected_cells": int(payload.get("affected_cells", 0)),
                    }
                )
            elif event_kind == "microbatch_fabricated" and isinstance(payload, dict):
                batch_output_mass += float(payload.get("mass", 0.0))
                recipe = payload.get("source_recipe", {})
                inputs = recipe.get("inputs", []) if isinstance(recipe, dict) else []
                test_scale = float(
                    header.get("config", {}).get("science", {}).get("test_scale", 1.0)
                )
                batch_input_mass += test_scale * sum(
                    float(item.get("mass", 0.0))
                    for item in inputs
                    if isinstance(item, dict)
                )
                steps = recipe.get("steps", []) if isinstance(recipe, dict) else []
                for step in steps:
                    if isinstance(step, dict):
                        operation_usage[str(step.get("operation", "unknown"))] += 1
                workspace = payload.get("workspace", {})
                if isinstance(workspace, dict):
                    facility = str(workspace.get("facility", "NONE"))
                    terrain = str(workspace.get("terrain", "unknown"))
                    facility_usage[facility if facility != "NONE" else terrain] += 1
            elif event_kind == "material_tested" and isinstance(payload, dict):
                test_utilities.append(float(payload.get("material_utility", 0.0)))
                passed_tests += int(bool(payload.get("passes_material_target", False)))
                properties = payload.get("properties", {})
                if isinstance(properties, dict):
                    for name, value in properties.items():
                        tested_properties.setdefault(str(name), []).append(float(value))
        elif kind == "snapshot" and isinstance(record.get("snapshot"), dict):
            snapshot = record["snapshot"]
            world = snapshot.get("world", {})
            if not isinstance(world, dict):
                continue
            candidate = snapshot.get("scenario", world.get("scenario", {}))
            if isinstance(candidate, dict) and candidate:
                descriptor = candidate
            fields = world.get("fields", {})
            if isinstance(fields, dict):
                point = {
                    "tick": int(snapshot.get("tick", 0)),
                    "means": {
                        str(name): round(float(np.mean(values)), 6)
                        for name, values in fields.items()
                        if isinstance(values, list) and values
                    },
                }
                if field_history and field_history[-1]["tick"] == point["tick"]:
                    field_history[-1] = point
                else:
                    field_history.append(point)
            final_world = world
            agents = snapshot.get("agents", {})
            final_agents = agents if isinstance(agents, dict) else {}
            artifacts = snapshot.get("artifacts", {})
            final_artifacts = artifacts if isinstance(artifacts, dict) else {}

    scenario = header.get("scenario", {})
    if not descriptor and isinstance(scenario, dict):
        descriptor = dict(scenario)
    scenario_id = str(descriptor.get("id", scenario.get("id", "legacy_biofoundry")))
    if scenario_id == "legacy_biofoundry" or final_world is None:
        return {
            "applicable": False,
            "scenario_id": scenario_id,
            "reason": "trace uses the legacy biological scenario",
        }

    resources = descriptor.get("catalogs", {}).get("resources", [])
    resource_names = {
        int(index): str(item.get("id", index))
        for index, item in enumerate(resources)
        if isinstance(item, dict)
    }
    named_extraction: dict[str, float] = Counter()
    for raw, amount in extraction.items():
        try:
            name = resource_names.get(int(raw), raw)
        except ValueError:
            name = raw
        named_extraction[name] += float(amount)

    remaining: dict[str, float] = Counter()
    kinds = final_world.get("resource_kind", [])
    masses = final_world.get("resource_mass", [])
    if isinstance(kinds, list) and isinstance(masses, list):
        for raw_kind, mass in zip(kinds, masses, strict=False):
            remaining[resource_names.get(int(raw_kind), str(raw_kind))] += float(mass)
    x_values = final_agents.get("x", [])
    y_values = final_agents.get("y", [])
    distinct_agent_cells = (
        len({(int(x), int(y)) for x, y in zip(x_values, y_values, strict=False)})
        if isinstance(x_values, list) and isinstance(y_values, list)
        else 0
    )
    property_names = descriptor.get("property_names", {})
    property_names = property_names if isinstance(property_names, dict) else {}
    property_summary = {
        str(property_names.get(name, name)): {
            "mean": round(float(np.mean(values)), 6),
            "minimum": round(float(np.min(values)), 6),
            "maximum": round(float(np.max(values)), 6),
        }
        for name, values in sorted(tested_properties.items())
        if values
    }
    service_aliases = {
        str(item.get("slot", "")): str(item.get("id", item.get("slot", "")))
        for item in descriptor.get("services", [])
        if isinstance(item, dict)
    }
    services = final_artifacts.get("services", {})
    services = services if isinstance(services, dict) else {}
    final_services = {
        service_aliases.get(str(name), str(name)): {
            "mean": round(float(np.mean(values)), 6),
            "maximum": round(float(np.max(values)), 6),
        }
        for name, values in services.items()
        if isinstance(values, list) and values
    }
    performances = final_artifacts.get("performance", [])
    artifact_performance = (
        {
            "mean": round(float(np.mean(performances)), 6),
            "maximum": round(float(np.max(performances)), 6),
        }
        if isinstance(performances, list) and performances
        else {"mean": 0.0, "maximum": 0.0}
    )
    return {
        "applicable": True,
        "scenario_id": scenario_id,
        "scenario_version": descriptor.get("version", scenario.get("version")),
        "package_hash": descriptor.get("package_hash", scenario.get("package_hash")),
        "resource_extraction_mass": {
            key: round(value, 6) for key, value in sorted(named_extraction.items())
        },
        "remaining_deposit_mass": {
            key: round(value, 6)
            for key, value in sorted(remaining.items())
            if key != "NONE"
        },
        "disturbances": dict(sorted(disturbances.items())),
        "disturbance_exposure": {
            "events": disturbance_events,
            "total_affected_cell_events": sum(
                int(item["affected_cells"]) for item in disturbance_events
            ),
        },
        "microbatches_fabricated": int(event_counts["microbatch_fabricated"]),
        "process_operation_usage": dict(sorted(operation_usage.items())),
        "facility_utilization": dict(sorted(facility_usage.items())),
        "process_yield": {
            "input_mass": round(batch_input_mass, 6),
            "output_mass": round(batch_output_mass, 6),
            "mass_yield_fraction": (
                round(batch_output_mass / batch_input_mass, 6)
                if batch_input_mass > 1e-12
                else None
            ),
        },
        "batches_tested": int(event_counts["material_tested"]),
        "material_tests": {
            "passed": passed_tests,
            "pass_fraction": (
                round(passed_tests / len(test_utilities), 6) if test_utilities else None
            ),
            "mean_utility": (
                round(float(np.mean(test_utilities)), 6) if test_utilities else None
            ),
            "properties": property_summary,
        },
        "artifacts_built": int(event_counts["artifact_built"]),
        "final_artifact_performance": artifact_performance,
        "final_artifact_services": final_services,
        "field_history": field_history,
        "final_distinct_agent_cells": distinct_agent_cells,
    }

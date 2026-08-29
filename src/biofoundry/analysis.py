"""Seed-level analysis and publication figures for collective-science studies."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .events import read_records
from .program_library import (
    PROGRAM_FORK_AUTHORSHIP_VERSION,
    program_fork_authorship_summary,
)


def _resolve_trace_path(record: dict[str, Any], summary_source: Path) -> Path | None:
    output = record.get("output")
    if not isinstance(output, str) or not output:
        return None
    path = Path(output).expanduser()
    candidates = [path] if path.is_absolute() else [summary_source.parent / path, path]
    if path.is_absolute():
        candidates.append(summary_source.parent / path.name)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _snapshot_positions(snapshot: dict[str, Any]) -> list[list[int]] | None:
    agents = snapshot.get("agents")
    if not isinstance(agents, dict):
        return None
    x_values = agents.get("x")
    y_values = agents.get("y")
    if not isinstance(x_values, list) or not isinstance(y_values, list):
        return None
    if len(x_values) != len(y_values):
        return None
    try:
        return [
            [int(x_value), int(y_value)]
            for x_value, y_value in zip(x_values, y_values, strict=True)
        ]
    except (TypeError, ValueError):
        return None


def _legacy_initial_positions(
    record: dict[str, Any], summary_source: Path
) -> list[list[int]] | None:
    """Recover pre-v2 starting positions from the authoritative tick-zero snapshot."""

    trace = _resolve_trace_path(record, summary_source)
    if trace is None:
        return None
    try:
        for trace_record in read_records(trace):
            if trace_record.get("type") != "snapshot":
                continue
            snapshot = trace_record.get("snapshot")
            if not isinstance(snapshot, dict) or int(snapshot.get("tick", -1)) != 0:
                continue
            return _snapshot_positions(snapshot)
    except (OSError, TypeError, ValueError):
        return None
    return None


def _normalize_program_fork_metrics(
    record: dict[str, Any], summary_source: Path
) -> str:
    """Upgrade legacy fork-authorship metrics from the recorded final snapshot."""

    ecology = record.get("technology_ecology")
    if not isinstance(ecology, dict):
        return "not-recorded"
    version = ecology.get("program_fork_authorship_version", 0)
    if isinstance(version, (int, float)) and version >= PROGRAM_FORK_AUTHORSHIP_VERSION:
        return "summary-v2"

    def invalidate(reason: str) -> str:
        ecology["agent_parent_program_forks"] = None
        ecology["cross_agent_program_forks"] = None
        ecology["cross_agent_program_fork_fraction"] = None
        return reason

    trace = _resolve_trace_path(record, summary_source)
    if trace is None:
        return invalidate("legacy-unverified")

    final_snapshot: dict[str, Any] | None = None
    try:
        for trace_record in read_records(trace):
            if trace_record.get("type") == "snapshot" and isinstance(
                trace_record.get("snapshot"), dict
            ):
                final_snapshot = trace_record["snapshot"]
    except (OSError, TypeError, ValueError):
        return invalidate("legacy-unverified")
    if final_snapshot is None:
        return invalidate("legacy-unverified")

    catalog = final_snapshot.get("program_catalog")
    edges = final_snapshot.get("program_lineage")
    agents = final_snapshot.get("agents", {})
    agent_ids = agents.get("ids") if isinstance(agents, dict) else None
    if not isinstance(catalog, list) or not isinstance(edges, list) or not isinstance(
        agent_ids, list
    ):
        return invalidate("legacy-unverified")
    expected_edges = ecology.get("program_forks")
    if isinstance(expected_edges, (int, float)) and int(expected_edges) != len(edges):
        return invalidate("legacy-unverified-truncated")
    programs = {
        str(item["program_id"]): item
        for item in catalog
        if isinstance(item, dict) and item.get("program_id")
    }
    if any(str(edge.get("parent_program_id", "")) not in programs for edge in edges):
        return invalidate("legacy-unverified-truncated")

    ecology.update(program_fork_authorship_summary(programs, edges, agent_ids))
    return "trace-recovered-v2"


def _normalize_initial_positions(
    record: dict[str, Any], summary_source: Path
) -> tuple[list[list[int]] | None, str]:
    position_version = record.get("position_summary_version", 0)
    if isinstance(position_version, (int, float)) and position_version >= 2:
        value = record.get("initial_positions")
        return (value if isinstance(value, list) else None), "summary-v2"
    if bool(record.get("independent_search")):
        value = record.get("matched_initial_positions", record.get("initial_positions"))
        return (value if isinstance(value, list) else None), "matched-control"
    recovered = _legacy_initial_positions(record, summary_source)
    if recovered is not None:
        return recovered, "trace-tick-0"
    value = record.get("initial_positions")
    return (value if isinstance(value, list) else None), "summary-unverified"


def _nested(record: dict[str, Any], path: str) -> float | None:
    value: Any = record
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return None if value is None else float(value)


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    spread = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def _bootstrap_mean(
    values: list[float], rng: np.random.Generator, resamples: int
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    array = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(array), size=(resamples, len(array)))
    samples = array[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _bootstrap_log2_slope(
    values_by_population: dict[int, list[float]],
    rng: np.random.Generator,
    resamples: int,
) -> tuple[float | None, float | None, float | None]:
    """Estimate change in outcome per population doubling with seed bootstrap."""

    sizes = sorted(size for size, values in values_by_population.items() if values)
    if len(sizes) < 2:
        return None, None, None
    x = np.log2(np.asarray(sizes, dtype=np.float64))
    means = np.asarray(
        [np.mean(values_by_population[size]) for size in sizes], dtype=np.float64
    )
    slope = float(np.polyfit(x, means, 1)[0])
    sampled_columns = []
    for size in sizes:
        values = np.asarray(values_by_population[size], dtype=np.float64)
        indices = rng.integers(0, len(values), size=(int(resamples), len(values)))
        sampled_columns.append(values[indices].mean(axis=1))
    sampled_means = np.column_stack(sampled_columns)
    centered_x = x - float(np.mean(x))
    denominator = float(np.sum(centered_x * centered_x))
    if denominator <= 0.0:
        return slope, None, None
    samples = np.sum(sampled_means * centered_x, axis=1) / denominator
    low, high = np.quantile(samples, [0.025, 0.975])
    return slope, float(low), float(high)


def _nonnegative_yerr(
    means: np.ndarray, lows: np.ndarray, highs: np.ndarray
) -> np.ndarray:
    """Return plotting errors robust to floating-point CI boundary noise."""

    return np.vstack(
        [
            np.maximum(0.0, means - lows),
            np.maximum(0.0, highs - means),
        ]
    )


def _analysis_rng(*parts: object) -> np.random.Generator:
    """Return a stable RNG whose stream is independent of analysis call order."""

    payload = "\x1f".join(str(part) for part in (20260807, *parts)).encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return np.random.default_rng(seed)


def analyze_study(
    summary_path: str | Path | Sequence[str | Path],
    output_dir: str | Path | None = None,
    *,
    resamples: int = 20_000,
    source_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Analyze independent episode seeds; never treat ticks or agents as replicates."""
    sources = (
        [Path(summary_path)]
        if isinstance(summary_path, (str, Path))
        else [Path(item) for item in summary_path]
    )
    if not sources:
        raise ValueError("at least one study summary is required")
    if source_labels is not None and len(source_labels) != len(sources):
        raise ValueError("--labels must provide exactly one label per study summary")

    records: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        loaded = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(loaded, list) or not loaded:
            raise ValueError(
                f"study summary {source} must contain at least one episode record"
            )
        label = None if source_labels is None else str(source_labels[index])
        for record in loaded:
            if not isinstance(record, dict):
                raise ValueError(f"study summary {source} contains a non-object record")
            normalized = dict(record)
            if label is not None:
                normalized["condition"] = label
            positions, position_source = _normalize_initial_positions(
                normalized, source
            )
            normalized["initial_positions"] = positions
            normalized["initial_positions_source"] = position_source
            normalized["program_fork_metric_source"] = (
                _normalize_program_fork_metrics(normalized, source)
            )
            records.append(normalized)

    destination = Path(output_dir) if output_dir else sources[0].parent / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(destination / ".matplotlib"))

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["condition"])].append(record)
    conditions = sorted(groups, key=lambda name: (name != "full", name))
    endpoints: dict[str, tuple[str, bool]] = {
        "ecosystem_resilience_auc": (
            "ecosystem_assay.intact.resilience_auc",
            False,
        ),
        "held_out_resilience_auc": (
            "held_out_generalization.resilience_auc.mean",
            False,
        ),
        "ecological_coupling": (
            "ecosystem_assay.mean_absolute_ecological_coupling",
            False,
        ),
        "reciprocal_support_pairs": (
            "ecosystem_assay.reciprocal_support_pair_count",
            False,
        ),
        "maximum_knockout_effect": (
            "ecosystem_assay.maximum_knockout_effect",
            False,
        ),
        "collaborative_artifact_fraction": (
            "technology_ecology.collaborative_artifact_fraction",
            False,
        ),
        "cross_agent_program_fork_fraction": (
            "technology_ecology.cross_agent_program_fork_fraction",
            False,
        ),
        "causal_closure_gain_fraction": (
            "technology_ecology.causal_closure_gain_fraction",
            False,
        ),
        "mean_distance_traveled": ("mobility.mean_distance_traveled", False),
        "mean_cells_visited": ("mobility.mean_distinct_cells_visited", False),
        "regional_crowding": ("mobility.regional_crowding", False),
        "outcome_success": ("research.outcome_success", True),
        "emergent_success": ("research.emergent_success", True),
        "artifact_performance": ("research.best_artifact_performance", False),
        "current_program_peak_performance": (
            "research.best_current_program_peak_performance",
            False,
        ),
        "final_state_artifact_performance": (
            "research.best_final_state_artifact_performance",
            False,
        ),
        "discovery_frontier_auc": ("discovery_frontier.normalized_auc", False),
        "portfolio_resilience": ("research.portfolio_resilience", False),
        "service_breadth": ("research.service_breadth", False),
        "self_organized_portfolio": ("research.self_organized_portfolio", True),
        "behavioral_novelty": ("research.best_behavioral_novelty", False),
        "validated_inventions": ("research.counts.validated_inventions", False),
        "material_utility": ("research.best_material_utility", False),
        "composition_synergy": ("research.composition_synergy", False),
        "specialization": ("swarm_metrics.excess_normalized_specialization", False),
        "causal_composition": ("swarm_metrics.causal_composition_success", True),
        "program_forks": ("swarm_metrics.program_forks", False),
        "program_lineage_depth": (
            "swarm_metrics.maximum_program_lineage_depth",
            False,
        ),
        "verified_program_adoption": (
            "swarm_metrics.maximum_verified_program_adoption",
            False,
        ),
        "message_reply_fraction": ("swarm_metrics.message_reply_fraction", False),
    }
    analysis: dict[str, Any] = {
        "unit_of_analysis": "independent simulation seed",
        "confidence_intervals": (
            "Wilson 95% for proportions; deterministic seed bootstrap 95% for means"
        ),
        "bootstrap_resamples": resamples,
        "sources": [str(source) for source in sources],
        "source_labels": list(source_labels) if source_labels is not None else None,
        "program_fork_metric_sources": {
            source: sum(
                record.get("program_fork_metric_source") == source
                for record in records
            )
            for source in sorted(
                {
                    str(record.get("program_fork_metric_source", "unknown"))
                    for record in records
                }
            )
        },
        "conditions": {},
        "paired_differences_from_full": {},
        "pairwise_population_contrasts": {},
        "population_scaling": {},
        "scaling_slopes": {},
        "cultural_gain": {},
        "cultural_gain_scaling_slopes": {},
        "held_out_checkpoint_trajectories": {},
    }
    table_rows: list[dict[str, Any]] = []
    for condition in conditions:
        episodes = groups[condition]
        condition_result: dict[str, Any] = {"n": len(episodes), "endpoints": {}}
        for endpoint, (path, binary) in endpoints.items():
            values = [
                value
                for record in episodes
                if (value := _nested(record, path)) is not None
            ]
            mean = float(np.mean(values)) if values else None
            if not values:
                low, high = None, None
            elif binary:
                low, high = _wilson(sum(value >= 0.5 for value in values), len(values))
            else:
                low, high = _bootstrap_mean(
                    values,
                    _analysis_rng("condition", condition, endpoint),
                    resamples,
                )
            estimate = {
                "mean": mean,
                "ci95_low": low,
                "ci95_high": high,
                "n_observed": len(values),
            }
            condition_result["endpoints"][endpoint] = estimate
            table_rows.append(
                {"condition": condition, "n": len(episodes), "endpoint": endpoint, **estimate}
            )
        calls = [float(record.get("model_calls", 0)) for record in episodes]
        errors = [float(record.get("model_errors", 0)) for record in episodes]
        condition_result["model_calls_mean"] = float(np.mean(calls))
        condition_result["model_error_rate"] = sum(errors) / max(1.0, sum(calls))
        analysis["conditions"][condition] = condition_result

    if "full" in groups:
        full_by_seed = {
            (
                int(record.get("population_size", record.get("agents", 0))),
                int(record["seed"]),
            ): record
            for record in groups["full"]
        }
        for condition in conditions:
            if condition == "full":
                continue
            other_by_seed = {
                (
                    int(record.get("population_size", record.get("agents", 0))),
                    int(record["seed"]),
                ): record
                for record in groups[condition]
            }
            shared = sorted(set(full_by_seed) & set(other_by_seed))
            paired: dict[str, Any] = {"n_pairs": len(shared), "endpoints": {}}
            for endpoint, (path, _binary) in endpoints.items():
                paired_values = [
                    (
                        _nested(full_by_seed[seed], path),
                        _nested(other_by_seed[seed], path),
                    )
                    for seed in shared
                ]
                differences = [
                    full - other
                    for full, other in paired_values
                    if full is not None and other is not None
                ]
                low, high = (
                    _bootstrap_mean(
                        differences,
                        _analysis_rng("full-paired", condition, endpoint),
                        resamples,
                    )
                    if differences
                    else (None, None)
                )
                paired["endpoints"][endpoint] = {
                    "full_minus_ablation": (
                        float(np.mean(differences)) if differences else None
                    ),
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_observed_pairs": len(differences),
                }
            analysis["paired_differences_from_full"][condition] = paired

    pairwise_rows: list[dict[str, Any]] = []
    for condition_a, condition_b in combinations(conditions, 2):
        by_key_a = {
            (
                int(record.get("population_size", record.get("agents", 0))),
                int(record["seed"]),
            ): record
            for record in groups[condition_a]
        }
        by_key_b = {
            (
                int(record.get("population_size", record.get("agents", 0))),
                int(record["seed"]),
            ): record
            for record in groups[condition_b]
        }
        shared = sorted(by_key_a.keys() & by_key_b.keys())
        contrast_key = f"{condition_a}-minus-{condition_b}"
        population_results: dict[str, Any] = {}
        for population_size in sorted({key[0] for key in shared}):
            keys = [key for key in shared if key[0] == population_size]
            endpoint_results: dict[str, Any] = {}
            for endpoint, (path, _binary) in endpoints.items():
                observations: list[dict[str, Any]] = []
                for key in keys:
                    value_a = _nested(by_key_a[key], path)
                    value_b = _nested(by_key_b[key], path)
                    if value_a is None or value_b is None:
                        continue
                    observations.append(
                        {"seed": key[1], "difference": value_a - value_b}
                    )
                differences = [
                    float(observation["difference"])
                    for observation in observations
                ]
                low, high = (
                    _bootstrap_mean(
                        differences,
                        _analysis_rng(
                            "pairwise",
                            condition_a,
                            condition_b,
                            population_size,
                            endpoint,
                        ),
                        resamples,
                    )
                    if differences
                    else (None, None)
                )
                estimate = {
                    "condition_a_minus_b": (
                        float(np.mean(differences)) if differences else None
                    ),
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_pairs": len(differences),
                    "paired_values": observations,
                }
                endpoint_results[endpoint] = estimate
                pairwise_rows.append(
                    {
                        "condition_a": condition_a,
                        "condition_b": condition_b,
                        "population_size": population_size,
                        "endpoint": endpoint,
                        "condition_a_minus_b": estimate[
                            "condition_a_minus_b"
                        ],
                        "ci95_low": low,
                        "ci95_high": high,
                        "n_pairs": len(differences),
                    }
                )
            population_results[str(population_size)] = {
                "n_pairs": len(keys),
                "endpoints": endpoint_results,
            }
        analysis["pairwise_population_contrasts"][contrast_key] = {
            "condition_a": condition_a,
            "condition_b": condition_b,
            "populations": population_results,
        }

    scaling_rows: list[dict[str, Any]] = []
    for condition in conditions:
        by_population: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in groups[condition]:
            population_size = int(
                record.get("population_size", record.get("agents", 0))
            )
            by_population[population_size].append(record)
        condition_scaling: dict[str, Any] = {}
        for population_size in sorted(by_population):
            episodes = by_population[population_size]
            estimates: dict[str, Any] = {}
            for endpoint, (path, binary) in endpoints.items():
                observations = [
                    {"seed": int(record["seed"]), "value": value}
                    for record in episodes
                    if (value := _nested(record, path)) is not None
                ]
                values = [float(item["value"]) for item in observations]
                mean = float(np.mean(values)) if values else None
                if not values:
                    low, high = None, None
                elif binary:
                    low, high = _wilson(
                        sum(value >= 0.5 for value in values), len(values)
                    )
                else:
                    low, high = _bootstrap_mean(
                        values,
                        _analysis_rng(
                            "population", condition, population_size, endpoint
                        ),
                        resamples,
                    )
                estimate = {
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_observed": len(values),
                    "seed_values": observations,
                }
                estimates[endpoint] = estimate
                scaling_rows.append(
                    {
                        "condition": condition,
                        "population_size": population_size,
                        "n": len(episodes),
                        "endpoint": endpoint,
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "n_observed": len(values),
                    }
                )
            condition_scaling[str(population_size)] = {
                "n": len(episodes),
                "endpoints": estimates,
            }
        analysis["population_scaling"][condition] = condition_scaling

        slope_endpoints: dict[str, Any] = {}
        for endpoint in (
            "artifact_performance",
            "current_program_peak_performance",
            "final_state_artifact_performance",
            "discovery_frontier_auc",
            "portfolio_resilience",
            "ecosystem_resilience_auc",
            "held_out_resilience_auc",
        ):
            values_by_population = {
                population_size: [
                    float(item["value"])
                    for item in condition_scaling[str(population_size)][
                        "endpoints"
                    ][endpoint]["seed_values"]
                ]
                for population_size in sorted(by_population)
            }
            slope, low, high = _bootstrap_log2_slope(
                values_by_population,
                _analysis_rng("scaling-slope", condition, endpoint),
                resamples,
            )
            slope_endpoints[endpoint] = {
                "change_per_population_doubling": slope,
                "ci95_low": low,
                "ci95_high": high,
                "population_sizes": sorted(
                    size for size, values in values_by_population.items() if values
                ),
            }
        analysis["scaling_slopes"][condition] = slope_endpoints

    gain_rows: list[dict[str, Any]] = []
    independent = groups.get("independent-search", [])
    if independent:
        independent_by_key = {
            (
                int(record.get("population_size", record.get("agents", 0))),
                int(record["seed"]),
            ): record
            for record in independent
        }
        gain_endpoints = (
            "artifact_performance",
            "current_program_peak_performance",
            "final_state_artifact_performance",
            "discovery_frontier_auc",
            "portfolio_resilience",
            "validated_inventions",
            "ecosystem_resilience_auc",
            "held_out_resilience_auc",
        )
        for condition in conditions:
            if condition == "independent-search":
                continue
            condition_by_key = {
                (
                    int(record.get("population_size", record.get("agents", 0))),
                    int(record["seed"]),
                ): record
                for record in groups[condition]
            }
            shared = condition_by_key.keys() & independent_by_key.keys()
            per_population: dict[str, Any] = {}
            for population_size in sorted({key[0] for key in shared}):
                keys = sorted(key for key in shared if key[0] == population_size)
                endpoint_results: dict[str, Any] = {}
                for endpoint in gain_endpoints:
                    path = endpoints[endpoint][0]
                    paired_values: list[dict[str, Any]] = []
                    for key in keys:
                        collective = _nested(condition_by_key[key], path)
                        isolated = _nested(independent_by_key[key], path)
                        if collective is not None and isolated is not None:
                            paired_values.append(
                                {
                                    "seed": key[1],
                                    "difference": collective - isolated,
                                }
                            )
                    differences = [
                        float(item["difference"]) for item in paired_values
                    ]
                    low, high = (
                        _bootstrap_mean(
                            differences,
                            _analysis_rng(
                                "cultural-gain",
                                condition,
                                population_size,
                                endpoint,
                            ),
                            resamples,
                        )
                        if differences
                        else (None, None)
                    )
                    estimate = {
                        "collective_minus_independent_envelope": (
                            float(np.mean(differences)) if differences else None
                        ),
                        "ci95_low": low,
                        "ci95_high": high,
                        "n_pairs": len(differences),
                        "paired_values": paired_values,
                    }
                    endpoint_results[endpoint] = estimate
                    gain_rows.append(
                        {
                            "condition": condition,
                            "population_size": population_size,
                            "endpoint": endpoint,
                            "collective_minus_independent_envelope": estimate[
                                "collective_minus_independent_envelope"
                            ],
                            "ci95_low": low,
                            "ci95_high": high,
                            "n_pairs": len(differences),
                        }
                    )
                per_population[str(population_size)] = {
                    "n_pairs": len(keys),
                    "endpoints": endpoint_results,
                }
            analysis["cultural_gain"][condition] = per_population

            gain_slopes: dict[str, Any] = {}
            for endpoint in gain_endpoints:
                values_by_population = {
                    int(population_size): [
                        float(item["difference"])
                        for item in cell["endpoints"][endpoint]["paired_values"]
                    ]
                    for population_size, cell in per_population.items()
                }
                slope, low, high = _bootstrap_log2_slope(
                    values_by_population,
                    _analysis_rng("cultural-gain-slope", condition, endpoint),
                    resamples,
                )
                gain_slopes[endpoint] = {
                    "change_in_cultural_gain_per_population_doubling": slope,
                    "ci95_low": low,
                    "ci95_high": high,
                    "population_sizes": sorted(
                        size for size, values in values_by_population.items() if values
                    ),
                }
            analysis["cultural_gain_scaling_slopes"][condition] = gain_slopes

    slope_rows: list[dict[str, Any]] = []
    for condition, endpoint_results in analysis["scaling_slopes"].items():
        for endpoint, estimate in endpoint_results.items():
            slope_rows.append(
                {
                    "condition": condition,
                    "endpoint": endpoint,
                    "change_per_population_doubling": estimate[
                        "change_per_population_doubling"
                    ],
                    "ci95_low": estimate["ci95_low"],
                    "ci95_high": estimate["ci95_high"],
                    "population_sizes": " ".join(
                        map(str, estimate["population_sizes"])
                    ),
                }
            )
    for condition, endpoint_results in analysis[
        "cultural_gain_scaling_slopes"
    ].items():
        for endpoint, estimate in endpoint_results.items():
            slope_rows.append(
                {
                    "condition": f"{condition}-cultural-gain",
                    "endpoint": endpoint,
                    "change_per_population_doubling": estimate[
                        "change_in_cultural_gain_per_population_doubling"
                    ],
                    "ci95_low": estimate["ci95_low"],
                    "ci95_high": estimate["ci95_high"],
                    "population_sizes": " ".join(
                        map(str, estimate["population_sizes"])
                    ),
                }
            )

    checkpoint_seed_rows: list[dict[str, Any]] = []
    checkpoint_lookup: dict[tuple[int, str, int, int], float] = {}
    for record in records:
        population_size = int(
            record.get("population_size", record.get("agents", 0))
        )
        condition = str(record["condition"])
        seed = int(record["seed"])
        observed_ticks: set[int] = set()
        for checkpoint in record.get("held_out_checkpoints", []):
            if not isinstance(checkpoint, dict):
                continue
            tick = int(checkpoint.get("tick", -1))
            if tick in observed_ticks:
                raise ValueError(
                    f"duplicate held-out checkpoint tick {tick} for "
                    f"{condition}, N={population_size}, seed={seed}"
                )
            observed_ticks.add(tick)
            value = _nested(
                checkpoint, "held_out_generalization.resilience_auc.mean"
            )
            if value is None:
                continue
            usage = dict(checkpoint.get("model_usage", {}))
            row = {
                "condition": condition,
                "population_size": population_size,
                "seed": seed,
                "tick": tick,
                "held_out_resilience_auc": float(value),
                "model_calls": int(checkpoint.get("model_calls", 0)),
                "input_tokens": int(usage.get("input_tokens", 0)),
                "output_tokens": int(usage.get("output_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
                "selected_member": checkpoint.get("selected_member"),
            }
            checkpoint_seed_rows.append(row)
            checkpoint_lookup[(population_size, condition, seed, tick)] = float(
                value
            )

    checkpoint_estimate_rows: list[dict[str, Any]] = []
    checkpoint_gain_rows: list[dict[str, Any]] = []
    if checkpoint_seed_rows:
        checkpoint_populations = sorted(
            {int(row["population_size"]) for row in checkpoint_seed_rows}
        )
        for population_size in checkpoint_populations:
            population_result: dict[str, Any] = {
                "conditions": {},
                "paired_differences_from_full": {},
            }
            population_conditions = sorted(
                {
                    str(row["condition"])
                    for row in checkpoint_seed_rows
                    if int(row["population_size"]) == population_size
                },
                key=lambda name: (name != "full", name),
            )
            for condition in population_conditions:
                condition_result: dict[str, Any] = {}
                ticks_observed = sorted(
                    {
                        int(row["tick"])
                        for row in checkpoint_seed_rows
                        if int(row["population_size"]) == population_size
                        and str(row["condition"]) == condition
                    }
                )
                for tick in ticks_observed:
                    observations = [
                        {
                            "seed": int(row["seed"]),
                            "value": float(row["held_out_resilience_auc"]),
                        }
                        for row in checkpoint_seed_rows
                        if int(row["population_size"]) == population_size
                        and str(row["condition"]) == condition
                        and int(row["tick"]) == tick
                    ]
                    values = [float(item["value"]) for item in observations]
                    low, high = _bootstrap_mean(
                        values,
                        _analysis_rng(
                            "checkpoint", population_size, condition, tick
                        ),
                        resamples,
                    )
                    estimate = {
                        "mean": float(np.mean(values)),
                        "ci95_low": low,
                        "ci95_high": high,
                        "n_observed": len(values),
                        "seed_values": observations,
                    }
                    condition_result[str(tick)] = estimate
                    checkpoint_estimate_rows.append(
                        {
                            "condition": condition,
                            "population_size": population_size,
                            "tick": tick,
                            **{
                                key: value
                                for key, value in estimate.items()
                                if key != "seed_values"
                            },
                        }
                    )
                population_result["conditions"][condition] = condition_result

            if "full" in population_conditions:
                for condition in population_conditions:
                    if condition == "full":
                        continue
                    paired_result: dict[str, Any] = {}
                    ticks_observed = sorted(
                        set(population_result["conditions"]["full"])
                        & set(population_result["conditions"][condition])
                    )
                    for tick_text in ticks_observed:
                        tick = int(tick_text)
                        seeds = sorted(
                            {
                                key[2]
                                for key in checkpoint_lookup
                                if key[0] == population_size
                                and key[1] == "full"
                                and key[3] == tick
                            }
                            & {
                                key[2]
                                for key in checkpoint_lookup
                                if key[0] == population_size
                                and key[1] == condition
                                and key[3] == tick
                            }
                        )
                        paired_values = [
                            {
                                "seed": seed,
                                "difference": checkpoint_lookup[
                                    (population_size, "full", seed, tick)
                                ]
                                - checkpoint_lookup[
                                    (population_size, condition, seed, tick)
                                ],
                            }
                            for seed in seeds
                        ]
                        differences = [
                            float(item["difference"]) for item in paired_values
                        ]
                        low, high = _bootstrap_mean(
                            differences,
                            _analysis_rng(
                                "checkpoint-paired",
                                population_size,
                                condition,
                                tick,
                            ),
                            resamples,
                        )
                        estimate = {
                            "full_minus_condition": float(np.mean(differences)),
                            "ci95_low": low,
                            "ci95_high": high,
                            "n_pairs": len(differences),
                            "paired_values": paired_values,
                        }
                        paired_result[tick_text] = estimate
                        checkpoint_gain_rows.append(
                            {
                                "condition": condition,
                                "population_size": population_size,
                                "tick": tick,
                                **{
                                    key: value
                                    for key, value in estimate.items()
                                    if key != "paired_values"
                                },
                            }
                        )
                    population_result["paired_differences_from_full"][condition] = (
                        paired_result
                    )
            analysis["held_out_checkpoint_trajectories"][str(population_size)] = (
                population_result
            )

    compute_rows: list[dict[str, Any]] = []
    for record in records:
        actions = dict(record.get("action_counts", {}))
        usage = dict(record.get("model_usage", {}))
        action_budget = dict(record.get("action_budget", {}))
        compute_rows.append(
            {
                "condition": str(record["condition"]),
                "population_size": int(
                    record.get("population_size", record.get("agents", 0))
                ),
                "seed": int(record["seed"]),
                "decision_opportunities": record.get("decision_opportunities"),
                "provider_calls": int(record.get("model_calls", 0)),
                "provider_errors": int(record.get("model_errors", 0)),
                "input_tokens": int(usage.get("input_tokens", 0)),
                "output_tokens": int(usage.get("output_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
                "non_wait_actions": sum(
                    int(value)
                    for name, value in actions.items()
                    if name != "WAIT" and isinstance(value, int)
                ),
                "action_attempts_used": int(action_budget.get("used", 0)),
                "action_attempts_blocked": int(action_budget.get("blocked", 0)),
            }
        )
    opportunity_checks: list[dict[str, Any]] = []
    compute_by_cell: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in compute_rows:
        compute_by_cell[(row["population_size"], row["seed"])].append(row)
    for (population_size, seed), rows in sorted(compute_by_cell.items()):
        opportunities = [
            int(row["decision_opportunities"])
            for row in rows
            if row["decision_opportunities"] is not None
        ]
        opportunity_checks.append(
            {
                "population_size": population_size,
                "seed": seed,
                "conditions_observed": len(rows),
                "decision_opportunities_matched": (
                    len(opportunities) == len(rows)
                    and len(set(opportunities)) <= 1
                ),
                "minimum": min(opportunities) if opportunities else None,
                "maximum": max(opportunities) if opportunities else None,
            }
        )
    records_by_cell: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_cell[
            (
                int(record.get("population_size", record.get("agents", 0))),
                int(record["seed"]),
            )
        ].append(record)
    for audit in opportunity_checks:
        cell = records_by_cell[(audit["population_size"], audit["seed"])]
        position_sets = [record.get("initial_positions") for record in cell]
        position_sources = [
            str(record.get("initial_positions_source", "summary-unverified"))
            for record in cell
        ]
        phase_sets = [record.get("macroturn_phases") for record in cell]
        audit["initial_positions_verified"] = all(
            source != "summary-unverified" for source in position_sources
        )
        audit["initial_position_sources"] = ";".join(
            sorted(set(position_sources))
        )
        audit["initial_positions_matched"] = (
            audit["initial_positions_verified"]
            and len(position_sets) == len(cell)
            and all(value is not None for value in position_sets)
            and len({json.dumps(value, sort_keys=True) for value in position_sets}) == 1
        )
        audit["macroturn_phases_matched"] = (
            all(value is not None for value in phase_sets)
            and len({json.dumps(value, sort_keys=True) for value in phase_sets}) == 1
        )
        revisions = {record.get("engine_revision") for record in cell}
        audit["engine_revision_matched"] = len(revisions) == 1
        audit["engine_revision"] = (
            next(iter(revisions)) if len(revisions) == 1 else "mixed"
        )
        audit["valid"] = bool(
            audit["decision_opportunities_matched"]
            and audit["initial_positions_matched"]
            and audit["macroturn_phases_matched"]
            and audit["engine_revision_matched"]
        )
    analysis["compute_audit"] = {
        "decision_opportunities": opportunity_checks,
        "note": (
            "Fixed scheduling matches model decision opportunities. Token use is "
            "reported separately because social information changes prompt length."
        ),
    }

    (destination / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (destination / "condition-endpoints.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    if scaling_rows:
        with (destination / "population-scaling.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scaling_rows[0]))
            writer.writeheader()
            writer.writerows(scaling_rows)
    if gain_rows:
        with (destination / "cultural-gain.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(gain_rows[0]))
            writer.writeheader()
            writer.writerows(gain_rows)
    if pairwise_rows:
        with (destination / "pairwise-population-contrasts.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pairwise_rows[0]))
            writer.writeheader()
            writer.writerows(pairwise_rows)
    if slope_rows:
        with (destination / "scaling-slopes.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(slope_rows[0]))
            writer.writeheader()
            writer.writerows(slope_rows)
    if compute_rows:
        with (destination / "compute-audit.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(compute_rows[0]))
            writer.writeheader()
            writer.writerows(compute_rows)
    if opportunity_checks:
        with (destination / "design-audit.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(opportunity_checks[0]))
            writer.writeheader()
            writer.writerows(opportunity_checks)
    if checkpoint_seed_rows:
        with (destination / "held-out-checkpoint-seed-values.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(checkpoint_seed_rows[0])
            )
            writer.writeheader()
            writer.writerows(checkpoint_seed_rows)
    if checkpoint_estimate_rows:
        with (destination / "held-out-checkpoint-estimates.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(checkpoint_estimate_rows[0])
            )
            writer.writeheader()
            writer.writerows(checkpoint_estimate_rows)
    if checkpoint_gain_rows:
        with (destination / "held-out-checkpoint-paired-gains.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(checkpoint_gain_rows[0]))
            writer.writeheader()
            writer.writerows(checkpoint_gain_rows)
    _plot_study(analysis, conditions, destination)
    populations = {
        int(record.get("population_size", record.get("agents", 0)))
        for record in records
    }
    if len(populations) > 1:
        _plot_population_scaling(analysis, conditions, destination)
    if checkpoint_seed_rows:
        _plot_held_out_checkpoints(analysis, destination)
    return analysis


def _plot_held_out_checkpoints(
    analysis: dict[str, Any], destination: Path
) -> None:
    """Render functional trajectories and paired cultural gains over discovery time."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "publication figures require `python -m pip install -e '.[analysis]'`"
        ) from exc

    palette = {
        "full": "#13795b",
        "no-communication": "#d8664b",
        "no-program-forking": "#d39b32",
        "no-explicit-culture": "#8a5ea7",
        "independent-search": "#526b75",
    }
    trajectories = dict(analysis.get("held_out_checkpoint_trajectories", {}))
    for population_text, population_result in sorted(
        trajectories.items(), key=lambda item: int(item[0])
    ):
        population_size = int(population_text)
        condition_results = dict(population_result.get("conditions", {}))
        conditions = sorted(
            condition_results, key=lambda name: (name != "full", name)
        )
        figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))

        for condition in conditions:
            estimates = dict(condition_results[condition])
            ticks = sorted(int(tick) for tick in estimates)
            if not ticks:
                continue
            means = np.asarray(
                [float(estimates[str(tick)]["mean"]) for tick in ticks],
                dtype=np.float64,
            )
            lows = np.asarray(
                [float(estimates[str(tick)]["ci95_low"]) for tick in ticks],
                dtype=np.float64,
            )
            highs = np.asarray(
                [float(estimates[str(tick)]["ci95_high"]) for tick in ticks],
                dtype=np.float64,
            )
            color = palette.get(condition, "#92a3a0")
            label = condition.replace("-", " ")
            axes[0].plot(
                ticks,
                means,
                marker="o",
                markersize=4.5,
                linewidth=1.8,
                color=color,
                label=label,
            )
            axes[0].fill_between(ticks, lows, highs, color=color, alpha=0.14)

        paired = dict(population_result.get("paired_differences_from_full", {}))
        for condition in sorted(paired):
            estimates = dict(paired[condition])
            ticks = sorted(int(tick) for tick in estimates)
            if not ticks:
                continue
            means = np.asarray(
                [
                    float(estimates[str(tick)]["full_minus_condition"])
                    for tick in ticks
                ],
                dtype=np.float64,
            )
            lows = np.asarray(
                [float(estimates[str(tick)]["ci95_low"]) for tick in ticks],
                dtype=np.float64,
            )
            highs = np.asarray(
                [float(estimates[str(tick)]["ci95_high"]) for tick in ticks],
                dtype=np.float64,
            )
            color = palette.get(condition, "#92a3a0")
            axes[1].errorbar(
                ticks,
                means,
                yerr=_nonnegative_yerr(means, lows, highs),
                marker="o",
                markersize=4.5,
                capsize=3,
                linewidth=1.6,
                color=color,
                label=f"full − {condition.replace('-', ' ')}",
            )

        axes[0].set_title("A", loc="left", fontweight="bold")
        axes[0].set_xlabel("Discovery tick")
        axes[0].set_ylabel("Held-out resilience AUC")
        axes[1].set_title("B", loc="left", fontweight="bold")
        axes[1].set_xlabel("Discovery tick")
        axes[1].set_ylabel("Paired resilience difference")
        axes[1].axhline(0.0, color="#526b75", linewidth=1.0, linestyle="--")
        for axis in axes:
            axis.grid(alpha=0.18)
            axis.tick_params(labelsize=8.5)
            axis.xaxis.label.set_size(9)
            axis.yaxis.label.set_size(9)

        handles: list[Any] = []
        labels: list[str] = []
        for axis in axes:
            axis_handles, axis_labels = axis.get_legend_handles_labels()
            for handle, label in zip(axis_handles, axis_labels, strict=True):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
        if handles:
            figure.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=min(3, len(handles)),
                frameon=False,
                fontsize=8.5,
            )
        figure.subplots_adjust(bottom=0.25, left=0.09, right=0.98, wspace=0.30)
        stem = destination / f"held-out-checkpoint-trajectories-n{population_size}"
        figure.savefig(
            stem.with_suffix(".png"),
            dpi=300,
            facecolor="white",
            bbox_inches="tight",
        )
        figure.savefig(
            stem.with_suffix(".pdf"), facecolor="white", bbox_inches="tight"
        )
        figure.savefig(
            stem.with_suffix(".svg"), facecolor="white", bbox_inches="tight"
        )
        plt.close(figure)


def _plot_population_scaling(
    analysis: dict[str, Any], conditions: list[str], destination: Path
) -> None:
    """Render population scaling without interpolating unevaluated design cells."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "publication figures require `python -m pip install -e '.[analysis]'`"
        ) from exc

    palette = {
        "full": "#13795b",
        "no-communication": "#d8664b",
        "no-program-forking": "#d39b32",
        "no-explicit-culture": "#8a5ea7",
        "independent-search": "#526b75",
    }
    markers = {
        "full": "o",
        "no-explicit-culture": "s",
        "no-communication": "^",
        "no-program-forking": "P",
        "independent-search": "D",
    }
    labels = {
        "full": "Full culture",
        "no-explicit-culture": "No explicit culture",
        "no-communication": "No communication",
        "no-program-forking": "No program forking",
        "independent-search": "Independent search",
    }
    condition_priority = {
        "full": 0,
        "no-explicit-culture": 1,
        "no-communication": 2,
        "no-program-forking": 3,
        "independent-search": 4,
    }
    displayed = [
        condition
        for condition in sorted(
            conditions, key=lambda name: (condition_priority.get(name, 99), name)
        )
        if condition in palette and analysis["population_scaling"].get(condition)
    ]
    population_sizes = sorted(
        {
            int(size)
            for condition in displayed
            for size in analysis["population_scaling"][condition]
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.6, 3.65))

    def curve(axis: Any, endpoint: str, title: str, ylabel: str) -> bool:
        plotted = False
        for condition in displayed:
            records = analysis["population_scaling"][condition]
            sizes = sorted(int(size) for size in records)
            estimates = [records[str(size)]["endpoints"][endpoint] for size in sizes]
            complete = [
                (size, estimate)
                for size, estimate in zip(sizes, estimates, strict=True)
                if estimate["mean"] is not None
            ]
            if not complete:
                continue
            plotted = True
            by_size = {size: estimate for size, estimate in complete}
            x = np.asarray(population_sizes, dtype=np.float64)
            means = np.asarray(
                [
                    np.nan if size not in by_size else by_size[size]["mean"]
                    for size in population_sizes
                ],
                dtype=np.float64,
            )
            lows = np.asarray(
                [
                    np.nan if size not in by_size else by_size[size]["ci95_low"]
                    for size in population_sizes
                ],
                dtype=np.float64,
            )
            highs = np.asarray(
                [
                    np.nan if size not in by_size else by_size[size]["ci95_high"]
                    for size in population_sizes
                ],
                dtype=np.float64,
            )
            axis.plot(
                x,
                means,
                marker=markers[condition],
                markersize=5.2,
                linewidth=1.8,
                color=palette[condition],
                label=labels[condition],
            )
            axis.fill_between(x, lows, highs, color=palette[condition], alpha=0.14)
            for size, estimate in complete:
                raw = [
                    float(item["value"])
                    for item in estimate.get("seed_values", [])
                ]
                if raw:
                    jitter = np.power(
                        2.0, np.linspace(-0.035, 0.035, len(raw), dtype=np.float64)
                    )
                    axis.scatter(
                        float(size) * jitter,
                        raw,
                        s=14,
                        color=palette[condition],
                        alpha=0.38,
                        linewidths=0,
                    )
        axis.set_xscale("log", base=2)
        axis.set_xticks(population_sizes, [str(size) for size in population_sizes])
        axis.set_title(title, loc="left", fontweight="bold", fontsize=10)
        axis.set_xlabel("Number of agents, N", fontsize=9)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.18)
        axis.tick_params(labelsize=8.5)
        axis.yaxis.label.set_size(9)
        if not plotted:
            axis.text(
                0.5,
                0.5,
                "not evaluated",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#526b75",
            )
        return plotted

    curve(
        axes[0],
        "discovery_frontier_auc",
        "A",
        "Discovery-frontier AUC",
    )
    held_out_available = any(
        cell["endpoints"]["held_out_resilience_auc"]["mean"] is not None
        for condition in displayed
        for cell in analysis["population_scaling"][condition].values()
    )
    ecology_endpoint = (
        "held_out_resilience_auc"
        if held_out_available
        else "ecosystem_resilience_auc"
    )
    curve(
        axes[1],
        ecology_endpoint,
        "B",
        "Held-out resilience AUC" if held_out_available else "Native resilience AUC",
    )
    curve(
        axes[2],
        "portfolio_resilience",
        "C",
        "Portfolio resilience",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=len(handles),
            frameon=False,
            fontsize=8.5,
            handletextpad=0.5,
            columnspacing=1.1,
        )
    figure.subplots_adjust(bottom=0.24, left=0.07, right=0.99, wspace=0.31)
    figure.savefig(
        destination / "population-scaling.png",
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
    )
    figure.savefig(
        destination / "population-scaling.pdf",
        facecolor="white",
        bbox_inches="tight",
    )
    figure.savefig(
        destination / "population-scaling.svg",
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(figure)
    _plot_cultural_gain_scaling(analysis, conditions, destination)


def _plot_cultural_gain_scaling(
    analysis: dict[str, Any], conditions: list[str], destination: Path
) -> None:
    """Plot each shared-world treatment against its matched isolated envelope."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "publication figures require `python -m pip install -e '.[analysis]'`"
        ) from exc

    palette = {
        "full": "#13795b",
        "no-explicit-culture": "#8a5ea7",
        "no-communication": "#d8664b",
        "no-program-forking": "#d39b32",
    }
    markers = {
        "full": "o",
        "no-explicit-culture": "s",
        "no-communication": "^",
        "no-program-forking": "P",
    }
    labels = {
        "full": "Full culture",
        "no-explicit-culture": "No explicit culture",
        "no-communication": "No communication",
        "no-program-forking": "No program forking",
    }
    priority = {
        "full": 0,
        "no-explicit-culture": 1,
        "no-communication": 2,
        "no-program-forking": 3,
    }
    cultural_gain = dict(analysis.get("cultural_gain", {}))
    displayed = [
        condition
        for condition in sorted(
            conditions, key=lambda name: (priority.get(name, 99), name)
        )
        if condition in palette and cultural_gain.get(condition)
    ]
    if not displayed:
        return
    population_sizes = sorted(
        {
            int(size)
            for condition in displayed
            for size in cultural_gain[condition]
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.6, 3.65))
    endpoints = (
        ("artifact_performance", "A", "Best-artifact gain"),
        ("discovery_frontier_auc", "B", "Discovery-frontier gain"),
        ("held_out_resilience_auc", "C", "Held-out resilience gain"),
    )

    for axis, (endpoint, panel, ylabel) in zip(axes, endpoints, strict=True):
        for condition in displayed:
            estimates = cultural_gain[condition]
            by_size = {
                int(size): cell["endpoints"][endpoint]
                for size, cell in estimates.items()
                if cell["endpoints"][endpoint][
                    "collective_minus_independent_envelope"
                ]
                is not None
            }
            if not by_size:
                continue
            x = np.asarray(population_sizes, dtype=np.float64)
            means = np.asarray(
                [
                    np.nan
                    if size not in by_size
                    else by_size[size]["collective_minus_independent_envelope"]
                    for size in population_sizes
                ],
                dtype=np.float64,
            )
            lows = np.asarray(
                [
                    np.nan if size not in by_size else by_size[size]["ci95_low"]
                    for size in population_sizes
                ],
                dtype=np.float64,
            )
            highs = np.asarray(
                [
                    np.nan if size not in by_size else by_size[size]["ci95_high"]
                    for size in population_sizes
                ],
                dtype=np.float64,
            )
            axis.errorbar(
                x,
                means,
                yerr=_nonnegative_yerr(means, lows, highs),
                marker=markers[condition],
                markersize=5.2,
                capsize=3,
                linewidth=1.8,
                color=palette[condition],
                label=labels[condition],
            )
            for size, estimate in by_size.items():
                raw = [
                    float(item["difference"])
                    for item in estimate.get("paired_values", [])
                ]
                if raw:
                    jitter = np.power(
                        2.0, np.linspace(-0.035, 0.035, len(raw), dtype=np.float64)
                    )
                    axis.scatter(
                        float(size) * jitter,
                        raw,
                        s=14,
                        color=palette[condition],
                        alpha=0.38,
                        linewidths=0,
                    )
        axis.axhline(0.0, color="#526b75", linewidth=1.0, linestyle="--")
        axis.set_xscale("log", base=2)
        axis.set_xticks(population_sizes, [str(size) for size in population_sizes])
        axis.set_title(panel, loc="left", fontweight="bold", fontsize=10)
        axis.set_xlabel("Number of agents, N", fontsize=9)
        axis.set_ylabel(ylabel, fontsize=9)
        axis.tick_params(labelsize=8.5)
        axis.grid(alpha=0.18)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=len(handles),
            frameon=False,
            fontsize=8.5,
            handletextpad=0.5,
            columnspacing=1.1,
        )
    figure.subplots_adjust(bottom=0.24, left=0.07, right=0.99, wspace=0.31)
    for suffix, options in (
        ("png", {"dpi": 300}),
        ("pdf", {}),
        ("svg", {}),
    ):
        figure.savefig(
            destination / f"cultural-gain-scaling.{suffix}",
            facecolor="white",
            bbox_inches="tight",
            **options,
        )
    plt.close(figure)


def _plot_study(
    analysis: dict[str, Any], conditions: list[str], destination: Path
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "publication figures require `python -m pip install -e '.[analysis]'`"
        ) from exc

    short_labels = {
        "full": "full",
        "global-landmarks": "+map",
        "no-communication": "−comms",
        "no-depot": "−depot",
        "no-feedback": "−feedback",
        "no-social-awareness": "−social",
    }
    labels = [short_labels.get(condition, condition.replace("-", "\n")) for condition in conditions]
    colors = ["#13795b" if condition == "full" else "#92a3a0" for condition in conditions]
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 7.2), constrained_layout=True)

    def values(endpoint: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        estimates = [analysis["conditions"][name]["endpoints"][endpoint] for name in conditions]
        means = np.asarray(
            [np.nan if item["mean"] is None else item["mean"] for item in estimates]
        )
        lows = np.asarray(
            [np.nan if item["ci95_low"] is None else item["ci95_low"] for item in estimates]
        )
        highs = np.asarray(
            [np.nan if item["ci95_high"] is None else item["ci95_high"] for item in estimates]
        )
        return means, means - lows, highs - means

    def bars(
        axis: Any,
        endpoint: str,
        title: str,
        ylabel: str,
        limit: tuple[float, float] | None = None,
    ) -> None:
        means, lower, upper = values(endpoint)
        axis.bar(range(len(conditions)), means, color=colors, width=0.68, zorder=2)
        axis.errorbar(
            range(len(conditions)),
            means,
            yerr=np.maximum(0.0, np.vstack([lower, upper])),
            fmt="none",
            ecolor="#21312d", capsize=3, linewidth=1.1, zorder=3,
        )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.set_xticks(range(len(conditions)), labels)
        if limit is not None:
            axis.set_ylim(*limit)
        axis.grid(axis="y", alpha=0.18, zorder=0)

    bars(axes[0, 0], "outcome_success", "A  Functional discovery", "episode success", (0, 1.05))
    bars(
        axes[0, 1],
        "emergent_success",
        "B  Causally distributed discovery",
        "episode success",
        (0, 1.05),
    )
    bars(axes[1, 0], "artifact_performance", "C  Field performance", "best performance")
    bars(
        axes[1, 1],
        "specialization",
        "D  Spontaneous specialization",
        "excess normalized mutual information",
    )
    figure.suptitle(
        "Open Invention: decentralized collective science",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(destination / "study-overview.png", dpi=240, facecolor="white")
    figure.savefig(destination / "study-overview.pdf", facecolor="white")
    plt.close(figure)

    portfolio_figure, portfolio_axes = plt.subplots(
        1, 3, figsize=(12.0, 3.8), constrained_layout=True
    )
    bars(
        portfolio_axes[0],
        "portfolio_resilience",
        "A  Portfolio resilience",
        "balanced service coverage",
        (0, 1.0),
    )
    bars(
        portfolio_axes[1],
        "service_breadth",
        "B  Service breadth",
        "services above threshold",
        (0, 6.2),
    )
    bars(
        portfolio_axes[2],
        "validated_inventions",
        "C  Validated inventions",
        "episode count",
    )
    portfolio_figure.suptitle(
        "Collective functional portfolio",
        fontsize=14,
        fontweight="bold",
    )
    portfolio_figure.savefig(
        destination / "study-portfolio.png", dpi=240, facecolor="white"
    )
    portfolio_figure.savefig(destination / "study-portfolio.pdf", facecolor="white")
    plt.close(portfolio_figure)

    ecology_figure, ecology_axes = plt.subplots(
        2, 3, figsize=(14.2, 7.6), constrained_layout=True
    )
    bars(
        ecology_axes[0, 0],
        "ecosystem_resilience_auc",
        "A  Frozen-society resilience",
        "agent-free service AUC",
        (0, 1.0),
    )
    bars(
        ecology_axes[0, 1],
        "collaborative_artifact_fraction",
        "B  Collaborative technologies",
        "fraction with multi-agent ancestry",
        (0, 1.0),
    )
    bars(
        ecology_axes[0, 2],
        "cross_agent_program_fork_fraction",
        "C  Cross-agent inheritance",
        "fraction of program forks",
        (0, 1.0),
    )
    bars(
        ecology_axes[1, 0],
        "ecological_coupling",
        "D  Technology coupling",
        "mean absolute knockout effect",
    )
    bars(
        ecology_axes[1, 1],
        "reciprocal_support_pairs",
        "E  Reciprocal dependence",
        "artifact pairs per episode",
    )
    bars(
        ecology_axes[1, 2],
        "mean_distance_traveled",
        "F  Society mobility",
        "steps traveled per agent",
    )
    ecology_figure.suptitle(
        "Co-evolution of a decentralized technological ecosystem",
        fontsize=14,
        fontweight="bold",
    )
    ecology_figure.savefig(
        destination / "study-technology-ecology.png", dpi=240, facecolor="white"
    )
    ecology_figure.savefig(
        destination / "study-technology-ecology.pdf", facecolor="white"
    )
    plt.close(ecology_figure)

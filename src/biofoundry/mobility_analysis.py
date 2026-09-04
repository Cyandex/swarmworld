"""Seed-level movement and spatial self-organization analysis.

The statistical replicate is always an independent simulation seed. Per-agent
trajectories are used to construct one seed-level observable; they are never treated
as independent samples. Independent-search records contribute matched endpoint
mobility summaries, but not a fictitious shared-space trajectory because their agents
occupy separate worlds.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .events import read_records
from .types import ActionType

MOBILITY_ANALYSIS_VERSION = 3
CONDITIONS = (
    "full",
    "no-explicit-culture",
    "no-communication",
    "independent-search",
)
SHARED_CONDITIONS = CONDITIONS[:3]
LABELS = {
    "full": "Full culture",
    "no-explicit-culture": "No explicit culture",
    "no-communication": "No communication",
    "independent-search": "Independent search",
}
COLORS = {
    "full": "#13795b",
    "no-explicit-culture": "#8a5ea7",
    "no-communication": "#d8664b",
    "independent-search": "#526b75",
}
MARKERS = {
    "full": "o",
    "no-explicit-culture": "s",
    "no-communication": "^",
    "independent-search": "D",
}

ENDPOINT_METRICS = (
    "mean_distance_traveled",
    "mean_distinct_cells_visited",
    "mean_net_displacement",
    "displacement_efficiency",
    "exploration_efficiency",
    "successful_moves_per_agent",
    "final_regional_crowding",
    "path_length_gini",
    "final_spatial_entropy",
    "final_artifact_attraction",
    "final_artifact_proximity_enrichment",
    "mean_regional_switch_fraction",
    "encounter_exposure_auc",
    "colocation_auc",
    "artifact_contact_auc",
    "artifact_bound_movement_auc",
    "moving_toward_artifact_auc",
    "artifact_contact_gini",
    "artifact_work_events_per_100_agent_ticks",
    "explicit_interactions_per_100_agent_ticks",
    "mobility_artifact_work_spearman",
    "mobility_social_interaction_spearman",
    "artifact_work_gini",
    "social_interaction_gini",
    "artifact_work_participation_fraction",
    "social_participation_fraction",
)

ACTIVITY_CATEGORIES = (
    "idle",
    "movement",
    "observation/testing",
    "material processing",
    "construction/control",
    "culture/coordination",
)
ACTIVITY_COLORS = (
    "#edf0ee",
    "#3c8dbc",
    "#e1a83a",
    "#7b9f5d",
    "#8a5ea7",
    "#d8664b",
)
MOBILITY_STATES = (
    "stationary/dispersed",
    "exploring",
    "artifact-local",
    "artifact-bound movement",
)
HIGHLIGHT_EVENT_KINDS = {
    "sample_inspected",
    "material_tested",
    "artifact_built",
    "artifact_program_installed",
    "message_delivered",
    "resource_traded",
    "knowledge_taught",
}

BEHAVIOR_FEATURES = (
    ("distance_traveled", "path length", "log1p"),
    ("net_displacement", "net displacement", "log1p"),
    ("displacement_efficiency", "path directness", "identity"),
    ("distinct_coarse_cells", "regions visited", "log1p"),
    ("movement_snapshot_fraction", "mobile snapshots", "identity"),
    ("artifact_contact_fraction", "artifact proximity", "identity"),
    (
        "artifact_bound_movement_fraction",
        "artifact-bound movement",
        "identity",
    ),
    (
        "moving_toward_artifact_fraction",
        "artifact-directed movement",
        "identity",
    ),
    ("encounter_exposure_per_snapshot", "nearby agents", "log1p"),
    ("artifact_work_rate_per_100_ticks", "technology work", "log1p"),
    ("social_interaction_rate_per_100_ticks", "social interaction", "log1p"),
    ("observation_testing_rate_per_100_ticks", "observation/testing", "log1p"),
    ("material_processing_rate_per_100_ticks", "material processing", "log1p"),
    (
        "construction_control_rate_per_100_ticks",
        "construction/control",
        "log1p",
    ),
    (
        "culture_coordination_rate_per_100_ticks",
        "culture/coordination",
        "log1p",
    ),
)

PHENOTYPE_COLORS = (
    "#3c8dbc",
    "#d8664b",
    "#8a5ea7",
    "#13795b",
    "#d9a21b",
    "#526b75",
)

REGIME_FEATURES = (
    ("mean_distance_traveled", "path length"),
    ("mean_distinct_cells_visited", "distinct cells"),
    ("mean_net_displacement", "net displacement"),
    ("displacement_efficiency", "path directness"),
    ("exploration_efficiency", "exploration efficiency"),
    ("successful_moves_per_agent", "successful moves"),
    ("early_mobility_rate", "early mobility"),
    ("late_mobility_rate", "late mobility"),
    ("moving_fraction_auc", "moving fraction"),
)

POPULATION_COLORS = {
    50: "#d9a21b",
    100: "#3c8dbc",
    200: "#8a5ea7",
}


def _population_color_map(populations: Iterable[int]) -> dict[int, str]:
    """Return stable paper colors plus deterministic fallbacks for other sizes."""

    ordered = sorted({int(population) for population in populations})
    fallback = ("#13795b", "#d8664b", "#526b75", "#b06c49", "#6f88a8", "#7a6f9b")
    return {
        population: POPULATION_COLORS.get(population, fallback[index % len(fallback)])
        for index, population in enumerate(ordered)
    }


def _stable_rng(*parts: object) -> np.random.Generator:
    key = "\x1f".join(map(str, parts)).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return np.random.default_rng(seed)


def _mean_ci(
    values: Iterable[float],
    *,
    key: tuple[object, ...],
    resamples: int,
) -> dict[str, Any]:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {
            "mean": None,
            "ci95_low": None,
            "ci95_high": None,
            "n": 0,
        }
    if array.size == 1:
        low = high = float(array[0])
    else:
        rng = _stable_rng("mobility", *key)
        indices = rng.integers(0, array.size, size=(int(resamples), array.size))
        draws = array[indices].mean(axis=1)
        low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "mean": round(float(np.mean(array)), 8),
        "ci95_low": round(float(low), 8),
        "ci95_high": round(float(high), 8),
        "n": int(array.size),
    }


def gini(values: Iterable[float]) -> float:
    """Return the finite-sample Gini coefficient for non-negative values."""

    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size or float(np.sum(array)) <= 1e-12:
        return 0.0
    array = np.sort(np.maximum(array, 0.0))
    ranks = np.arange(1, array.size + 1, dtype=np.float64)
    numerator = float(np.sum((2.0 * ranks - array.size - 1.0) * array))
    return float(np.clip(numerator / (array.size * float(np.sum(array))), 0.0, 1.0))


def _pearson(x: Iterable[float], y: Iterable[float]) -> float | None:
    first = np.asarray(list(x), dtype=np.float64)
    second = np.asarray(list(y), dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    first = first[valid]
    second = second[valid]
    if first.size < 3 or float(np.std(first)) <= 1e-12 or float(np.std(second)) <= 1e-12:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _rank_values(values: np.ndarray) -> np.ndarray:
    """Return average ranks for a one-dimensional array, including ties."""

    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    cursor = 0
    while cursor < values.size:
        stop = cursor + 1
        while stop < values.size and values[order[stop]] == values[order[cursor]]:
            stop += 1
        ranks[order[cursor:stop]] = 0.5 * (cursor + stop - 1)
        cursor = stop
    return ranks


def _spearman(x: Iterable[float], y: Iterable[float]) -> float | None:
    first = np.asarray(list(x), dtype=np.float64)
    second = np.asarray(list(y), dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    first = first[valid]
    second = second[valid]
    if first.size < 3 or np.unique(first).size < 2 or np.unique(second).size < 2:
        return None
    return _pearson(_rank_values(first), _rank_values(second))


def _activity_category(verb: int) -> int:
    action = ActionType(int(verb))
    if action == ActionType.MOVE:
        return 1
    if action in {ActionType.INSPECT, ActionType.TEST}:
        return 2
    if action in {
        ActionType.HARVEST,
        ActionType.DEPOSIT,
        ActionType.OPERATE,
        ActionType.METABOLIZE,
    }:
        return 3
    if action in {
        ActionType.BUILD,
        ActionType.REPAIR,
        ActionType.DISMANTLE,
        ActionType.WRITE_PROGRAM,
        ActionType.FORK_PROGRAM,
        ActionType.COMBINE_DESIGN,
    }:
        return 4
    if action in {
        ActionType.COMMUNICATE,
        ActionType.PUBLISH,
        ActionType.DEPOSIT_INSIGHT,
        ActionType.PROPOSE_RECIPE,
        ActionType.CLAIM_TASK,
        ActionType.TEACH,
        ActionType.TRADE,
    }:
        return 5
    return 0


def _event_agent(payload: dict[str, Any]) -> str | None:
    for key in ("agent", "author", "sender", "teacher"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _encounter_observables(
    x: np.ndarray,
    y: np.ndarray,
    *,
    radius: float,
) -> tuple[float, float]:
    """Return mean neighbors and the fraction sharing an exact cell."""

    if x.size <= 1:
        return 0.0, 0.0
    dx = x[:, None].astype(np.float64) - x[None, :].astype(np.float64)
    dy = y[:, None].astype(np.float64) - y[None, :].astype(np.float64)
    distance = np.hypot(dx, dy)
    np.fill_diagonal(distance, np.inf)
    mean_neighbors = float(np.mean(np.sum(distance <= radius, axis=1)))
    colocation = float(np.mean(np.any(distance <= 1e-12, axis=1)))
    return mean_neighbors, colocation


def _agent_neighbor_counts(
    x: np.ndarray,
    y: np.ndarray,
    *,
    radius: float,
) -> np.ndarray:
    """Return the number of other agents within ``radius`` for every agent."""

    if x.size <= 1:
        return np.zeros(x.size, dtype=np.int64)
    dx = x[:, None].astype(np.float64) - x[None, :].astype(np.float64)
    dy = y[:, None].astype(np.float64) - y[None, :].astype(np.float64)
    within = dx * dx + dy * dy <= float(radius) ** 2
    np.fill_diagonal(within, False)
    return np.sum(within, axis=1).astype(np.int64)


def _slope(points: list[dict[str, Any]], metric: str, start: int, stop: int) -> float | None:
    selected = [
        point
        for point in points
        if start <= int(point.get("tick", -1)) <= stop
        and isinstance(point.get(metric), (int, float))
    ]
    if len(selected) < 2:
        return None
    ticks = np.asarray([point["tick"] for point in selected], dtype=np.float64)
    values = np.asarray([point[metric] for point in selected], dtype=np.float64)
    centered = ticks - float(np.mean(ticks))
    denominator = float(np.sum(centered * centered))
    if denominator <= 1e-12:
        return None
    return float(np.sum(centered * (values - float(np.mean(values)))) / denominator)


def _auc(points: list[dict[str, Any]], metric: str) -> float | None:
    selected = [
        point
        for point in points
        if isinstance(point.get(metric), (int, float))
        and math.isfinite(float(point[metric]))
    ]
    if len(selected) < 2:
        return None
    ticks = np.asarray([point["tick"] for point in selected], dtype=np.float64)
    values = np.asarray([point[metric] for point in selected], dtype=np.float64)
    duration = max(1.0, float(ticks[-1] - ticks[0]))
    return float(np.trapezoid(values, ticks) / duration)


def _resolve_trace(record: dict[str, Any], summary_source: Path) -> Path | None:
    output = record.get("output")
    if not isinstance(output, str) or not output:
        return None
    candidates = [Path(output), summary_source.parent / Path(output).name]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _load_records(summary_paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source_value in summary_paths:
        source = Path(source_value)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"study summary must contain a JSON list: {source}")
        for item in payload:
            record = dict(item)
            record["_summary_source"] = str(source.resolve())
            trace = _resolve_trace(record, source)
            record["_resolved_trace"] = str(trace) if trace is not None else None
            records.append(record)
    return records


def _nearest_distances(
    x: np.ndarray,
    y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> np.ndarray:
    if not x.size or not target_x.size:
        return np.asarray([], dtype=np.float64)
    minimum = np.full(x.size, np.inf, dtype=np.float64)
    for tx, ty in zip(target_x, target_y, strict=True):
        distance = np.hypot(x - float(tx), y - float(ty))
        np.minimum(minimum, distance, out=minimum)
    return minimum


def _nearest_neighbor_distance(x: np.ndarray, y: np.ndarray) -> float:
    if x.size <= 1:
        return 0.0
    dx = x[:, None].astype(np.float64) - x[None, :].astype(np.float64)
    dy = y[:, None].astype(np.float64) - y[None, :].astype(np.float64)
    distance = np.hypot(dx, dy)
    np.fill_diagonal(distance, np.inf)
    return float(np.mean(np.min(distance, axis=1)))


def _coarse_bins(
    x: np.ndarray,
    y: np.ndarray,
    *,
    width: int,
    height: int,
    columns: int = 8,
    rows: int = 6,
) -> np.ndarray:
    bin_x = np.minimum(columns - 1, np.maximum(0, x * columns // max(1, width)))
    bin_y = np.minimum(rows - 1, np.maximum(0, y * rows // max(1, height)))
    return bin_y.astype(np.int64) * columns + bin_x.astype(np.int64)


def _spatial_entropy(bins: np.ndarray, population: int, total_bins: int = 48) -> float:
    if population <= 1:
        return 0.0
    counts = np.bincount(bins, minlength=total_bins).astype(np.float64)
    probabilities = counts[counts > 0] / population
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    maximum = math.log(max(2, min(population, total_bins)))
    return float(np.clip(entropy / maximum, 0.0, 1.0))


def _active_artifacts(snapshot: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    artifacts = dict(snapshot.get("artifacts", {}))
    count = int(artifacts.get("display_count", artifacts.get("count", 0)))
    x = np.asarray(artifacts.get("x", [])[:count], dtype=np.float64)
    y = np.asarray(artifacts.get("y", [])[:count], dtype=np.float64)
    retired = np.asarray(artifacts.get("retired", [False] * count)[:count], dtype=np.bool_)
    if x.size != count or y.size != count:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    active = ~retired if retired.size == count else np.ones(count, dtype=np.bool_)
    return x[active], y[active]


def _world_grid(world: dict[str, Any], key: str, *, dtype: Any) -> np.ndarray:
    values = np.asarray(world.get(key, []), dtype=dtype)
    width = int(world.get("width", 0))
    height = int(world.get("height", 0))
    if values.ndim == 1 and width > 0 and height > 0 and values.size == width * height:
        return values.reshape(height, width)
    return values


def _top_resource_cells(world: dict[str, Any], limit: int = 64) -> tuple[np.ndarray, np.ndarray]:
    mass = _world_grid(world, "resource_mass", dtype=np.float64)
    if mass.ndim != 2 or not np.any(mass > 0):
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    flat = mass.ravel()
    positive = np.nonzero(flat > 0)[0]
    if positive.size > limit:
        order = np.argsort(flat[positive], kind="stable")[-limit:]
        positive = positive[order]
    y, x = np.unravel_index(positive, mass.shape)
    return x.astype(np.float64), y.astype(np.float64)


def extract_spatial_trace(
    path: str | Path,
    *,
    proximity_radius: float = 3.0,
    encounter_radius: float = 2.0,
    retain_positions: bool = False,
) -> dict[str, Any]:
    """Extract spatial, interaction, and activity observables from one trace."""

    points: list[dict[str, Any]] = []
    trajectories: dict[str, list[list[float]]] = defaultdict(list)
    activity_counts: dict[str, Counter[int]] = defaultdict(Counter)
    activity_bins: dict[str, dict[int, Counter[int]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    artifact_work: Counter[str] = Counter()
    social_interactions: Counter[str] = Counter()
    successful_actions: Counter[str] = Counter()
    artifact_participants: dict[str, set[str]] = defaultdict(set)
    interaction_edges: Counter[tuple[str, str]] = Counter()
    social_partners: dict[str, set[str]] = defaultdict(set)
    artifacts_worked: dict[str, set[str]] = defaultdict(set)
    highlight_events: list[dict[str, Any]] = []
    transition_counts = np.zeros(
        (len(MOBILITY_STATES), len(MOBILITY_STATES)), dtype=np.int64
    )
    previous_bins: dict[str, int] | None = None
    previous_distance: dict[str, float] | None = None
    previous_positions: dict[str, tuple[float, float]] | None = None
    previous_states: dict[str, int] | None = None
    previous_tick: int | None = None
    contact_hits: Counter[str] = Counter()
    contact_opportunities: Counter[str] = Counter()
    motion_hits: Counter[str] = Counter()
    motion_opportunities: Counter[str] = Counter()
    artifact_bound_motion_hits: Counter[str] = Counter()
    moving_toward_artifact_hits: Counter[str] = Counter()
    moving_toward_artifact_opportunities: Counter[str] = Counter()
    encounter_exposure_sum: Counter[str] = Counter()
    encounter_opportunities: Counter[str] = Counter()
    coarse_cells: dict[str, set[int]] = defaultdict(set)
    initial_positions: dict[str, tuple[float, float]] = {}
    final_positions: dict[str, tuple[float, float]] = {}
    interval_speed_sum: Counter[str] = Counter()
    explicit_dyads = 0
    artifact_work_events = 0
    last_tick = 0
    terrain_cache: np.ndarray | None = None
    world_width = 0
    world_height = 0
    compact_snapshots: list[dict[str, Any]] = []

    artifact_work_kinds = {
        "microbatch_fabricated",
        "material_tested",
        "artifact_built",
        "artifact_program_installed",
        "artifact_repaired",
        "artifact_dismantled",
    }

    for record in read_records(path):
        tick = int(record.get("tick", 0))
        last_tick = max(last_tick, tick)
        if record.get("type") == "event":
            kind = str(record.get("kind", ""))
            payload = dict(record.get("payload", {}))
            agent_id = _event_agent(payload)
            if kind == "action_result" and payload.get("success") is True and agent_id:
                try:
                    category = _activity_category(int(payload.get("verb", 0)))
                except (TypeError, ValueError):
                    category = 0
                successful_actions[agent_id] += 1
                activity_counts[agent_id][category] += 1
                if retain_positions:
                    activity_bins[agent_id][tick // 24][category] += 1

            pairs: list[tuple[str, str]] = []
            if kind == "message_delivered":
                sender = str(payload.get("sender", ""))
                pairs.extend(
                    (sender, str(recipient))
                    for recipient in payload.get("recipients", [])
                    if sender and recipient
                )
            elif kind == "resource_traded":
                sender = str(payload.get("sender", ""))
                recipient = str(payload.get("recipient", ""))
                if sender and recipient:
                    pairs.append((sender, recipient))
            elif kind == "knowledge_taught":
                teacher = str(payload.get("teacher", ""))
                pairs.extend(
                    (teacher, str(recipient))
                    for recipient in payload.get("recipients", [])
                    if teacher and recipient
                )
            for sender, recipient in pairs:
                explicit_dyads += 1
                social_interactions[sender] += 1
                social_interactions[recipient] += 1
                social_partners[sender].add(recipient)
                social_partners[recipient].add(sender)
                interaction_edges[(sender, recipient)] += 1

            if kind in artifact_work_kinds and agent_id:
                artifact_work_events += 1
                artifact_work[agent_id] += 1
                artifact_id = payload.get("artifact_id")
                if isinstance(artifact_id, str) and artifact_id:
                    artifact_participants[artifact_id].add(agent_id)
                    artifacts_worked[agent_id].add(artifact_id)
                if kind == "artifact_built":
                    batch = payload.get("batch")
                    contributors = (
                        batch.get("contributors", []) if isinstance(batch, dict) else []
                    )
                    for contributor in contributors:
                        if contributor:
                            artifact_participants[str(artifact_id)].add(str(contributor))
                            artifacts_worked[str(contributor)].add(str(artifact_id))

            if retain_positions and kind in HIGHLIGHT_EVENT_KINDS:
                highlight_events.append(
                    {
                        "tick": tick,
                        "kind": kind,
                        "agent": agent_id,
                        "x": payload.get("x"),
                        "y": payload.get("y"),
                        "artifact_id": payload.get("artifact_id"),
                        "peers": [recipient for _, recipient in pairs],
                    }
                )
            continue
        if record.get("type") != "snapshot":
            continue
        snapshot = dict(record.get("snapshot", {}))
        agents = dict(snapshot.get("agents", {}))
        count = int(agents.get("display_count", agents.get("count", 0)))
        ids = [str(value) for value in agents.get("ids", [])[:count]]
        x = np.asarray(agents.get("x", [])[:count], dtype=np.int64)
        y = np.asarray(agents.get("y", [])[:count], dtype=np.int64)
        distance = np.asarray(
            agents.get("distance_traveled", [])[:count], dtype=np.float64
        )
        visited = np.asarray(
            agents.get("distinct_cells_visited", [])[:count], dtype=np.float64
        )
        if len(ids) != count or x.size != count or y.size != count:
            continue

        tick = int(snapshot.get("tick", 0))
        last_tick = max(last_tick, tick)
        world = dict(snapshot.get("world", {}))
        width = int(world.get("width", 1))
        height = int(world.get("height", 1))
        world_width = width
        world_height = height
        diagonal = max(1.0, math.hypot(width - 1, height - 1))
        bins = _coarse_bins(x, y, width=width, height=height)
        current_bins = dict(zip(ids, bins.tolist(), strict=True))
        current_distance = dict(zip(ids, distance.tolist(), strict=True))
        regional_switch_fraction = 0.0
        relocating_fraction = 0.0
        if previous_bins is not None:
            shared = [agent_id for agent_id in ids if agent_id in previous_bins]
            if shared:
                regional_switch_fraction = float(
                    np.mean(
                        [
                            current_bins[agent_id] != previous_bins[agent_id]
                            for agent_id in shared
                        ]
                    )
                )
        if previous_distance is not None:
            shared = [agent_id for agent_id in ids if agent_id in previous_distance]
            if shared:
                relocating_fraction = float(
                    np.mean(
                        [
                            current_distance[agent_id]
                            > previous_distance[agent_id] + 1e-12
                            for agent_id in shared
                        ]
                    )
                )
        center_x = float(np.mean(x)) if x.size else 0.0
        center_y = float(np.mean(y)) if y.size else 0.0
        radius = (
            float(np.sqrt(np.mean((x - center_x) ** 2 + (y - center_y) ** 2)))
            if x.size
            else 0.0
        )
        artifact_x, artifact_y = _active_artifacts(snapshot)
        resource_x, resource_y = _top_resource_cells(world)
        agent_artifact_distance = _nearest_distances(x, y, artifact_x, artifact_y)
        agent_resource_distance = _nearest_distances(x, y, resource_x, resource_y)

        terrain = _world_grid(world, "terrain", dtype=np.int16)
        if terrain_cache is None and terrain.shape == (height, width):
            terrain_cache = terrain.copy()
        if terrain.shape == (height, width):
            walk_y, walk_x = np.nonzero(terrain > 0)
        else:
            walk_y, walk_x = np.mgrid[0:height, 0:width]
            walk_x = walk_x.ravel()
            walk_y = walk_y.ravel()
        null_artifact_distance = _nearest_distances(
            walk_x, walk_y, artifact_x, artifact_y
        )
        null_resource_distance = _nearest_distances(
            walk_x, walk_y, resource_x, resource_y
        )

        def attraction(observed: np.ndarray, baseline: np.ndarray) -> float | None:
            if not observed.size or not baseline.size:
                return None
            denominator = float(np.mean(baseline))
            if denominator <= 1e-12:
                return None
            return float(1.0 - float(np.mean(observed)) / denominator)

        artifact_near = (
            float(np.mean(agent_artifact_distance <= proximity_radius))
            if agent_artifact_distance.size
            else None
        )
        artifact_available = (
            float(np.mean(null_artifact_distance <= proximity_radius))
            if null_artifact_distance.size
            else None
        )
        enrichment = (
            artifact_near / artifact_available
            if artifact_near is not None
            and artifact_available is not None
            and artifact_available > 1e-12
            else None
        )
        mean_neighbors, colocation_fraction = _encounter_observables(
            x,
            y,
            radius=encounter_radius,
        )
        current_positions = {
            agent_id: (float(px), float(py))
            for agent_id, px, py in zip(ids, x, y, strict=True)
        }
        for agent_id, px, py in zip(ids, x, y, strict=True):
            position = (float(px), float(py))
            initial_positions.setdefault(agent_id, position)
            final_positions[agent_id] = position
        for agent_id, coarse_cell in zip(ids, bins, strict=True):
            coarse_cells[agent_id].add(int(coarse_cell))
        neighbor_counts = _agent_neighbor_counts(x, y, radius=encounter_radius)
        for agent_id, neighbors in zip(ids, neighbor_counts, strict=True):
            encounter_exposure_sum[agent_id] += int(neighbors)
            encounter_opportunities[agent_id] += 1
        interval_speed = np.zeros(count, dtype=np.float64)
        moved = np.zeros(count, dtype=np.bool_)
        if previous_distance is not None and previous_tick is not None:
            dt = tick - previous_tick
            if dt > 0:
                for index, agent_id in enumerate(ids):
                    if agent_id in previous_distance:
                        delta = max(
                            0.0,
                            current_distance[agent_id] - previous_distance[agent_id],
                        )
                        interval_speed[index] = delta / dt
                        moved[index] = delta > 1e-12
                        motion_opportunities[agent_id] += 1
                        motion_hits[agent_id] += int(moved[index])
                        interval_speed_sum[agent_id] += float(interval_speed[index])
        near_artifact = (
            agent_artifact_distance <= proximity_radius
            if agent_artifact_distance.size == count
            else np.zeros(count, dtype=np.bool_)
        )
        if artifact_x.size:
            for agent_id, is_near in zip(ids, near_artifact, strict=True):
                contact_opportunities[agent_id] += 1
                contact_hits[agent_id] += int(is_near)
        current_states: dict[str, int] = {}
        for agent_id, is_moving, is_near in zip(
            ids, moved, near_artifact, strict=True
        ):
            state = (
                3
                if is_moving and is_near
                else 1
                if is_moving
                else 2
                if is_near
                else 0
            )
            current_states[agent_id] = state
            artifact_bound_motion_hits[agent_id] += int(is_moving and is_near)
            if previous_states is not None and agent_id in previous_states:
                transition_counts[previous_states[agent_id], state] += 1

        moving_toward_artifact: float | None = None
        if previous_positions is not None and artifact_x.size and np.any(moved):
            moving_indices = np.nonzero(moved)[0]
            moving_indices = np.asarray(
                [
                    index
                    for index in moving_indices
                    if ids[index] in previous_positions
                ],
                dtype=np.int64,
            )
            if moving_indices.size:
                previous_x = np.asarray(
                    [previous_positions[ids[index]][0] for index in moving_indices],
                    dtype=np.float64,
                )
                previous_y = np.asarray(
                    [previous_positions[ids[index]][1] for index in moving_indices],
                    dtype=np.float64,
                )
                before = _nearest_distances(
                    previous_x, previous_y, artifact_x, artifact_y
                )
                after = agent_artifact_distance[moving_indices]
                toward = after + 1e-12 < before
                moving_toward_artifact = float(np.mean(toward))
                for index, is_toward in zip(moving_indices, toward, strict=True):
                    agent_id = ids[int(index)]
                    moving_toward_artifact_opportunities[agent_id] += 1
                    moving_toward_artifact_hits[agent_id] += int(is_toward)

        point = {
            "tick": tick,
            "population": count,
            "world_width": width,
            "world_height": height,
            "artifact_count": int(artifact_x.size),
            "mean_distance_traveled": (
                float(np.mean(distance)) if distance.size else 0.0
            ),
            "mean_distinct_cells_visited": (
                float(np.mean(visited)) if visited.size else 0.0
            ),
            "relocating_agent_fraction": relocating_fraction,
            "mean_interval_speed": (
                float(np.mean(interval_speed)) if interval_speed.size else 0.0
            ),
            "regional_switch_fraction": regional_switch_fraction,
            "spatial_entropy": _spatial_entropy(bins, count),
            "radius_of_gyration_normalized": radius / diagonal,
            "nearest_neighbor_distance_normalized": (
                _nearest_neighbor_distance(x, y) / diagonal
            ),
            "path_length_gini": gini(distance),
            "artifact_attraction": attraction(
                agent_artifact_distance, null_artifact_distance
            ),
            "artifact_proximity_fraction": artifact_near,
            "artifact_proximity_availability": artifact_available,
            "artifact_proximity_enrichment": enrichment,
            "artifact_bound_movement_fraction": (
                float(np.mean(moved & near_artifact)) if moved.size else 0.0
            ),
            "moving_toward_artifact_fraction": moving_toward_artifact,
            "encounter_exposure": mean_neighbors,
            "colocation_fraction": colocation_fraction,
            "resource_attraction": attraction(
                agent_resource_distance, null_resource_distance
            ),
        }
        points.append(point)
        if retain_positions:
            for index, (agent_id, px, py) in enumerate(zip(ids, x, y, strict=True)):
                nearest_artifact = (
                    float(agent_artifact_distance[index])
                    if agent_artifact_distance.size == count
                    else None
                )
                trajectories[agent_id].append(
                    [
                        tick,
                        int(px),
                        int(py),
                        float(distance[index]) if distance.size == count else 0.0,
                        nearest_artifact,
                    ]
                )
            compact_snapshots.append(
                {
                    "tick": tick,
                    "agents": {
                        "ids": ids,
                        "x": x.tolist(),
                        "y": y.tolist(),
                        "distance_traveled": distance.tolist(),
                    },
                    "artifacts": {
                        key: value
                        for key, value in dict(snapshot.get("artifacts", {})).items()
                        if key
                        in {
                            "count",
                            "display_count",
                            "ids",
                            "x",
                            "y",
                            "retired",
                            "name",
                        }
                    },
                }
            )
        previous_bins = current_bins
        previous_distance = current_distance
        previous_positions = current_positions
        previous_states = current_states
        previous_tick = tick

    deduplicated = {int(point["tick"]): point for point in points}
    ordered = [deduplicated[tick] for tick in sorted(deduplicated)]
    final_distance = previous_distance or {}
    agent_ids = sorted(
        set(final_distance)
        | set(successful_actions)
        | set(artifact_work)
        | set(social_interactions)
    )
    horizon = max(1, last_tick)
    agent_rows: list[dict[str, Any]] = []
    for agent_id in agent_ids:
        opportunities = contact_opportunities[agent_id]
        motion_trials = motion_opportunities[agent_id]
        toward_trials = moving_toward_artifact_opportunities[agent_id]
        encounter_trials = encounter_opportunities[agent_id]
        start = initial_positions.get(agent_id)
        stop = final_positions.get(agent_id)
        net_displacement = (
            float(math.hypot(stop[0] - start[0], stop[1] - start[1]))
            if start is not None and stop is not None
            else 0.0
        )
        distance_traveled = float(final_distance.get(agent_id, 0.0))
        activity_total = int(sum(activity_counts[agent_id].values()))
        activity_payload: dict[str, Any] = {}
        for category_index, category_name in enumerate(ACTIVITY_CATEGORIES):
            field = category_name.replace("/", "_").replace(" ", "_").replace("-", "_")
            count = int(activity_counts[agent_id][category_index])
            activity_payload[f"{field}_actions"] = count
            activity_payload[f"{field}_rate_per_100_ticks"] = (
                100.0 * count / max(1, horizon)
            )
        agent_rows.append(
            {
                "agent_id": agent_id,
                "distance_traveled": distance_traveled,
                "net_displacement": net_displacement,
                "displacement_efficiency": (
                    net_displacement / distance_traveled
                    if distance_traveled > 1e-12
                    else 0.0
                ),
                "distinct_coarse_cells": int(len(coarse_cells[agent_id])),
                "movement_snapshot_fraction": (
                    motion_hits[agent_id] / motion_trials if motion_trials else 0.0
                ),
                "mean_snapshot_speed": (
                    interval_speed_sum[agent_id] / motion_trials
                    if motion_trials
                    else 0.0
                ),
                "artifact_contact_fraction": (
                    contact_hits[agent_id] / opportunities if opportunities else None
                ),
                "artifact_bound_movement_fraction": (
                    artifact_bound_motion_hits[agent_id] / motion_trials
                    if motion_trials
                    else 0.0
                ),
                "moving_toward_artifact_fraction": (
                    moving_toward_artifact_hits[agent_id] / toward_trials
                    if toward_trials
                    else None
                ),
                "encounter_exposure_per_snapshot": (
                    encounter_exposure_sum[agent_id] / encounter_trials
                    if encounter_trials
                    else 0.0
                ),
                "artifact_work_events": int(artifact_work[agent_id]),
                "artifact_work_rate_per_100_ticks": (
                    100.0 * artifact_work[agent_id] / max(1, horizon)
                ),
                "unique_artifacts_worked": int(len(artifacts_worked[agent_id])),
                "social_interactions": int(social_interactions[agent_id]),
                "social_interaction_rate_per_100_ticks": (
                    100.0 * social_interactions[agent_id] / max(1, horizon)
                ),
                "unique_social_partners": int(len(social_partners[agent_id])),
                "successful_actions": int(successful_actions[agent_id]),
                "successful_action_rate_per_100_ticks": (
                    100.0 * activity_total / max(1, horizon)
                ),
                **activity_payload,
            }
        )
    distance_values = [row["distance_traveled"] for row in agent_rows]
    work_values = [row["artifact_work_events"] for row in agent_rows]
    social_values = [row["social_interactions"] for row in agent_rows]
    population = len(agent_rows)
    transition_probabilities = np.divide(
        transition_counts,
        transition_counts.sum(axis=1, keepdims=True),
        out=np.zeros_like(transition_counts, dtype=np.float64),
        where=transition_counts.sum(axis=1, keepdims=True) > 0,
    )
    contact_values = [
        float(row["artifact_contact_fraction"])
        for row in agent_rows
        if isinstance(row.get("artifact_contact_fraction"), (int, float))
    ]
    summary = {
        "encounter_exposure_auc": _auc(ordered, "encounter_exposure"),
        "colocation_auc": _auc(ordered, "colocation_fraction"),
        "artifact_contact_auc": _auc(ordered, "artifact_proximity_fraction"),
        "artifact_bound_movement_auc": _auc(
            ordered, "artifact_bound_movement_fraction"
        ),
        "moving_toward_artifact_auc": _auc(
            ordered, "moving_toward_artifact_fraction"
        ),
        "artifact_contact_gini": gini(contact_values),
        "artifact_work_events_per_100_agent_ticks": (
            100.0 * artifact_work_events / max(1, population * horizon)
        ),
        "explicit_interactions_per_100_agent_ticks": (
            100.0 * explicit_dyads / max(1, population * horizon)
        ),
        "mobility_artifact_work_spearman": _spearman(distance_values, work_values),
        "mobility_social_interaction_spearman": _spearman(
            distance_values, social_values
        ),
        "artifact_work_gini": gini(work_values),
        "social_interaction_gini": gini(social_values),
        "artifact_work_participation_fraction": (
            float(np.mean(np.asarray(work_values) > 0)) if work_values else 0.0
        ),
        "social_participation_fraction": (
            float(np.mean(np.asarray(social_values) > 0)) if social_values else 0.0
        ),
    }
    activity_matrix: dict[str, list[int]] | None = None
    if retain_positions:
        maximum_bin = max(0, last_tick // 24)
        activity_matrix = {}
        for agent_id in agent_ids:
            categories: list[int] = []
            for bin_index in range(maximum_bin + 1):
                counts = activity_bins[agent_id].get(bin_index, Counter())
                category = (
                    max(counts, key=lambda value: (counts[value], value))
                    if counts
                    else 0
                )
                categories.append(int(category))
            activity_matrix[agent_id] = categories
    return {
        "path": str(Path(path).resolve()),
        "points": ordered,
        "trajectories": dict(trajectories) if retain_positions else None,
        "agent_rows": agent_rows,
        "summary": summary,
        "transition_counts": transition_counts.tolist(),
        "transition_probabilities": transition_probabilities.tolist(),
        "details": (
            {
                "snapshots": compact_snapshots,
                "terrain": terrain_cache.tolist() if terrain_cache is not None else [],
                "world_width": world_width,
                "world_height": world_height,
                "activity_bin_width": 24,
                "activity_matrix": activity_matrix,
                "activity_categories": list(ACTIVITY_CATEGORIES),
                "highlight_events": highlight_events,
                "interaction_edges": [
                    {"sender": sender, "recipient": recipient, "count": count}
                    for (sender, recipient), count in interaction_edges.most_common()
                ],
                "artifact_participants": {
                    artifact_id: sorted(participants)
                    for artifact_id, participants in artifact_participants.items()
                },
            }
            if retain_positions
            else None
        ),
    }


def _episode_metrics(record: dict[str, Any]) -> dict[str, Any]:
    population = int(record.get("population_size", record.get("agents", 0)))
    initial = np.asarray(record.get("initial_positions", []), dtype=np.float64)
    final = np.asarray(record.get("final_positions", []), dtype=np.float64)
    net = np.asarray([], dtype=np.float64)
    if initial.ndim == 2 and final.shape == initial.shape and initial.shape[1:] == (2,):
        net = np.linalg.norm(final - initial, axis=1)

    mobility = dict(record.get("mobility", {}))
    distance = float(mobility.get("mean_distance_traveled", 0.0) or 0.0)
    visited = float(mobility.get("mean_distinct_cells_visited", 0.0) or 0.0)
    dynamics = list(dict(record.get("dynamics_history", {})).get("points", []))
    early_rate = _slope(dynamics, "mean_distance_traveled", 0, 200)
    late_rate = _slope(dynamics, "mean_distance_traveled", 600, 800)
    interval_rates: list[float] = []
    interval_artifacts: list[float] = []
    for previous, current in zip(dynamics, dynamics[1:], strict=False):
        dt = int(current.get("tick", 0)) - int(previous.get("tick", 0))
        if dt <= 0:
            continue
        interval_rates.append(
            (float(current.get("mean_distance_traveled", 0.0))
             - float(previous.get("mean_distance_traveled", 0.0)))
            / dt
        )
        interval_artifacts.append(float(current.get("artifact_count", 0.0)))

    result = {
        "condition": str(record.get("condition", "unknown")),
        "population_size": population,
        "seed": int(record.get("seed", -1)),
        "trace": record.get("_resolved_trace"),
        "mean_distance_traveled": distance,
        "mean_distinct_cells_visited": visited,
        "successful_moves_per_agent": (
            float(mobility.get("successful_moves", 0.0) or 0.0) / max(1, population)
        ),
        "mean_net_displacement": float(np.mean(net)) if net.size else None,
        "median_net_displacement": float(np.median(net)) if net.size else None,
        "p90_net_displacement": float(np.quantile(net, 0.9)) if net.size else None,
        "stationary_fraction": float(np.mean(net < 0.5)) if net.size else None,
        "displacement_efficiency": (
            float(np.mean(net)) / distance if net.size and distance > 1e-12 else None
        ),
        "exploration_efficiency": (
            max(0.0, visited - 1.0) / distance if distance > 1e-12 else None
        ),
        "final_regional_crowding": (
            float(mobility["regional_crowding"])
            if isinstance(mobility.get("regional_crowding"), (int, float))
            else None
        ),
        "early_mobility_rate": early_rate,
        "late_mobility_rate": late_rate,
        "late_to_early_mobility_ratio": (
            late_rate / early_rate
            if early_rate is not None and late_rate is not None and early_rate > 1e-12
            else None
        ),
        "moving_fraction_auc": _auc(dynamics, "moving_agent_fraction"),
        "crowding_auc": _auc(dynamics, "regional_crowding"),
        "movement_artifact_correlation": _pearson(interval_rates, interval_artifacts),
        "discovery_frontier_auc": dict(record.get("discovery_frontier", {})).get(
            "normalized_auc"
        ),
        "portfolio_resilience": dict(record.get("research", {})).get(
            "portfolio_resilience"
        ),
        "held_out_resilience_auc": dict(
            record.get("held_out_generalization", {})
        ).get("resilience_auc", {}).get("mean"),
        "validated_inventions": dict(record.get("technology_ecology", {})).get(
            "validated_inventions"
        ),
    }
    return result


def _augment_time_points(record: dict[str, Any]) -> list[dict[str, Any]]:
    points = list(dict(record.get("dynamics_history", {})).get("points", []))
    result: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for point in points:
        row = {
            "condition": str(record.get("condition", "unknown")),
            "population_size": int(record.get("population_size", 0)),
            "seed": int(record.get("seed", -1)),
            **point,
        }
        distance = float(point.get("mean_distance_traveled", 0.0))
        visited = float(point.get("mean_distinct_cells_visited", 1.0))
        row["exploration_efficiency"] = (
            max(0.0, visited - 1.0) / distance if distance > 1e-12 else 0.0
        )
        if previous is None:
            row["interval_mobility_rate"] = 0.0
            row["interval_new_cell_rate"] = 0.0
        else:
            dt = int(point.get("tick", 0)) - int(previous.get("tick", 0))
            row["interval_mobility_rate"] = (
                (distance - float(previous.get("mean_distance_traveled", 0.0))) / dt
                if dt > 0
                else 0.0
            )
            row["interval_new_cell_rate"] = (
                (visited - float(previous.get("mean_distinct_cells_visited", 1.0))) / dt
                if dt > 0
                else 0.0
            )
        result.append(row)
        previous = point
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _cluster_descriptors(
    centroids: np.ndarray,
    fields: list[str],
) -> list[str]:
    """Give data-driven clusters cautious, post-hoc behavioral descriptors."""

    index = {field: position for position, field in enumerate(fields)}

    def score(cluster: int, names: tuple[str, ...]) -> float:
        values = [centroids[cluster, index[name]] for name in names if name in index]
        return float(np.mean(values)) if values else 0.0

    domains = {
        "mobile exploration": (
            "distance_traveled",
            "net_displacement",
            "distinct_coarse_cells",
            "movement_snapshot_fraction",
        ),
        "artifact-centered work": (
            "artifact_contact_fraction",
            "artifact_bound_movement_fraction",
            "artifact_work_rate_per_100_ticks",
            "construction_control_rate_per_100_ticks",
        ),
        "social coordination": (
            "encounter_exposure_per_snapshot",
            "social_interaction_rate_per_100_ticks",
            "culture_coordination_rate_per_100_ticks",
        ),
        "experimental processing": (
            "observation_testing_rate_per_100_ticks",
            "material_processing_rate_per_100_ticks",
        ),
    }
    domain_scores = {
        name: np.asarray(
            [score(cluster, feature_names) for cluster in range(centroids.shape[0])]
        )
        for name, feature_names in domains.items()
    }
    activity_fields = tuple(
        field
        for field in fields
        if field.endswith("_rate_per_100_ticks")
        or field == "movement_snapshot_fraction"
    )
    mobility_fields = domains["mobile exploration"]
    domains["relatively stationary / low-activity"] = tuple()
    activity_scores = np.asarray(
        [score(cluster, activity_fields) for cluster in range(centroids.shape[0])]
    )
    mobility_scores = np.asarray(
        [score(cluster, mobility_fields) for cluster in range(centroids.shape[0])]
    )
    domain_scores["relatively stationary / low-activity"] = (
        -mobility_scores - 0.25 * activity_scores
    )
    labels: list[str] = []
    for cluster in range(centroids.shape[0]):
        advantages = {}
        for name in domains:
            peers = np.delete(domain_scores[name], cluster)
            baseline = float(np.mean(peers)) if peers.size else 0.0
            advantages[name] = float(domain_scores[name][cluster] - baseline)
        best = max(domains, key=advantages.get)
        labels.append(best)
    return labels


def _build_behavior_embedding(
    agent_rows: list[dict[str, Any]],
    *,
    random_state: int = 1729,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Embed and cluster agent behavior without using condition labels as features."""

    if len(agent_rows) < 4:
        return [], [], [], {"status": "insufficient_rows"}
    if not os.environ.get("LOKY_MAX_CPU_COUNT"):
        os.environ["LOKY_MAX_CPU_COUNT"] = "1"
    os.environ.setdefault(
        "NUMBA_CACHE_DIR",
        str(Path(tempfile.gettempdir()) / "biofoundry-numba"),
    )
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "biofoundry-matplotlib"),
    )
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import RobustScaler

    fields = [field for field, _, _ in BEHAVIOR_FEATURES]
    display = {field: label for field, label, _ in BEHAVIOR_FEATURES}
    transforms = {field: transform for field, _, transform in BEHAVIOR_FEATURES}
    raw = np.full((len(agent_rows), len(fields)), np.nan, dtype=np.float64)
    for row_index, row in enumerate(agent_rows):
        for feature_index, field in enumerate(fields):
            value = row.get(field)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                parsed = float(value)
                if transforms[field] == "log1p":
                    parsed = math.log1p(max(0.0, parsed))
                raw[row_index, feature_index] = parsed
    episode_members: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for row_index, row in enumerate(agent_rows):
        episode_members[
            (str(row["condition"]), int(row["population_size"]), int(row["seed"]))
        ].append(row_index)
    balanced_per_episode = min(len(indices) for indices in episode_members.values())
    training_indices: list[int] = []
    for key, indices in sorted(episode_members.items()):
        if len(indices) == balanced_per_episode:
            training_indices.extend(indices)
            continue
        rng = _stable_rng("behavior-training", *key, random_state)
        chosen = rng.choice(indices, size=balanced_per_episode, replace=False)
        training_indices.extend(sorted(int(value) for value in chosen))
    training_index = np.asarray(training_indices, dtype=np.int64)
    imputation: dict[str, float] = {}
    for feature_index, field in enumerate(fields):
        training_values = raw[training_index, feature_index]
        finite = training_values[np.isfinite(training_values)]
        median = float(np.median(finite)) if finite.size else 0.0
        raw[~np.isfinite(raw[:, feature_index]), feature_index] = median
        imputation[field] = median
    active = np.std(raw[training_index], axis=0) > 1e-12
    matrix_fields = [field for field, keep in zip(fields, active, strict=True) if keep]
    if len(matrix_fields) < 2:
        return [], [], [], {"status": "insufficient_feature_variation"}
    raw = raw[:, active]
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    scaler.fit(raw[training_index])
    standardized = scaler.transform(raw)
    standardized = np.nan_to_num(standardized, nan=0.0, posinf=10.0, neginf=-10.0)
    standardized = np.clip(standardized, -10.0, 10.0)

    pca = PCA(n_components=2, svd_solver="full")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        pca.fit(standardized[training_index])
        pca_coordinates = pca.transform(standardized)
    if not np.all(np.isfinite(pca_coordinates)):
        raise ValueError("non-finite PCA behavior coordinates")
    maximum_clusters = min(6, len(training_index) - 1)
    candidate_scores: dict[int, float] = {}
    candidate_models: dict[int, Any] = {}
    sample_size = min(2_000, len(training_index))
    for clusters in range(2, maximum_clusters + 1):
        with warnings.catch_warnings(), np.errstate(
            divide="ignore",
            invalid="ignore",
            over="ignore",
        ):
            warnings.filterwarnings(
                "ignore",
                message="Could not find the number of physical cores",
                category=UserWarning,
            )
            model = KMeans(
                n_clusters=clusters,
                n_init=32,
                random_state=random_state,
            ).fit(standardized[training_index])
        if not np.all(np.isfinite(model.cluster_centers_)):
            raise ValueError("non-finite behavior-cluster centroids")
        candidate_models[clusters] = model
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            candidate_scores[clusters] = float(
                silhouette_score(
                    standardized[training_index],
                    model.labels_,
                    sample_size=sample_size,
                    random_state=random_state,
                )
            )
    chosen_clusters = max(candidate_scores, key=candidate_scores.get)
    cluster_model = candidate_models[chosen_clusters]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        original_labels = np.asarray(
            cluster_model.predict(standardized), dtype=np.int64
        )
    original_centroids = np.asarray(cluster_model.cluster_centers_, dtype=np.float64)

    mobility_fields = {
        "distance_traveled",
        "net_displacement",
        "distinct_coarse_cells",
        "movement_snapshot_fraction",
    }
    mobility_indices = [
        index for index, field in enumerate(matrix_fields) if field in mobility_fields
    ]
    mobility_scores = (
        np.mean(original_centroids[:, mobility_indices], axis=1)
        if mobility_indices
        else original_centroids[:, 0]
    )
    order = np.argsort(mobility_scores, kind="stable")
    remap = {int(old): int(new) for new, old in enumerate(order)}
    labels = np.asarray([remap[int(value)] for value in original_labels], dtype=np.int64)
    centroids = original_centroids[order]
    descriptors = _cluster_descriptors(centroids, matrix_fields)

    embedding_method = "PCA fallback"
    nonlinear_coordinates = pca_coordinates.copy()
    nonlinear_parameters: dict[str, Any] = {}
    try:
        import umap

        neighbors = min(30, max(2, len(training_index) - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=neighbors,
            min_dist=0.15,
            metric="euclidean",
            random_state=random_state,
            transform_seed=random_state,
            n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            reducer.fit(standardized[training_index])
            nonlinear_coordinates = reducer.transform(standardized)
        if not np.all(np.isfinite(nonlinear_coordinates)):
            raise ValueError("non-finite UMAP behavior coordinates")
        embedding_method = "UMAP"
        nonlinear_parameters = {
            "n_neighbors": neighbors,
            "min_dist": 0.15,
            "metric": "euclidean",
            "random_state": random_state,
        }
    except (ImportError, RuntimeError):
        pass

    embedded_rows: list[dict[str, Any]] = []
    for index, source in enumerate(agent_rows):
        cluster = int(labels[index])
        embedded_rows.append(
            {
                **source,
                "behavior_cluster": cluster + 1,
                "behavior_descriptor": descriptors[cluster],
                "pca_1": float(pca_coordinates[index, 0]),
                "pca_2": float(pca_coordinates[index, 1]),
                "nonlinear_1": float(nonlinear_coordinates[index, 0]),
                "nonlinear_2": float(nonlinear_coordinates[index, 1]),
                "nonlinear_method": embedding_method,
            }
        )

    profile_rows: list[dict[str, Any]] = []
    for cluster in range(chosen_clusters):
        members = labels == cluster
        profile = {
            "behavior_cluster": cluster + 1,
            "behavior_descriptor": descriptors[cluster],
            "n_agents": int(np.sum(members)),
            "fraction": float(np.mean(members)),
        }
        for feature_index, field in enumerate(matrix_fields):
            profile[f"z_{field}"] = float(np.mean(standardized[members, feature_index]))
        profile_rows.append(profile)

    fraction_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for row in embedded_rows:
        grouped[
            (str(row["condition"]), int(row["population_size"]), int(row["seed"]))
        ].append(int(row["behavior_cluster"]))
    for (condition, population, seed), cluster_values in sorted(grouped.items()):
        for cluster in range(1, chosen_clusters + 1):
            fraction_rows.append(
                {
                    "condition": condition,
                    "population_size": population,
                    "seed": seed,
                    "behavior_cluster": cluster,
                    "behavior_descriptor": descriptors[cluster - 1],
                    "fraction": cluster_values.count(cluster) / len(cluster_values),
                }
            )

    loadings = []
    for feature_index, field in enumerate(matrix_fields):
        loadings.append(
            {
                "field": field,
                "label": display[field],
                "pc1": float(pca.components_[0, feature_index]),
                "pc2": float(pca.components_[1, feature_index]),
            }
        )
    model_report = {
        "status": "complete",
        "method_note": (
            "Clusters are descriptive agent-level phenotypes, not independent "
            "replicates or assigned simulator roles. Condition, population, seed, and "
            "agent identity were excluded from the feature matrix."
        ),
        "n_agents": len(agent_rows),
        "n_fit_agents": int(len(training_index)),
        "fit_episodes": len(episode_members),
        "balanced_agents_per_episode": balanced_per_episode,
        "fit_sampling": (
            "Equal agent count per condition-population-seed episode; deterministic "
            "without-replacement subsampling where needed. All agents are transformed "
            "and classified after fitting."
        ),
        "features": matrix_fields,
        "feature_labels": {field: display[field] for field in matrix_fields},
        "transforms": {field: transforms[field] for field in matrix_fields},
        "scaling": "median centered; divided by 5th-to-95th percentile range; clipped to [-10, 10]",
        "imputation_medians": {
            field: imputation[field] for field in matrix_fields
        },
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_loadings": loadings,
        "nonlinear_method": embedding_method,
        "nonlinear_parameters": nonlinear_parameters,
        "cluster_method": "k-means on standardized original features",
        "candidate_silhouette_scores": {
            str(key): value for key, value in candidate_scores.items()
        },
        "selected_clusters": chosen_clusters,
        "selected_silhouette": candidate_scores[chosen_clusters],
        "cluster_descriptors": {
            str(index + 1): descriptor
            for index, descriptor in enumerate(descriptors)
        },
        "random_state": random_state,
    }
    return embedded_rows, profile_rows, fraction_rows, model_report


def _build_regime_embedding(
    episodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build a four-condition PCA using only movement endpoints valid in all cases."""

    if len(episodes) < 4:
        return [], [], {"status": "insufficient_rows"}
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler

    fields = [field for field, _ in REGIME_FEATURES]
    labels = {field: label for field, label in REGIME_FEATURES}
    raw = np.full((len(episodes), len(fields)), np.nan, dtype=np.float64)
    for row_index, row in enumerate(episodes):
        for feature_index, field in enumerate(fields):
            value = row.get(field)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                raw[row_index, feature_index] = float(value)
    medians: dict[str, float] = {}
    for feature_index, field in enumerate(fields):
        finite = raw[np.isfinite(raw[:, feature_index]), feature_index]
        median = float(np.median(finite)) if finite.size else 0.0
        raw[~np.isfinite(raw[:, feature_index]), feature_index] = median
        medians[field] = median
    active = np.std(raw, axis=0) > 1e-12
    active_fields = [
        field for field, keep in zip(fields, active, strict=True) if keep
    ]
    if len(active_fields) < 2:
        return [], [], {"status": "insufficient_feature_variation"}
    raw = raw[:, active]
    scaler = RobustScaler(quantile_range=(5.0, 95.0)).fit(raw)
    standardized = scaler.transform(raw)
    standardized = np.nan_to_num(standardized, nan=0.0, posinf=10.0, neginf=-10.0)
    standardized = np.clip(standardized, -10.0, 10.0)
    pca = PCA(n_components=2, svd_solver="full")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        coordinates = pca.fit_transform(standardized)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("non-finite four-condition movement-regime coordinates")

    regime_rows: list[dict[str, Any]] = []
    for row_index, source in enumerate(episodes):
        row = {
            "condition": str(source["condition"]),
            "population_size": int(source["population_size"]),
            "seed": int(source["seed"]),
            "regime_pc1": float(coordinates[row_index, 0]),
            "regime_pc2": float(coordinates[row_index, 1]),
        }
        for feature_index, field in enumerate(active_fields):
            row[field] = float(source[field])
            row[f"z_{field}"] = float(standardized[row_index, feature_index])
        regime_rows.append(row)

    profile_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for population in sorted({int(row["population_size"]) for row in episodes}):
            selected = [
                row
                for row in regime_rows
                if row["condition"] == condition
                and int(row["population_size"]) == population
            ]
            if not selected:
                continue
            for field in active_fields:
                profile_rows.append(
                    {
                        "condition": condition,
                        "population_size": population,
                        "feature": field,
                        "feature_label": labels[field],
                        "mean": float(np.mean([row[field] for row in selected])),
                        "mean_z": float(
                            np.mean([row[f"z_{field}"] for row in selected])
                        ),
                        "n_seeds": len(selected),
                    }
                )

    loadings = [
        {
            "field": field,
            "label": labels[field],
            "pc1": float(pca.components_[0, index]),
            "pc2": float(pca.components_[1, index]),
        }
        for index, field in enumerate(active_fields)
    ]
    report = {
        "status": "complete",
        "method_note": (
            "One row is one independent simulation seed. The common movement feature "
            "set is defined for shared societies and isolated one-agent worlds. "
            "Condition, population size, and seed are excluded from PCA fitting."
        ),
        "n_episodes": len(episodes),
        "features": active_fields,
        "feature_labels": {field: labels[field] for field in active_fields},
        "imputation_medians": {field: medians[field] for field in active_fields},
        "scaling": "median centered; divided by 5th-to-95th percentile range",
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_loadings": loadings,
    }
    return regime_rows, profile_rows, report


def _group_estimates(
    episodes: list[dict[str, Any]], *, resamples: int
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        grouped[(episode["condition"], episode["population_size"])].append(episode)
    result: dict[str, Any] = defaultdict(dict)
    for (condition, population), rows in sorted(grouped.items()):
        metrics: dict[str, Any] = {}
        for metric in ENDPOINT_METRICS:
            values = [
                float(row[metric])
                for row in rows
                if isinstance(row.get(metric), (int, float))
                and math.isfinite(float(row[metric]))
            ]
            metrics[metric] = _mean_ci(
                values,
                key=("endpoint", condition, population, metric),
                resamples=resamples,
            )
        result[condition][str(population)] = {
            "n_seeds": len(rows),
            "metrics": metrics,
        }
    return dict(result)


def _paired_contrasts(
    episodes: list[dict[str, Any]], *, resamples: int
) -> list[dict[str, Any]]:
    index = {
        (row["condition"], row["population_size"], row["seed"]): row
        for row in episodes
    }
    populations = sorted({row["population_size"] for row in episodes})
    rows: list[dict[str, Any]] = []
    for population in populations:
        for first, second in combinations(CONDITIONS, 2):
            for metric in ENDPOINT_METRICS:
                paired: list[float] = []
                seeds: list[int] = []
                for seed in sorted({row["seed"] for row in episodes}):
                    left = index.get((first, population, seed))
                    right = index.get((second, population, seed))
                    if left is None or right is None:
                        continue
                    left_value = left.get(metric)
                    right_value = right.get(metric)
                    if not isinstance(left_value, (int, float)) or not isinstance(
                        right_value, (int, float)
                    ):
                        continue
                    if not math.isfinite(float(left_value)) or not math.isfinite(
                        float(right_value)
                    ):
                        continue
                    paired.append(float(left_value) - float(right_value))
                    seeds.append(seed)
                estimate = _mean_ci(
                    paired,
                    key=("contrast", population, first, second, metric),
                    resamples=resamples,
                )
                rows.append(
                    {
                        "population_size": population,
                        "condition_a": first,
                        "condition_b": second,
                        "metric": metric,
                        "a_minus_b": estimate["mean"],
                        "ci95_low": estimate["ci95_low"],
                        "ci95_high": estimate["ci95_high"],
                        "n_pairs": estimate["n"],
                        "seeds": ";".join(map(str, seeds)),
                    }
                )
    return rows


def _behavior_fraction_contrasts(
    fraction_rows: list[dict[str, Any]],
    *,
    resamples: int,
) -> list[dict[str, Any]]:
    """Return paired seed-level contrasts for descriptive behavior prevalence."""

    index = {
        (
            str(row["condition"]),
            int(row["population_size"]),
            int(row["seed"]),
            int(row["behavior_cluster"]),
        ): row
        for row in fraction_rows
    }
    populations = sorted({int(row["population_size"]) for row in fraction_rows})
    clusters = sorted({int(row["behavior_cluster"]) for row in fraction_rows})
    seeds = sorted({int(row["seed"]) for row in fraction_rows})
    output: list[dict[str, Any]] = []
    for population in populations:
        for cluster in clusters:
            for first, second in combinations(SHARED_CONDITIONS, 2):
                differences: list[float] = []
                paired_seeds: list[int] = []
                descriptor = ""
                for seed in seeds:
                    left = index.get((first, population, seed, cluster))
                    right = index.get((second, population, seed, cluster))
                    if left is None or right is None:
                        continue
                    differences.append(float(left["fraction"]) - float(right["fraction"]))
                    paired_seeds.append(seed)
                    descriptor = str(left["behavior_descriptor"])
                estimate = _mean_ci(
                    differences,
                    key=(
                        "behavior-contrast",
                        population,
                        cluster,
                        first,
                        second,
                    ),
                    resamples=resamples,
                )
                output.append(
                    {
                        "population_size": population,
                        "behavior_cluster": cluster,
                        "behavior_descriptor": descriptor,
                        "condition_a": first,
                        "condition_b": second,
                        "mean_difference_a_minus_b": estimate["mean"],
                        "ci95_low": estimate["ci95_low"],
                        "ci95_high": estimate["ci95_high"],
                        "n_paired_seeds": estimate["n"],
                        "paired_seeds": ";".join(map(str, paired_seeds)),
                    }
                )
    return output


def _save_figure(figure: Any, destination: Path, stem: str) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for suffix, options in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        path = destination / f"{stem}.{suffix}"
        figure.savefig(path, facecolor="white", bbox_inches="tight", **options)
        outputs[suffix] = str(path.resolve())
    return outputs


def _curve(
    axis: Any,
    rows: list[dict[str, Any]],
    *,
    population: int,
    metric: str,
    ylabel: str,
    panel: str,
) -> None:
    for condition in SHARED_CONDITIONS:
        selected = [
            row
            for row in rows
            if row["condition"] == condition and row["population_size"] == population
        ]
        ticks = sorted({int(row["tick"]) for row in selected})
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        plotted_ticks: list[int] = []
        for tick in ticks:
            values = [
                float(row[metric])
                for row in selected
                if int(row["tick"]) == tick
                and isinstance(row.get(metric), (int, float))
                and math.isfinite(float(row[metric]))
            ]
            if not values:
                continue
            estimate = _mean_ci(
                values,
                key=("curve", condition, population, metric, tick),
                resamples=5_000,
            )
            plotted_ticks.append(tick)
            means.append(float(estimate["mean"]))
            lows.append(float(estimate["ci95_low"]))
            highs.append(float(estimate["ci95_high"]))
        if not plotted_ticks:
            continue
        axis.plot(
            plotted_ticks,
            means,
            color=COLORS[condition],
            linewidth=1.9,
            label=LABELS[condition],
        )
        axis.fill_between(
            plotted_ticks,
            lows,
            highs,
            color=COLORS[condition],
            alpha=0.14,
            linewidth=0,
        )
    axis.set_title(panel, loc="left")
    axis.set_xlabel("Simulation tick")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)


def _render_dynamics_overview(
    time_rows: list[dict[str, Any]],
    spatial_rows: list[dict[str, Any]],
    destination: Path,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.2))
    _curve(
        axes[0, 0],
        time_rows,
        population=200,
        metric="mean_distance_traveled",
        ylabel="Mean cumulative path length (cells)",
        panel="A",
    )
    _curve(
        axes[0, 1],
        time_rows,
        population=200,
        metric="mean_distinct_cells_visited",
        ylabel="Mean distinct cells visited",
        panel="B",
    )
    _curve(
        axes[1, 0],
        time_rows,
        population=200,
        metric="regional_crowding",
        ylabel="Regional crowding probability",
        panel="C",
    )
    _curve(
        axes[1, 1],
        spatial_rows,
        population=200,
        metric="artifact_attraction",
        ylabel="Artifact attraction index",
        panel="D",
    )
    axes[1, 1].axhline(0, color="#7d8c87", linewidth=0.9, linestyle="--")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
    )
    figure.subplots_adjust(bottom=0.14, hspace=0.38, wspace=0.31)
    return _save_figure(figure, destination, "movement-dynamics-overview")


def _render_scaling(
    episodes: list[dict[str, Any]],
    estimates: dict[str, Any],
    destination: Path,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    specifications = (
        ("mean_distance_traveled", "Mean path length (cells)"),
        ("mean_distinct_cells_visited", "Mean distinct cells visited"),
        ("mean_net_displacement", "Mean net displacement (cells)"),
        ("exploration_efficiency", "New cells per path cell"),
        ("displacement_efficiency", "Net displacement / path length"),
        ("final_regional_crowding", "Regional crowding probability"),
    )
    populations = sorted({int(row["population_size"]) for row in episodes})
    figure, axes = plt.subplots(2, 3, figsize=(12.2, 6.8))
    for panel_index, (axis, (metric, ylabel)) in enumerate(
        zip(axes.ravel(), specifications, strict=True)
    ):
        for condition in CONDITIONS:
            x: list[int] = []
            mean: list[float] = []
            low: list[float] = []
            high: list[float] = []
            for population in populations:
                estimate = (
                    estimates.get(condition, {})
                    .get(str(population), {})
                    .get("metrics", {})
                    .get(metric, {})
                )
                if estimate.get("mean") is None:
                    continue
                x.append(population)
                mean.append(float(estimate["mean"]))
                low.append(float(estimate["ci95_low"]))
                high.append(float(estimate["ci95_high"]))
            if not x:
                continue
            values = np.asarray(mean)
            axis.errorbar(
                x,
                values,
                yerr=np.vstack((values - np.asarray(low), np.asarray(high) - values)),
                marker=MARKERS[condition],
                color=COLORS[condition],
                linewidth=1.6,
                markersize=4.6,
                capsize=2.5,
                label=LABELS[condition],
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(populations, [str(value) for value in populations])
        axis.set_xlabel("Number of agents, N")
        axis.set_ylabel(ylabel)
        axis.set_title(chr(ord("A") + panel_index), loc="left")
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        frameon=False,
    )
    figure.subplots_adjust(bottom=0.16, hspace=0.42, wspace=0.36)
    return _save_figure(figure, destination, "movement-population-scaling")


def _render_self_organization(
    episodes: list[dict[str, Any]],
    estimates: dict[str, Any],
    destination: Path,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.0))
    populations = sorted({int(row["population_size"]) for row in episodes})
    specs = (
        ("path_length_gini", "Path-length Gini", "A"),
        ("final_spatial_entropy", "Normalized spatial entropy", "B"),
        ("final_artifact_attraction", "Artifact attraction index", "C"),
        ("mean_regional_switch_fraction", "Regional switches / snapshot", "D"),
    )
    for axis, (metric, ylabel, panel) in zip(axes.ravel(), specs, strict=True):
        for condition in SHARED_CONDITIONS:
            mean: list[float] = []
            low: list[float] = []
            high: list[float] = []
            x: list[int] = []
            for population in populations:
                estimate = (
                    estimates.get(condition, {})
                    .get(str(population), {})
                    .get("metrics", {})
                    .get(metric, {})
                )
                if estimate.get("mean") is None:
                    continue
                x.append(population)
                mean.append(float(estimate["mean"]))
                low.append(float(estimate["ci95_low"]))
                high.append(float(estimate["ci95_high"]))
            if not x:
                continue
            values = np.asarray(mean)
            axis.errorbar(
                x,
                values,
                yerr=np.vstack((values - np.asarray(low), np.asarray(high) - values)),
                marker=MARKERS[condition],
                color=COLORS[condition],
                linewidth=1.6,
                markersize=4.6,
                capsize=2.5,
                label=LABELS[condition],
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(populations, [str(value) for value in populations])
        axis.set_xlabel("Number of agents, N")
        axis.set_ylabel(ylabel)
        axis.set_title(panel, loc="left")
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
    )
    figure.subplots_adjust(bottom=0.15, hspace=0.42, wspace=0.32)
    return _save_figure(figure, destination, "movement-self-organization")


def _terrain_palette() -> tuple[Any, Any]:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    colors = [
        "#245c72",
        "#6e9faa",
        "#9ab477",
        "#708c5f",
        "#b9a767",
        "#a3b989",
        "#baa474",
        "#b79068",
        "#c2ad82",
    ]
    return ListedColormap(colors), BoundaryNorm(np.arange(-0.5, 9.5, 1), 9)


def _snapshot_payloads(path: str | Path) -> list[dict[str, Any]]:
    snapshots: dict[int, dict[str, Any]] = {}
    for record in read_records(path):
        if record.get("type") == "snapshot":
            snapshot = dict(record.get("snapshot", {}))
            snapshots[int(snapshot.get("tick", 0))] = snapshot
    return [snapshots[tick] for tick in sorted(snapshots)]


def _render_trajectory_atlas(
    representative: dict[str, dict[str, Any]],
    destination: Path,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    cmap, norm = _terrain_palette()
    for axis, condition in zip(axes, SHARED_CONDITIONS, strict=True):
        item = representative[condition]
        snapshots = _snapshot_payloads(item["trace"])
        first = snapshots[0]
        final = snapshots[-1]
        terrain = _world_grid(dict(first["world"]), "terrain", dtype=np.int16)
        axis.imshow(terrain, cmap=cmap, norm=norm, origin="upper", alpha=0.82)
        ids = [str(value) for value in first["agents"]["ids"]]
        selected_indices = np.linspace(0, len(ids) - 1, min(24, len(ids)), dtype=int)
        for index in selected_indices:
            path_x = [int(snapshot["agents"]["x"][index]) for snapshot in snapshots]
            path_y = [int(snapshot["agents"]["y"][index]) for snapshot in snapshots]
            axis.plot(path_x, path_y, color=COLORS[condition], alpha=0.44, linewidth=0.75)
            axis.scatter(
                path_x[0],
                path_y[0],
                s=8,
                color="#ffffff",
                edgecolor="#17302a",
                linewidth=0.4,
                zorder=5,
            )
            axis.scatter(
                path_x[-1],
                path_y[-1],
                s=11,
                color=COLORS[condition],
                edgecolor="#17302a",
                linewidth=0.35,
                zorder=5,
            )
        agents = dict(final["agents"])
        axis.scatter(
            agents["x"],
            agents["y"],
            s=4,
            color=COLORS[condition],
            alpha=0.28,
            linewidths=0,
            label="final agent position",
        )
        artifact_x, artifact_y = _active_artifacts(final)
        axis.scatter(
            artifact_x,
            artifact_y,
            marker="D",
            s=17,
            color="#e1a83a",
            edgecolor="#17302a",
            linewidth=0.35,
            zorder=6,
            label="artifact",
        )
        metrics = item["episode"]
        attraction = metrics.get("final_artifact_attraction")
        attraction_label = (
            f"; attraction {attraction:.2f}"
            if isinstance(attraction, (int, float))
            else ""
        )
        axis.set_title(
            f"{LABELS[condition]}\n"
            f"path {metrics['mean_distance_traveled']:.1f}; "
            f"cells {metrics['mean_distinct_cells_visited']:.1f}"
            f"{attraction_label}",
            fontsize=9,
        )
        axis.set_xlim(-0.5, terrain.shape[1] - 0.5)
        axis.set_ylim(terrain.shape[0] - 0.5, -0.5)
        axis.set_xlabel("World x (cells)")
        axis.set_ylabel("World y (cells)")
        axis.set_aspect("equal")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    return _save_figure(figure, destination, "movement-trajectory-atlas")


def _nearest_snapshot(snapshots: list[dict[str, Any]], tick: int) -> dict[str, Any]:
    return min(snapshots, key=lambda item: abs(int(item.get("tick", 0)) - tick))


def _render_occupancy_evolution(
    representative: dict[str, dict[str, Any]],
    destination: Path,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    target_ticks = (0, 400, 800)
    figure, axes = plt.subplots(3, 3, figsize=(10.4, 8.4), constrained_layout=True)
    maximum = 1.0
    matrices: dict[tuple[str, int], tuple[np.ndarray, dict[str, Any]]] = {}
    for condition in SHARED_CONDITIONS:
        snapshots = _snapshot_payloads(representative[condition]["trace"])
        for tick in target_ticks:
            snapshot = _nearest_snapshot(snapshots, tick)
            agents = dict(snapshot["agents"])
            height = int(snapshot["world"]["height"])
            width = int(snapshot["world"]["width"])
            histogram, _, _ = np.histogram2d(
                agents["y"],
                agents["x"],
                bins=(18, 24),
                range=((0, height), (0, width)),
            )
            matrices[(condition, tick)] = (histogram, snapshot)
            maximum = max(maximum, float(np.max(histogram)))
    for row, condition in enumerate(SHARED_CONDITIONS):
        for column, tick in enumerate(target_ticks):
            axis = axes[row, column]
            matrix, snapshot = matrices[(condition, tick)]
            axis.imshow(
                matrix,
                origin="upper",
                extent=(0, snapshot["world"]["width"], snapshot["world"]["height"], 0),
                cmap="magma",
                vmin=0,
                vmax=maximum,
                interpolation="bilinear",
                aspect="equal",
            )
            artifact_x, artifact_y = _active_artifacts(snapshot)
            axis.scatter(
                artifact_x,
                artifact_y,
                marker="D",
                s=11,
                color="#37b6a2",
                edgecolor="white",
                linewidth=0.35,
            )
            if row == 0:
                axis.set_title(f"Tick {int(snapshot['tick'])}")
            if column == 0:
                axis.set_ylabel(LABELS[condition])
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
    return _save_figure(figure, destination, "movement-occupancy-evolution")


def _focal_agents(trace: dict[str, Any], limit: int = 18) -> list[str]:
    rows = list(trace.get("agent_rows", []))
    rankings = (
        sorted(rows, key=lambda row: (-row["distance_traveled"], row["agent_id"])),
        sorted(rows, key=lambda row: (-row["artifact_work_events"], row["agent_id"])),
        sorted(rows, key=lambda row: (-row["social_interactions"], row["agent_id"])),
        sorted(rows, key=lambda row: (-row["successful_actions"], row["agent_id"])),
    )
    selected: list[str] = []
    depth = 0
    while len(selected) < min(limit, len(rows)):
        added = False
        for ranking in rankings:
            if depth >= len(ranking):
                continue
            agent_id = str(ranking[depth]["agent_id"])
            if agent_id not in selected:
                selected.append(agent_id)
                added = True
                if len(selected) >= limit:
                    break
        depth += 1
        if not added and depth >= len(rows):
            break
    return selected


def _event_position(
    event: dict[str, Any], trajectories: dict[str, list[list[float]]]
) -> tuple[float, float] | None:
    if isinstance(event.get("x"), (int, float)) and isinstance(
        event.get("y"), (int, float)
    ):
        return float(event["x"]), float(event["y"])
    agent_id = event.get("agent")
    points = trajectories.get(str(agent_id), [])
    if not points:
        return None
    target_tick = int(event.get("tick", 0))
    point = min(points, key=lambda item: abs(int(item[0]) - target_tick))
    return float(point[1]), float(point[2])


def _render_speed_interaction_atlas(
    representative: dict[str, dict[str, Any]],
    destination: Path,
) -> dict[str, str]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(1, 3, figsize=(13.8, 5.2))
    cmap, terrain_norm = _terrain_palette()
    selected_by_condition: dict[str, list[str]] = {}
    speeds: list[float] = []
    for condition in SHARED_CONDITIONS:
        trace = representative[condition]["trace"]
        selected = _focal_agents(trace)
        selected_by_condition[condition] = selected
        for agent_id in selected:
            points = trace["trajectories"].get(agent_id, [])
            for previous, current in zip(points, points[1:], strict=False):
                dt = max(1, int(current[0]) - int(previous[0]))
                speeds.append(max(0.0, float(current[3]) - float(previous[3])) / dt)
    upper = float(np.quantile(speeds, 0.95)) if speeds else 1.0
    speed_norm = mpl.colors.Normalize(vmin=0.0, vmax=max(upper, 1e-6))
    event_styles = {
        "observation": ("o", "#4c8fb5", "observation / test"),
        "construction": ("D", "#e1a83a", "artifact built"),
        "program": ("^", "#8a5ea7", "program installed"),
        "social": ("*", "#d8664b", "explicit interaction"),
    }

    for axis, condition in zip(axes, SHARED_CONDITIONS, strict=True):
        trace = representative[condition]["trace"]
        details = dict(trace.get("details") or {})
        terrain = np.asarray(details.get("terrain", []), dtype=np.int16)
        if terrain.ndim == 2:
            axis.imshow(terrain, cmap=cmap, norm=terrain_norm, origin="upper", alpha=0.72)
        trajectories = dict(trace.get("trajectories") or {})
        focal = set(selected_by_condition[condition])
        for agent_id in selected_by_condition[condition]:
            points = trajectories.get(agent_id, [])
            if len(points) < 2:
                continue
            segments = []
            segment_speeds = []
            for previous, current in zip(points, points[1:], strict=False):
                segments.append(
                    [
                        (float(previous[1]), float(previous[2])),
                        (float(current[1]), float(current[2])),
                    ]
                )
                dt = max(1, int(current[0]) - int(previous[0]))
                segment_speeds.append(
                    max(0.0, float(current[3]) - float(previous[3])) / dt
                )
            collection = LineCollection(
                segments,
                cmap="viridis",
                norm=speed_norm,
                linewidths=1.15,
                alpha=0.76,
                zorder=3,
            )
            collection.set_array(np.asarray(segment_speeds, dtype=np.float64))
            axis.add_collection(collection)
            axis.scatter(
                points[0][1],
                points[0][2],
                s=10,
                color="white",
                edgecolor="#17302a",
                linewidth=0.45,
                zorder=6,
            )

        grouped_events: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for event in details.get("highlight_events", []):
            if event.get("agent") not in focal:
                continue
            kind = str(event.get("kind", ""))
            group = (
                "construction"
                if kind == "artifact_built"
                else "program"
                if kind == "artifact_program_installed"
                else "social"
                if kind in {"message_delivered", "resource_traded", "knowledge_taught"}
                else "observation"
            )
            position = _event_position(event, trajectories)
            if position is not None:
                grouped_events[group].append(position)
        for group, positions in grouped_events.items():
            if len(positions) > 80:
                indices = np.linspace(0, len(positions) - 1, 80, dtype=int)
                positions = [positions[index] for index in indices]
            marker, color, _ = event_styles[group]
            axis.scatter(
                [position[0] for position in positions],
                [position[1] for position in positions],
                marker=marker,
                s=17 if group != "social" else 24,
                color=color,
                edgecolor="white",
                linewidth=0.3,
                alpha=0.8,
                zorder=7,
            )

        snapshots = list(details.get("snapshots", []))
        if snapshots:
            final = snapshots[-1]
            artifacts = dict(final.get("artifacts", {}))
            count = int(artifacts.get("display_count", artifacts.get("count", 0)))
            ids = [str(value) for value in artifacts.get("ids", [])[:count]]
            retired = list(artifacts.get("retired", [False] * count))[:count]
            participants = dict(details.get("artifact_participants", {}))
            for artifact_id, px, py, is_retired in zip(
                ids,
                artifacts.get("x", [])[:count],
                artifacts.get("y", [])[:count],
                retired,
                strict=True,
            ):
                if is_retired:
                    continue
                size = 19 + 8 * math.sqrt(max(1, len(participants.get(artifact_id, []))))
                axis.scatter(
                    px,
                    py,
                    marker="s",
                    s=size,
                    color="#17302a",
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=8,
                )
        axis.set_title(LABELS[condition], fontsize=10)
        axis.set_xlim(-0.5, max(0.5, float(details.get("world_width", 1)) - 0.5))
        axis.set_ylim(max(0.5, float(details.get("world_height", 1)) - 0.5), -0.5)
        axis.set_xlabel("World x (cells)")
        axis.set_ylabel("World y (cells)")
        axis.set_aspect("equal")

    figure.subplots_adjust(
        left=0.055,
        right=0.995,
        top=0.91,
        bottom=0.33,
        wspace=0.24,
    )
    scalar = mpl.cm.ScalarMappable(norm=speed_norm, cmap="viridis")
    colorbar_axis = figure.add_axes((0.35, 0.175, 0.30, 0.022))
    colorbar = figure.colorbar(
        scalar,
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.set_label("Path speed (cells per tick)")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=6,
            label=label,
        )
        for marker, color, label in event_styles.values()
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#17302a",
            markersize=6,
            label="active artifact (size: participants)",
        )
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        frameon=False,
    )
    return _save_figure(figure, destination, "movement-speed-interaction-atlas")


def _render_activity_raster(
    representative: dict[str, dict[str, Any]],
    destination: Path,
) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    figure, axes = plt.subplots(3, 1, figsize=(10.8, 7.4), constrained_layout=True)
    cmap = ListedColormap(ACTIVITY_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(ACTIVITY_CATEGORIES) + 0.5), cmap.N)
    for axis, condition in zip(axes, SHARED_CONDITIONS, strict=True):
        trace = representative[condition]["trace"]
        details = dict(trace.get("details") or {})
        matrix_by_agent = dict(details.get("activity_matrix") or {})
        rows = {str(row["agent_id"]): row for row in trace.get("agent_rows", [])}
        ordered_agents = sorted(
            matrix_by_agent,
            key=lambda agent_id: (
                -int(rows.get(agent_id, {}).get("artifact_work_events", 0)),
                -int(rows.get(agent_id, {}).get("social_interactions", 0)),
                -float(rows.get(agent_id, {}).get("distance_traveled", 0.0)),
                agent_id,
            ),
        )
        matrix = np.asarray(
            [matrix_by_agent[agent_id] for agent_id in ordered_agents], dtype=np.int16
        )
        width = int(details.get("activity_bin_width", 24))
        axis.imshow(
            matrix,
            cmap=cmap,
            norm=norm,
            origin="upper",
            interpolation="nearest",
            aspect="auto",
            extent=(0, matrix.shape[1] * width, matrix.shape[0], 0),
        )
        axis.set_title(LABELS[condition], loc="left", fontsize=10)
        axis.set_ylabel("Agents (ranked)")
        axis.set_xlabel("Simulation tick")
        axis.set_yticks([0, matrix.shape[0] - 1], ["1", str(matrix.shape[0])])
    figure.legend(
        handles=[
            Patch(facecolor=color, edgecolor="none", label=label)
            for color, label in zip(ACTIVITY_COLORS, ACTIVITY_CATEGORIES, strict=True)
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        frameon=False,
    )
    return _save_figure(figure, destination, "movement-activity-raster")


def _render_state_transitions(
    transition_rows: list[dict[str, Any]],
    destination: Path,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.05), constrained_layout=True)
    images = []
    for axis, condition in zip(axes, SHARED_CONDITIONS, strict=True):
        selected = [
            row
            for row in transition_rows
            if row["condition"] == condition and int(row["population_size"]) == 200
        ]
        matrix = np.zeros((len(MOBILITY_STATES), len(MOBILITY_STATES)), dtype=np.float64)
        for source_index, source_state in enumerate(MOBILITY_STATES):
            for target_index, target_state in enumerate(MOBILITY_STATES):
                values = [
                    float(row["probability"])
                    for row in selected
                    if row["source_state"] == source_state
                    and row["target_state"] == target_state
                ]
                matrix[source_index, target_index] = float(np.mean(values)) if values else 0.0
        image = axis.imshow(matrix, cmap="magma", vmin=0.0, vmax=1.0)
        images.append(image)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > 0.45 else "#17302a",
                    fontsize=7.5,
                )
        labels = ["stationary", "exploring", "artifact-local", "artifact-bound"]
        axis.set_xticks(range(4), labels, rotation=35, ha="right")
        axis.set_yticks(range(4), labels)
        axis.set_xlabel("Next snapshot state")
        axis.set_ylabel("Current snapshot state")
        axis.set_title(LABELS[condition], fontsize=10)
    colorbar = figure.colorbar(images[-1], ax=axes, location="bottom", shrink=0.45, pad=0.16)
    colorbar.set_label("Transition probability (mean across four seeds)")
    return _save_figure(figure, destination, "movement-state-transitions")


def _render_interaction_scaling(
    estimates: dict[str, Any],
    destination: Path,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    specifications = (
        ("encounter_exposure_auc", "Mean neighbors within 2 cells", "A"),
        ("artifact_contact_auc", "Agents within 3 cells of artifact", "B"),
        ("artifact_bound_movement_auc", "Moving and artifact-local fraction", "C"),
        (
            "artifact_work_events_per_100_agent_ticks",
            "Technology-work events / 100 agent-ticks",
            "D",
        ),
        ("artifact_contact_gini", "Artifact-contact Gini", "E"),
        (
            "mobility_artifact_work_spearman",
            "Mobility–technology-work Spearman ρ",
            "F",
        ),
    )
    populations = (50, 100, 200)
    figure, axes = plt.subplots(2, 3, figsize=(12.2, 6.8))
    for axis, (metric, ylabel, panel) in zip(axes.ravel(), specifications, strict=True):
        for condition in SHARED_CONDITIONS:
            x: list[int] = []
            means: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for population in populations:
                estimate = (
                    estimates.get(condition, {})
                    .get(str(population), {})
                    .get("metrics", {})
                    .get(metric, {})
                )
                if estimate["mean"] is None:
                    continue
                x.append(population)
                means.append(float(estimate["mean"]))
                lows.append(float(estimate["ci95_low"]))
                highs.append(float(estimate["ci95_high"]))
            values = np.asarray(means)
            axis.errorbar(
                x,
                values,
                yerr=np.vstack((values - np.asarray(lows), np.asarray(highs) - values)),
                marker=MARKERS[condition],
                color=COLORS[condition],
                linewidth=1.6,
                markersize=4.6,
                capsize=2.5,
                label=LABELS[condition],
            )
        if metric.endswith("spearman"):
            axis.axhline(0, color="#7d8c87", linewidth=0.8, linestyle="--")
        axis.set_xscale("log", base=2)
        axis.set_xticks(populations, [str(value) for value in populations])
        axis.set_xlabel("Number of agents, N")
        axis.set_ylabel(ylabel)
        axis.set_title(panel, loc="left")
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
    )
    figure.subplots_adjust(bottom=0.16, hspace=0.42, wspace=0.32)
    return _save_figure(figure, destination, "movement-interaction-scaling")


def _render_social_topology(
    full_trace: dict[str, Any],
    destination: Path,
) -> dict[str, str]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    details = dict(full_trace.get("details") or {})
    rows = {str(row["agent_id"]): row for row in full_trace.get("agent_rows", [])}
    edges = list(details.get("interaction_edges", []))
    participants = Counter()
    for edge in edges:
        participants[str(edge["sender"])] += int(edge["count"])
        participants[str(edge["recipient"])] += int(edge["count"])
    selected = [agent_id for agent_id, _ in participants.most_common(36)]
    selected_set = set(selected)
    selected_edges = [
        edge
        for edge in edges
        if edge["sender"] in selected_set and edge["recipient"] in selected_set
    ][:90]
    angles = np.linspace(0, 2 * np.pi, max(1, len(selected)), endpoint=False)
    positions = {
        agent_id: (math.cos(angle), math.sin(angle))
        for agent_id, angle in zip(selected, angles, strict=True)
    }
    distances = np.asarray(
        [float(rows.get(agent_id, {}).get("distance_traveled", 0.0)) for agent_id in selected]
    )
    minimum_distance = float(np.min(distances)) if distances.size else 0.0
    maximum_distance = float(np.max(distances)) if distances.size else 1.0
    norm = mpl.colors.Normalize(
        vmin=minimum_distance,
        vmax=max(maximum_distance, minimum_distance + 1e-6),
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), constrained_layout=True)
    network_axis, scatter_axis = axes
    maximum_edge = max([int(edge["count"]) for edge in selected_edges], default=1)
    for index, edge in enumerate(selected_edges):
        start = positions[str(edge["sender"])]
        stop = positions[str(edge["recipient"])]
        rad = 0.12 if index % 2 == 0 else -0.12
        arrow = FancyArrowPatch(
            start,
            stop,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-",
            linewidth=0.25 + 1.4 * int(edge["count"]) / maximum_edge,
            color="#7d8c87",
            alpha=0.22,
            zorder=1,
        )
        network_axis.add_patch(arrow)
    node_x = [positions[agent_id][0] for agent_id in selected]
    node_y = [positions[agent_id][1] for agent_id in selected]
    sizes = [24 + 5 * math.sqrt(participants[agent_id]) for agent_id in selected]
    scatter = network_axis.scatter(
        node_x,
        node_y,
        s=sizes,
        c=distances,
        cmap="viridis",
        norm=norm,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    network_axis.set_title("A", loc="left")
    network_axis.set_aspect("equal")
    network_axis.set_axis_off()
    colorbar = figure.colorbar(scatter, ax=network_axis, location="bottom", shrink=0.65)
    colorbar.set_label("Agent path length (cells)")

    all_rows = list(rows.values())
    x = np.asarray([row["distance_traveled"] for row in all_rows], dtype=np.float64)
    y = np.asarray([row["social_interactions"] for row in all_rows], dtype=np.float64)
    work = np.asarray([row["artifact_work_events"] for row in all_rows], dtype=np.float64)
    scatter_axis.scatter(
        x,
        y,
        s=15 + 7 * np.sqrt(work),
        color=COLORS["full"],
        alpha=0.48,
        edgecolor="white",
        linewidth=0.35,
    )
    top_agents = sorted(
        all_rows,
        key=lambda row: (-row["social_interactions"], row["agent_id"]),
    )[:5]
    for row in top_agents:
        scatter_axis.annotate(
            str(row["agent_id"]).replace("agent_", "A"),
            (row["distance_traveled"], row["social_interactions"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    rho = full_trace["summary"].get("mobility_social_interaction_spearman")
    scatter_axis.text(
        0.98,
        0.96,
        f"seed-level agent correlation ρ = {rho:.2f}" if isinstance(rho, float) else "ρ undefined",
        transform=scatter_axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    scatter_axis.set_title("B", loc="left")
    scatter_axis.set_xlabel("Agent path length (cells)")
    scatter_axis.set_ylabel("Explicit interaction participations")
    scatter_axis.grid(alpha=0.18)
    scatter_axis.spines[["top", "right"]].set_visible(False)
    return _save_figure(figure, destination, "movement-social-topology")


def _render_behavior_embedding(
    embedded_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    fraction_rows: list[dict[str, Any]],
    model_report: dict[str, Any],
    destination: Path,
) -> dict[str, str]:
    """Render a transparent linear/nonlinear behavioral phenotype atlas."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(2, 2, figsize=(12.6, 9.2))
    clusters = sorted({int(row["behavior_cluster"]) for row in embedded_rows})
    descriptors = {
        int(row["behavior_cluster"]): str(row["behavior_descriptor"])
        for row in profile_rows
    }

    def scatter_embedding(
        axis: Any,
        x_field: str,
        y_field: str,
        x_label: str,
        y_label: str,
        panel: str,
    ) -> None:
        for cluster in clusters:
            selected = [
                row for row in embedded_rows if int(row["behavior_cluster"]) == cluster
            ]
            axis.scatter(
                [float(row[x_field]) for row in selected],
                [float(row[y_field]) for row in selected],
                s=7,
                alpha=0.28,
                linewidths=0,
                color=PHENOTYPE_COLORS[(cluster - 1) % len(PHENOTYPE_COLORS)],
                rasterized=True,
            )
            center_x = float(np.mean([float(row[x_field]) for row in selected]))
            center_y = float(np.mean([float(row[y_field]) for row in selected]))
            axis.text(
                center_x,
                center_y,
                f"C{cluster}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": PHENOTYPE_COLORS[
                        (cluster - 1) % len(PHENOTYPE_COLORS)
                    ],
                    "linewidth": 0.8,
                    "boxstyle": "round,pad=0.2",
                },
            )
        axis.set_title(panel, loc="left")
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.12)
        axis.spines[["top", "right"]].set_visible(False)

    explained = model_report.get("pca_explained_variance_ratio", [0.0, 0.0])
    scatter_embedding(
        axes[0, 0],
        "pca_1",
        "pca_2",
        f"PC1 ({100 * float(explained[0]):.1f}% variance)",
        f"PC2 ({100 * float(explained[1]):.1f}% variance)",
        "A",
    )
    loadings = sorted(
        model_report.get("pca_loadings", []),
        key=lambda row: float(row["pc1"]) ** 2 + float(row["pc2"]) ** 2,
        reverse=True,
    )[:6]
    x_limits = axes[0, 0].get_xlim()
    y_limits = axes[0, 0].get_ylim()
    x_scale = 0.16 * (x_limits[1] - x_limits[0])
    y_scale = 0.16 * (y_limits[1] - y_limits[0])
    loading_key: list[str] = []
    for loading_index, loading in enumerate(loadings, start=1):
        end_x = float(loading["pc1"]) * x_scale
        end_y = float(loading["pc2"]) * y_scale
        axes[0, 0].annotate(
            "",
            xy=(end_x, end_y),
            xytext=(0.0, 0.0),
            arrowprops={"arrowstyle": "->", "color": "#334b4b", "lw": 0.8},
        )
        axes[0, 0].text(
            end_x,
            end_y,
            str(loading_index),
            fontsize=7,
            ha="center",
            va="center",
            fontweight="bold",
            color="#334b4b",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.8,
                "boxstyle": "square,pad=0.08",
            },
        )
        loading_key.append(f"{loading_index}  {loading['label']}")
    axes[0, 0].scatter(
        [0.0],
        [0.0],
        s=10,
        color="#334b4b",
        zorder=5,
    )
    axes[0, 0].text(
        0.985,
        0.985,
        "PCA loading vectors\narrow points toward higher values\n"
        + "\n".join(loading_key),
        transform=axes[0, 0].transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        linespacing=1.25,
        color="#334b4b",
        bbox={
            "facecolor": "white",
            "edgecolor": "#d4dcda",
            "linewidth": 0.6,
            "alpha": 0.9,
            "boxstyle": "square,pad=0.35",
        },
    )

    method = str(model_report.get("nonlinear_method", "nonlinear"))
    scatter_embedding(
        axes[0, 1],
        "nonlinear_1",
        "nonlinear_2",
        f"{method} 1",
        f"{method} 2",
        "B",
    )

    feature_fields = list(model_report.get("features", []))
    feature_labels = dict(model_report.get("feature_labels", {}))
    profile_matrix = np.asarray(
        [
            [float(row.get(f"z_{field}", 0.0)) for field in feature_fields]
            for row in profile_rows
        ],
        dtype=np.float64,
    )
    bound = max(1.0, float(np.quantile(np.abs(profile_matrix), 0.95)))
    image = axes[1, 0].imshow(
        profile_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-bound,
        vmax=bound,
    )
    axes[1, 0].set_title("C", loc="left")
    axes[1, 0].set_yticks(range(len(profile_rows)))
    axes[1, 0].set_yticklabels(
        [f"C{int(row['behavior_cluster'])}" for row in profile_rows]
    )
    axes[1, 0].set_xticks(range(len(feature_fields)))
    axes[1, 0].set_xticklabels(
        [feature_labels.get(field, field) for field in feature_fields],
        rotation=48,
        ha="right",
    )
    axes[1, 0].set_xlabel("Robust-scaled behavioral feature")
    axes[1, 0].set_ylabel("Behavior group")
    colorbar = figure.colorbar(image, ax=axes[1, 0], fraction=0.035, pad=0.02)
    colorbar.set_label("Cluster mean (robust-scaled units)")

    axis = axes[1, 1]
    axis.set_title("D", loc="left")
    selected_population = max(int(row["population_size"]) for row in fraction_rows)
    selected_fractions = [
        row
        for row in fraction_rows
        if int(row["population_size"]) == selected_population
    ]
    x_positions = np.arange(len(SHARED_CONDITIONS), dtype=np.float64)
    bottom = np.zeros(len(SHARED_CONDITIONS), dtype=np.float64)
    for cluster in clusters:
        values = []
        for condition in SHARED_CONDITIONS:
            samples = [
                float(row["fraction"])
                for row in selected_fractions
                if row["condition"] == condition
                and int(row["behavior_cluster"]) == cluster
            ]
            values.append(float(np.mean(samples)) if samples else 0.0)
        axis.bar(
            x_positions,
            values,
            bottom=bottom,
            width=0.68,
            color=PHENOTYPE_COLORS[(cluster - 1) % len(PHENOTYPE_COLORS)],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += np.asarray(values)
    axis.set_xticks(x_positions)
    axis.set_xticklabels([LABELS[condition] for condition in SHARED_CONDITIONS])
    axis.set_ylabel(f"Fraction of agents at N={selected_population}")
    axis.set_ylim(0, 1)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.14)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=PHENOTYPE_COLORS[(cluster - 1) % len(PHENOTYPE_COLORS)],
            markeredgewidth=0,
            label=f"C{cluster} · {descriptors[cluster]}",
        )
        for cluster in clusters
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=min(3, len(handles)),
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.075, 1, 1))
    return _save_figure(figure, destination, "movement-behavior-embedding")


def _render_behavior_scaling(
    fraction_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    destination: Path,
) -> dict[str, str]:
    """Render seed-level behavior-group prevalence across population sizes."""

    import matplotlib.pyplot as plt

    clusters = sorted({int(row["behavior_cluster"]) for row in fraction_rows})
    columns = min(3, len(clusters))
    rows = math.ceil(len(clusters) / columns)
    figure, axes_array = plt.subplots(
        rows,
        columns,
        figsize=(4.0 * columns, 3.15 * rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes = list(axes_array.ravel())
    descriptors = {
        int(row["behavior_cluster"]): str(row["behavior_descriptor"])
        for row in profile_rows
    }
    population_ticks = sorted(
        {int(row["population_size"]) for row in fraction_rows}
    )
    panel_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for axis_index, cluster in enumerate(clusters):
        axis = axes[axis_index]
        for condition in SHARED_CONDITIONS:
            populations = sorted(
                {
                    int(row["population_size"])
                    for row in fraction_rows
                    if row["condition"] == condition
                    and int(row["behavior_cluster"]) == cluster
                }
            )
            if not populations:
                continue
            means: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for population in populations:
                values = [
                    float(row["fraction"])
                    for row in fraction_rows
                    if row["condition"] == condition
                    and int(row["population_size"]) == population
                    and int(row["behavior_cluster"]) == cluster
                ]
                estimate = _mean_ci(
                    values,
                    key=("behavior-fraction", condition, population, cluster),
                    resamples=10_000,
                )
                means.append(float(estimate["mean"]))
                lows.append(float(estimate["ci95_low"]))
                highs.append(float(estimate["ci95_high"]))
            errors = np.asarray(
                [np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)]
            )
            axis.errorbar(
                populations,
                means,
                yerr=errors,
                color=COLORS[condition],
                marker=MARKERS[condition],
                markersize=4.5,
                linewidth=1.5,
                capsize=2.2,
                label=LABELS[condition],
            )
        axis.set_title(
            f"{panel_letters[axis_index]}  C{cluster} · {descriptors[cluster]}",
            loc="left",
            fontsize=9,
        )
        axis.set_xticks(population_ticks)
        axis.set_xlabel("Agents")
        axis.set_ylabel("Fraction of agents")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.16)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[len(clusters) :]:
        axis.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    return _save_figure(figure, destination, "movement-behavior-scaling")


def _render_behavior_condition_facets(
    embedded_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    model_report: dict[str, Any],
    destination: Path,
) -> dict[str, str]:
    """Facet the common agent-level UMAP by shared-world condition."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(1, 3, figsize=(12.6, 4.25), sharex=True, sharey=True)
    clusters = sorted({int(row["behavior_cluster"]) for row in embedded_rows})
    descriptors = {
        int(row["behavior_cluster"]): str(row["behavior_descriptor"])
        for row in profile_rows
    }
    x_values = np.asarray([float(row["nonlinear_1"]) for row in embedded_rows])
    y_values = np.asarray([float(row["nonlinear_2"]) for row in embedded_rows])
    x_padding = max(0.1, 0.04 * float(np.ptp(x_values)))
    y_padding = max(0.1, 0.04 * float(np.ptp(y_values)))
    panel_letters = "ABC"
    for axis_index, condition in enumerate(SHARED_CONDITIONS):
        axis = axes[axis_index]
        condition_rows = [
            row for row in embedded_rows if row["condition"] == condition
        ]
        for cluster in clusters:
            selected = [
                row
                for row in condition_rows
                if int(row["behavior_cluster"]) == cluster
            ]
            axis.scatter(
                [float(row["nonlinear_1"]) for row in selected],
                [float(row["nonlinear_2"]) for row in selected],
                s=7,
                alpha=0.33,
                linewidths=0,
                color=PHENOTYPE_COLORS[(cluster - 1) % len(PHENOTYPE_COLORS)],
                rasterized=True,
            )
            if selected:
                axis.text(
                    float(np.median([float(row["nonlinear_1"]) for row in selected])),
                    float(np.median([float(row["nonlinear_2"]) for row in selected])),
                    f"C{cluster}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": PHENOTYPE_COLORS[
                            (cluster - 1) % len(PHENOTYPE_COLORS)
                        ],
                        "linewidth": 0.7,
                        "boxstyle": "round,pad=0.18",
                    },
                )
        axis.set_title(f"{panel_letters[axis_index]}  {LABELS[condition]}", loc="left")
        axis.set_xlabel(f"{model_report['nonlinear_method']} 1")
        axis.set_xlim(float(np.min(x_values)) - x_padding, float(np.max(x_values)) + x_padding)
        axis.set_ylim(float(np.min(y_values)) - y_padding, float(np.max(y_values)) + y_padding)
        axis.grid(alpha=0.12)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(f"{model_report['nonlinear_method']} 2")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=PHENOTYPE_COLORS[(cluster - 1) % len(PHENOTYPE_COLORS)],
            markeredgewidth=0,
            label=f"C{cluster} · {descriptors[cluster]}",
        )
        for cluster in clusters
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=len(handles),
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    return _save_figure(figure, destination, "movement-behavior-condition-facets")


def _render_four_condition_regime_embedding(
    regime_rows: list[dict[str, Any]],
    model_report: dict[str, Any],
    destination: Path,
) -> dict[str, str]:
    """Render one common seed-level movement-regime PCA in four condition facets."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes_array = plt.subplots(2, 2, figsize=(9.8, 8.0), sharex=True, sharey=True)
    axes = list(axes_array.ravel())
    x_values = np.asarray([float(row["regime_pc1"]) for row in regime_rows])
    y_values = np.asarray([float(row["regime_pc2"]) for row in regime_rows])
    x_padding = max(0.1, 0.08 * float(np.ptp(x_values)))
    y_padding = max(0.1, 0.08 * float(np.ptp(y_values)))
    explained = model_report["pca_explained_variance_ratio"]
    panel_letters = "ABCD"
    populations = sorted({int(row["population_size"]) for row in regime_rows})
    population_colors = _population_color_map(populations)
    for axis_index, condition in enumerate(CONDITIONS):
        axis = axes[axis_index]
        means: list[tuple[float, float]] = []
        for population in populations:
            selected = [
                row
                for row in regime_rows
                if row["condition"] == condition
                and int(row["population_size"]) == population
            ]
            if not selected:
                continue
            x = [float(row["regime_pc1"]) for row in selected]
            y = [float(row["regime_pc2"]) for row in selected]
            axis.scatter(
                x,
                y,
                s=30,
                color=population_colors[population],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.9,
                zorder=3,
            )
            mean = (float(np.mean(x)), float(np.mean(y)))
            means.append(mean)
            axis.scatter(
                [mean[0]],
                [mean[1]],
                s=92,
                marker="D",
                color=population_colors[population],
                edgecolor="#283b3b",
                linewidth=0.8,
                zorder=4,
            )
        axis.plot(
            [point[0] for point in means],
            [point[1] for point in means],
            color="#526b75",
            linewidth=1.1,
            alpha=0.75,
            zorder=2,
        )
        if len(means) >= 2:
            axis.annotate(
                "",
                xy=means[-1],
                xytext=means[-2],
                arrowprops={"arrowstyle": "->", "color": "#526b75", "lw": 1.1},
            )
        axis.axhline(0, color="#aab6b3", linewidth=0.55, zorder=0)
        axis.axvline(0, color="#aab6b3", linewidth=0.55, zorder=0)
        axis.set_title(f"{panel_letters[axis_index]}  {LABELS[condition]}", loc="left")
        axis.set_xlim(float(np.min(x_values)) - x_padding, float(np.max(x_values)) + x_padding)
        axis.set_ylim(float(np.min(y_values)) - y_padding, float(np.max(y_values)) + y_padding)
        axis.grid(alpha=0.11)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel(f"PC1 ({100 * float(explained[0]):.1f}% variance)")
        axis.set_ylabel(f"PC2 ({100 * float(explained[1]):.1f}% variance)")

    loadings = sorted(
        model_report["pca_loadings"],
        key=lambda row: float(row["pc1"]) ** 2 + float(row["pc2"]) ** 2,
        reverse=True,
    )[:6]
    x_scale = 0.18 * (float(np.max(x_values)) - float(np.min(x_values)))
    y_scale = 0.18 * (float(np.max(y_values)) - float(np.min(y_values)))
    loading_key: list[str] = []
    for index, loading in enumerate(loadings, start=1):
        end_x = float(loading["pc1"]) * x_scale
        end_y = float(loading["pc2"]) * y_scale
        axes[0].annotate(
            str(index),
            xy=(end_x, end_y),
            xytext=(0, 0),
            fontsize=6.5,
            fontweight="bold",
            ha="center",
            va="center",
            arrowprops={"arrowstyle": "->", "color": "#334b4b", "lw": 0.7},
        )
        loading_key.append(f"{index}  {loading['label']}")
    axes[0].text(
        0.98,
        0.98,
        "\n".join(loading_key),
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color="#334b4b",
        bbox={
            "facecolor": "white",
            "edgecolor": "#d4dcda",
            "linewidth": 0.6,
            "alpha": 0.92,
            "boxstyle": "square,pad=0.3",
        },
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="",
            markersize=6,
            markerfacecolor=population_colors[population],
            markeredgecolor="#283b3b",
            label=f"N={population}",
        )
        for population in populations
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=max(1, len(populations)),
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.055, 1, 1))
    return _save_figure(figure, destination, "movement-four-condition-regime-embedding")


def _render_four_condition_signatures(
    profile_rows: list[dict[str, Any]],
    model_report: dict[str, Any],
    destination: Path,
) -> dict[str, str]:
    """Render common movement signatures separately for all four conditions."""

    import matplotlib.pyplot as plt

    # Preserve a complete, readable row key in every condition panel. The wider
    # canvas and explicit inter-panel gutter keep long feature labels from
    # intruding into the neighboring heatmap in paper and standalone exports.
    figure, axes_array = plt.subplots(2, 2, figsize=(11.6, 8.5))
    axes = list(axes_array.ravel())
    fields = list(model_report["features"])
    labels = dict(model_report["feature_labels"])
    all_values = np.asarray([float(row["mean_z"]) for row in profile_rows])
    bound = max(1.0, float(np.quantile(np.abs(all_values), 0.98)))
    panel_letters = "ABCD"
    populations = sorted({int(row["population_size"]) for row in profile_rows})
    population_index = {population: index for index, population in enumerate(populations)}
    image = None
    for axis_index, condition in enumerate(CONDITIONS):
        axis = axes[axis_index]
        matrix = np.full((len(fields), len(populations)), np.nan, dtype=np.float64)
        for row in profile_rows:
            if row["condition"] != condition:
                continue
            field_index = fields.index(str(row["feature"]))
            column_index = population_index[int(row["population_size"])]
            matrix[field_index, column_index] = float(row["mean_z"])
        image = axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-bound,
            vmax=bound,
        )
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if not math.isfinite(float(value)):
                    continue
                axis.text(
                    column_index,
                    row_index,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if abs(value) > 0.55 * bound else "#273837",
                )
        axis.set_title(f"{panel_letters[axis_index]}  {LABELS[condition]}", loc="left")
        axis.set_xticks(range(len(populations)))
        axis.set_xticklabels([f"N={population}" for population in populations])
        axis.set_yticks(range(len(fields)))
        axis.set_yticklabels([labels[field] for field in fields])
    if image is not None:
        colorbar_axis = figure.add_axes((0.28, 0.052, 0.44, 0.022))
        colorbar = figure.colorbar(
            image,
            cax=colorbar_axis,
            orientation="horizontal",
        )
        colorbar.set_label("Mean movement signature (robust-scaled units)")
    figure.subplots_adjust(
        left=0.16,
        right=0.985,
        top=0.95,
        bottom=0.15,
        wspace=0.58,
        hspace=0.28,
    )
    return _save_figure(figure, destination, "movement-four-condition-signatures")


def render_mobility_figures(
    episodes: list[dict[str, Any]],
    time_rows: list[dict[str, Any]],
    spatial_rows: list[dict[str, Any]],
    estimates: dict[str, Any],
    records: list[dict[str, Any]],
    detailed_traces: dict[tuple[str, int, int], dict[str, Any]],
    transition_rows: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    behavior_profiles: list[dict[str, Any]],
    behavior_fractions: list[dict[str, Any]],
    behavior_model: dict[str, Any],
    regime_rows: list[dict[str, Any]],
    regime_profiles: list[dict[str, Any]],
    regime_model: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Render publication PNG, PDF, and SVG mobility figures."""

    destination.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(destination / ".matplotlib"))
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "svg.fonttype": "none",
        }
    )
    figures = {
        "dynamics_overview": _render_dynamics_overview(
            time_rows, spatial_rows, destination
        ),
        "population_scaling": _render_scaling(episodes, estimates, destination),
        "self_organization": _render_self_organization(
            episodes, estimates, destination
        ),
    }
    if behavior_rows and behavior_profiles and behavior_fractions:
        figures["behavior_embedding"] = _render_behavior_embedding(
            behavior_rows,
            behavior_profiles,
            behavior_fractions,
            behavior_model,
            destination,
        )
        figures["behavior_scaling"] = _render_behavior_scaling(
            behavior_fractions,
            behavior_profiles,
            destination,
        )
        figures["behavior_condition_facets"] = _render_behavior_condition_facets(
            behavior_rows,
            behavior_profiles,
            behavior_model,
            destination,
        )
    regime_conditions = {str(row["condition"]) for row in regime_rows}
    if regime_rows and regime_profiles and set(CONDITIONS).issubset(regime_conditions):
        figures["four_condition_regime_embedding"] = (
            _render_four_condition_regime_embedding(
                regime_rows,
                regime_model,
                destination,
            )
        )
        figures["four_condition_signatures"] = _render_four_condition_signatures(
            regime_profiles,
            regime_model,
            destination,
        )
    candidates = [
        record
        for record in records
        if record.get("condition") in SHARED_CONDITIONS
        and int(record.get("population_size", 0)) == 200
        and int(record.get("seed", -1)) == 3202
        and record.get("_resolved_trace")
    ]
    representative: dict[str, dict[str, Any]] = {}
    episode_index = {
        (row["condition"], row["population_size"], row["seed"]): row
        for row in episodes
    }
    for record in candidates:
        condition = str(record["condition"])
        representative[condition] = {
            "trace": str(record["_resolved_trace"]),
            "episode": episode_index[(condition, 200, 3202)],
        }
    if set(representative) == set(SHARED_CONDITIONS):
        figures["trajectory_atlas"] = _render_trajectory_atlas(
            representative, destination
        )
        figures["occupancy_evolution"] = _render_occupancy_evolution(
            representative, destination
        )
    detailed_representative = {
        condition: {
            "trace": detailed_traces[(condition, 200, 3202)],
            "episode": episode_index[(condition, 200, 3202)],
        }
        for condition in SHARED_CONDITIONS
        if (condition, 200, 3202) in detailed_traces
    }
    if set(detailed_representative) == set(SHARED_CONDITIONS):
        figures["speed_interaction_atlas"] = _render_speed_interaction_atlas(
            detailed_representative, destination
        )
        figures["activity_raster"] = _render_activity_raster(
            detailed_representative, destination
        )
        figures["state_transitions"] = _render_state_transitions(
            transition_rows, destination
        )
        figures["interaction_scaling"] = _render_interaction_scaling(
            estimates, destination
        )
        figures["social_topology"] = _render_social_topology(
            detailed_representative["full"]["trace"], destination
        )
    return figures


def analyze_mobility(
    summary_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    resamples: int = 20_000,
    proximity_radius: float = 3.0,
    encounter_radius: float = 2.0,
) -> dict[str, Any]:
    """Analyze movement across matched studies and write all data and figures."""

    if resamples <= 0:
        raise ValueError("resamples must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    records = _load_records(summary_paths)
    episodes = [_episode_metrics(record) for record in records]
    time_rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("condition") in SHARED_CONDITIONS:
            time_rows.extend(_augment_time_points(record))

    spatial_rows: list[dict[str, Any]] = []
    agent_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    detailed_traces: dict[tuple[str, int, int], dict[str, Any]] = {}
    shared_records = [
        record
        for record in records
        if record.get("condition") in SHARED_CONDITIONS
        and record.get("_resolved_trace")
    ]
    episode_index = {
        (row["condition"], row["population_size"], row["seed"]): row
        for row in episodes
    }
    for record in tqdm(
        shared_records,
        desc="Spatial movement traces",
        unit="trace",
    ):
        key = (
            str(record["condition"]),
            int(record["population_size"]),
            int(record["seed"]),
        )
        retain_details = key[1:] == (200, 3202)
        trace = extract_spatial_trace(
            str(record["_resolved_trace"]),
            proximity_radius=proximity_radius,
            encounter_radius=encounter_radius,
            retain_positions=retain_details,
        )
        prefix = {
            "condition": str(record["condition"]),
            "population_size": int(record["population_size"]),
            "seed": int(record["seed"]),
        }
        rows = [{**prefix, **point} for point in trace["points"]]
        spatial_rows.extend(rows)
        agent_rows.extend({**prefix, **row} for row in trace["agent_rows"])
        probabilities = np.asarray(trace["transition_probabilities"], dtype=np.float64)
        counts = np.asarray(trace["transition_counts"], dtype=np.int64)
        for source_index, source_state in enumerate(MOBILITY_STATES):
            for target_index, target_state in enumerate(MOBILITY_STATES):
                transition_rows.append(
                    {
                        **prefix,
                        "source_state": source_state,
                        "target_state": target_state,
                        "count": int(counts[source_index, target_index]),
                        "probability": float(
                            probabilities[source_index, target_index]
                        ),
                    }
                )
        if retain_details:
            detailed_traces[key] = trace
        if not rows:
            continue
        episode = episode_index[
            (prefix["condition"], prefix["population_size"], prefix["seed"])
        ]
        final = rows[-1]
        episode.update(
            {
                "path_length_gini": final.get("path_length_gini"),
                "final_spatial_entropy": final.get("spatial_entropy"),
                "final_radius_of_gyration_normalized": final.get(
                    "radius_of_gyration_normalized"
                ),
                "final_nearest_neighbor_distance_normalized": final.get(
                    "nearest_neighbor_distance_normalized"
                ),
                "final_artifact_attraction": final.get("artifact_attraction"),
                "final_artifact_proximity_enrichment": final.get(
                    "artifact_proximity_enrichment"
                ),
                "final_resource_attraction": final.get("resource_attraction"),
                "mean_regional_switch_fraction": float(
                    np.mean([row["regional_switch_fraction"] for row in rows[1:]])
                )
                if len(rows) > 1
                else 0.0,
                "artifact_attraction_auc": _auc(rows, "artifact_attraction"),
                "resource_attraction_auc": _auc(rows, "resource_attraction"),
                **trace["summary"],
            }
        )

    estimates = _group_estimates(episodes, resamples=resamples)
    contrasts = _paired_contrasts(episodes, resamples=resamples)
    (
        behavior_rows,
        behavior_profiles,
        behavior_fractions,
        behavior_model,
    ) = _build_behavior_embedding(agent_rows)
    behavior_contrasts = _behavior_fraction_contrasts(
        behavior_fractions,
        resamples=resamples,
    )
    regime_rows, regime_profiles, regime_model = _build_regime_embedding(episodes)
    _write_csv(destination / "mobility-episodes.csv", episodes)
    _write_csv(destination / "mobility-time-series.csv", time_rows)
    _write_csv(destination / "mobility-spatial-time-series.csv", spatial_rows)
    _write_csv(destination / "mobility-agent-dynamics.csv", agent_rows)
    _write_csv(destination / "mobility-state-transitions.csv", transition_rows)
    _write_csv(destination / "mobility-paired-contrasts.csv", contrasts)
    _write_csv(destination / "mobility-behavior-embedding.csv", behavior_rows)
    _write_csv(
        destination / "mobility-behavior-cluster-profiles.csv",
        behavior_profiles,
    )
    _write_csv(
        destination / "mobility-behavior-episode-fractions.csv",
        behavior_fractions,
    )
    _write_csv(
        destination / "mobility-behavior-paired-contrasts.csv",
        behavior_contrasts,
    )
    (destination / "mobility-behavior-model.json").write_text(
        json.dumps(behavior_model, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(destination / "mobility-four-condition-regimes.csv", regime_rows)
    _write_csv(
        destination / "mobility-four-condition-signatures.csv",
        regime_profiles,
    )
    (destination / "mobility-four-condition-regime-model.json").write_text(
        json.dumps(regime_model, indent=2, sort_keys=True), encoding="utf-8"
    )
    figures = render_mobility_figures(
        episodes,
        time_rows,
        spatial_rows,
        estimates,
        records,
        detailed_traces,
        transition_rows,
        behavior_rows,
        behavior_profiles,
        behavior_fractions,
        behavior_model,
        regime_rows,
        regime_profiles,
        regime_model,
        destination,
    )
    report = {
        "version": MOBILITY_ANALYSIS_VERSION,
        "unit_of_analysis": "independent simulation seed",
        "agent_level_role": (
            "Agent trajectories construct seed-level observables and visualizations; "
            "agents are not treated as replicates."
        ),
        "independent_search_note": (
            "Independent endpoint movement is aggregated across isolated members. "
            "No shared-space trajectory, crowding, or artifact attraction is defined."
        ),
        "proximity_radius_cells": proximity_radius,
        "encounter_radius_cells": encounter_radius,
        "bootstrap_resamples": resamples,
        "sources": [str(Path(path).resolve()) for path in summary_paths],
        "episodes": episodes,
        "estimates": estimates,
        "paired_contrasts": contrasts,
        "behavior_embedding": behavior_model,
        "behavior_cluster_profiles": behavior_profiles,
        "behavior_paired_contrasts": behavior_contrasts,
        "four_condition_regime_embedding": regime_model,
        "figures": figures,
    }
    (destination / "mobility-analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report

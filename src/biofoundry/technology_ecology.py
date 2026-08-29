"""Causal measurements for cumulative, co-evolving artifact societies."""

from __future__ import annotations

import copy
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .artifacts import SERVICE_NAMES
from .dynamics import regional_crowding
from .events import read_records
from .program_library import program_fork_authorship_summary
from .simulation import BioFoundrySimulation
from .types import ActionType

INTERACTION_EPSILON = 1e-4


def _portfolio_state(simulation: BioFoundrySimulation) -> tuple[float, np.ndarray]:
    """Measure current balanced service coverage without using historical peaks."""

    count = simulation.artifacts.count
    active = np.nonzero(~simulation.artifacts.retired[:count])[0]
    if active.size == 0:
        return 0.0, np.zeros(len(SERVICE_NAMES), dtype=np.float64)
    coverage = np.max(
        simulation.artifacts.services[active].astype(np.float64), axis=0
    )
    mean = float(np.mean(coverage))
    balance = float(np.min(coverage) / mean) if mean > 1e-12 else 0.0
    return float(np.clip(mean * (0.5 + 0.5 * balance), 0.0, 1.0)), coverage


def _run_agent_free_assay(
    simulation: BioFoundrySimulation,
    horizon: int,
    *,
    description: str,
    show_progress: bool,
) -> dict[str, Any]:
    resilience: list[float] = []
    coverage_history: list[np.ndarray] = []
    artifact_performance: list[np.ndarray] = []
    environmental_history: list[list[float]] = []
    disturbances: list[dict[str, Any]] = []
    iterator: Iterable[int] = range(horizon)
    if show_progress:
        iterator = tqdm(iterator, desc=description, unit="tick", leave=False)
    for _ in iterator:
        result = simulation.step({})
        disturbances.extend(
            {"tick": event.tick, **event.payload}
            for event in result.events
            if event.kind == "environmental_disturbance"
        )
        value, coverage = _portfolio_state(simulation)
        resilience.append(value)
        coverage_history.append(coverage)
        current = simulation.artifacts.performance[: simulation.artifacts.count].astype(
            np.float64
        ).copy()
        current[simulation.artifacts.retired[: simulation.artifacts.count]] = 0.0
        artifact_performance.append(current)
        environmental_history.append(
            [
                float(np.mean(simulation.world.moisture)),
                float(np.mean(simulation.world.nutrients)),
                float(np.mean(simulation.world.contamination)),
                float(np.mean(simulation.world.temperature)),
            ]
        )

    coverage_array = np.asarray(coverage_history, dtype=np.float64)
    performance_array = np.asarray(artifact_performance, dtype=np.float64)
    environment_array = np.asarray(environmental_history, dtype=np.float64)
    return {
        "resilience_auc": float(np.mean(resilience)) if resilience else 0.0,
        "resilience_curve": resilience,
        "service_auc": (
            np.mean(coverage_array, axis=0)
            if coverage_array.size
            else np.zeros(len(SERVICE_NAMES), dtype=np.float64)
        ),
        "artifact_auc": (
            np.mean(performance_array, axis=0)
            if performance_array.size
            else np.zeros(simulation.artifacts.count, dtype=np.float64)
        ),
        "environment_curve": environment_array,
        "disturbances": disturbances,
    }


def technology_ecology_summary(simulation: BioFoundrySimulation) -> dict[str, Any]:
    """Describe inheritance, collaboration, diversification, and improvement."""

    artifacts: list[dict[str, Any]] = []
    collaborative = 0
    if simulation.research is not None:
        publications = simulation.research.publications
    else:
        publications = {}

    for index in range(simulation.artifacts.count):
        provenance = simulation.artifacts.provenance[index]
        lineage = (
            simulation.research.publication_lineage(
                set(provenance.get("causal_parents", []))
            )
            if simulation.research is not None
            else {"record_ids": [], "authors": [], "max_depth": 0}
        )
        program_authors = {
            str(entry.get("program", {}).get("author", ""))
            for entry in provenance.get("program_history", [])
            if entry.get("program", {}).get("author")
            and entry.get("program", {}).get("author") != "system"
        }
        participants = {
            str(provenance.get("creator", "")),
            *map(str, provenance.get("contributors", [])),
            *map(str, lineage.get("authors", [])),
            *program_authors,
        }
        participants.discard("")
        is_collaborative = len(participants) >= 2
        collaborative += int(is_collaborative)
        services = simulation.artifacts.lifetime_peak_services[index].astype(np.float64)
        dominant = SERVICE_NAMES[int(np.argmax(services))] if np.any(services) else "none"
        artifacts.append(
            {
                "artifact_id": f"artifact_{index:08d}",
                "name": simulation.artifacts.specs[index].get("name", "Untitled"),
                "creator": simulation.artifacts.creator[index],
                "created_tick": int(simulation.artifacts.created_tick[index]),
                "position": [
                    int(simulation.artifacts.x[index]),
                    int(simulation.artifacts.y[index]),
                ],
                "participants": sorted(participants),
                "collaborative": is_collaborative,
                "citation_depth": int(lineage.get("max_depth", 0)),
                "citation_records": list(lineage.get("record_ids", [])),
                "performance": round(
                    float(simulation.artifacts.lifetime_peak_performance[index]), 6
                ),
                "current_program_peak_performance": round(
                    float(simulation.artifacts.peak_performance[index]), 6
                ),
                "final_state_performance": round(
                    float(simulation.artifacts.performance[index]), 6
                ),
                "lifetime_peak_tick": int(
                    simulation.artifacts.lifetime_peak_tick[index]
                ),
                "lifetime_peak_program_id": (
                    simulation.artifacts.lifetime_peak_program_id[index] or None
                ),
                "dominant_service": dominant,
                "services": {
                    name: round(float(services[position]), 6)
                    for position, name in enumerate(SERVICE_NAMES)
                },
                "program_id": (
                    simulation.artifacts.programs[index].program_id
                    if simulation.artifacts.programs[index] is not None
                    else None
                ),
            }
        )

    fork_authorship = program_fork_authorship_summary(
        simulation.program_library.programs,
        simulation.program_library.lineage_edges,
        simulation.agent_ids,
    )

    fingerprints = simulation.artifacts.lifetime_peak_services[
        : simulation.artifacts.count
    ].astype(np.float64)
    dissimilarities: list[float] = []
    for first in range(len(fingerprints)):
        for second in range(first + 1, len(fingerprints)):
            denominator = float(
                np.linalg.norm(fingerprints[first]) * np.linalg.norm(fingerprints[second])
            )
            if denominator > 1e-12:
                similarity = float(
                    np.dot(fingerprints[first], fingerprints[second]) / denominator
                )
                dissimilarities.append(1.0 - similarity)

    improvements: list[dict[str, Any]] = []
    running_best = -1.0
    for artifact in sorted(
        artifacts,
        key=lambda item: (
            max(item["created_tick"], item["lifetime_peak_tick"]),
            item["artifact_id"],
        ),
    ):
        performance = float(artifact["performance"])
        if performance > running_best + 1e-9:
            improvements.append(
                {
                    "artifact_id": artifact["artifact_id"],
                    "tick": max(
                        artifact["created_tick"], artifact["lifetime_peak_tick"]
                    ),
                    "performance": performance,
                    "gain": round(performance - max(0.0, running_best), 6),
                    "collaborative": artifact["collaborative"],
                }
            )
            running_best = performance

    referenced_publications = {
        record_id for item in artifacts for record_id in item["citation_records"]
    }
    artifacts_by_id = {str(item["artifact_id"]): item for item in artifacts}
    closed_improvements = []
    for improvement in improvements:
        artifact = artifacts_by_id[str(improvement["artifact_id"])]
        closed = bool(
            artifact["collaborative"]
            and int(artifact["citation_depth"]) > 0
            and artifact["program_id"]
            and float(artifact["performance"]) > 0.0
        )
        improvement["causally_closed"] = closed
        if closed:
            closed_improvements.append(improvement)
    total_gain = sum(float(item["gain"]) for item in improvements)
    closed_gain = sum(float(item["gain"]) for item in closed_improvements)
    return {
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "collaborative_artifacts": collaborative,
        "collaborative_artifact_fraction": round(
            collaborative / max(1, len(artifacts)), 6
        ),
        "participating_agents": len(
            {participant for item in artifacts for participant in item["participants"]}
        ),
        "publication_reuse_fraction": round(
            len(referenced_publications) / max(1, len(publications)), 6
        ),
        "program_forks": len(simulation.program_library.lineage_edges),
        **fork_authorship,
        "mean_behavioral_dissimilarity": (
            round(float(np.mean(dissimilarities)), 6) if dissimilarities else 0.0
        ),
        "cumulative_improvements": improvements,
        "cumulative_improvement_count": len(improvements),
        "collaborative_improvement_count": sum(
            int(item["collaborative"]) for item in improvements
        ),
        "causally_closed_improvement_count": len(closed_improvements),
        "causal_closure_fraction": round(
            len(closed_improvements) / max(1, len(improvements)), 6
        ),
        "causal_closure_gain_fraction": round(
            closed_gain / max(1e-12, total_gain), 6
        ),
    }


def evaluate_technological_ecosystem(
    simulation: BioFoundrySimulation,
    *,
    horizon: int | None = None,
    show_progress: bool = True,
    detailed: bool = True,
    disturbance_seed: int | None = None,
    include_knockouts: bool = True,
) -> dict[str, Any]:
    """Freeze agents, run one native disturbance cycle, then knock out artifacts.

    Every trajectory starts from an exact deep copy of the final discovery state.
    The intact and knockout worlds therefore receive identical deterministic future
    weather. Differences are caused only by the removed artifact and its ecological
    effects on the remaining technological ecosystem.
    """

    assay_horizon = int(horizon or simulation.config.evaluation.horizon)
    assay_source = copy.deepcopy(simulation)
    schedule = None
    if disturbance_seed is not None:
        schedule = assay_source.world.configure_evaluation_disturbances(
            disturbance_seed
        )
    intact = _run_agent_free_assay(
        copy.deepcopy(assay_source),
        assay_horizon,
        description="Intact technological ecosystem",
        show_progress=show_progress,
    )
    artifact_count = simulation.artifacts.count
    knockout_results: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    sources: Iterable[int] = range(artifact_count) if include_knockouts else ()
    if show_progress and artifact_count and include_knockouts:
        sources = tqdm(
            sources,
            total=artifact_count,
            desc="Artifact knockout assays",
            unit="artifact",
        )
    intact_artifact_auc = np.asarray(intact["artifact_auc"], dtype=np.float64)
    for source in sources:
        knockout = copy.deepcopy(assay_source)
        knockout.artifacts.retired[source] = True
        knockout.artifacts.health[source] = np.float32(0.0)
        knockout.artifacts.performance[source] = np.float32(0.0)
        knockout.artifacts.services[source].fill(np.float32(0.0))
        knockout.artifacts.programs[source] = None
        result = _run_agent_free_assay(
            knockout,
            assay_horizon,
            description=f"Knock out artifact {source}",
            show_progress=False,
        )
        target_auc = np.asarray(result["artifact_auc"], dtype=np.float64)
        system_effect = float(intact["resilience_auc"] - result["resilience_auc"])
        knockout_results.append(
            {
                "artifact_id": f"artifact_{source:08d}",
                "resilience_auc_without": round(float(result["resilience_auc"]), 6),
                "system_effect": round(system_effect, 6),
            }
        )
        for target in range(artifact_count):
            if target == source:
                continue
            effect = float(intact_artifact_auc[target] - target_auc[target])
            if abs(effect) < 1e-9:
                continue
            interactions.append(
                {
                    "source": f"artifact_{source:08d}",
                    "target": f"artifact_{target:08d}",
                    "effect": round(effect, 6),
                    "kind": "support" if effect > 0 else "inhibition",
                }
            )

    effect_lookup = {
        (item["source"], item["target"]): float(item["effect"])
        for item in interactions
    }
    reciprocal_pairs: list[dict[str, Any]] = []
    for first in range(artifact_count):
        for second in range(first + 1, artifact_count):
            first_id = f"artifact_{first:08d}"
            second_id = f"artifact_{second:08d}"
            forward = effect_lookup.get((first_id, second_id), 0.0)
            reverse = effect_lookup.get((second_id, first_id), 0.0)
            if forward > INTERACTION_EPSILON and reverse > INTERACTION_EPSILON:
                reciprocal_pairs.append(
                    {
                        "artifacts": [first_id, second_id],
                        "effects": [round(forward, 6), round(reverse, 6)],
                        "strength": round(min(forward, reverse), 6),
                    }
                )

    off_diagonal = [abs(float(item["effect"])) for item in interactions]
    intact_public = {
        "resilience_auc": round(float(intact["resilience_auc"]), 6),
        "resilience_curve": [round(float(value), 6) for value in intact["resilience_curve"]],
        "service_auc": {
            name: round(float(intact["service_auc"][index]), 6)
            for index, name in enumerate(SERVICE_NAMES)
        },
        "artifact_auc": {
            f"artifact_{index:08d}": round(float(value), 6)
            for index, value in enumerate(intact_artifact_auc)
        },
        "environment_curve": [
            [round(float(value), 6) for value in row]
            for row in np.asarray(intact["environment_curve"])
        ],
        "disturbances": intact["disturbances"],
    }
    knockout_results.sort(key=lambda item: -abs(float(item["system_effect"])))
    interactions.sort(key=lambda item: -abs(float(item["effect"])))
    reciprocal_pairs.sort(key=lambda item: -float(item["strength"]))
    report = {
        "suite": simulation.config.evaluation.suite,
        "disturbance_schedule": schedule,
        "agent_actions_during_assay": 0,
        "horizon": assay_horizon,
        "interaction_epsilon": INTERACTION_EPSILON,
        "intact": intact_public,
        "knockouts": knockout_results,
        "interactions": interactions,
        "mean_absolute_ecological_coupling": round(
            float(np.sum(off_diagonal)) / max(1, artifact_count * (artifact_count - 1)),
            6,
        ),
        "supported_interaction_count": sum(
            float(item["effect"]) > INTERACTION_EPSILON for item in interactions
        ),
        "inhibitory_interaction_count": sum(
            float(item["effect"]) < -INTERACTION_EPSILON for item in interactions
        ),
        "reciprocal_support_pairs": reciprocal_pairs,
        "reciprocal_support_pair_count": len(reciprocal_pairs),
        "maximum_knockout_effect": round(
            max(
                (float(item["system_effect"]) for item in knockout_results),
                default=0.0,
            ),
            6,
        ),
    }
    if not detailed:
        report["intact"] = {
            "resilience_auc": intact_public["resilience_auc"],
            "service_auc": intact_public["service_auc"],
            "disturbance_count": len(intact_public["disturbances"]),
        }
        report["knockout_count"] = len(knockout_results)
        report.pop("knockouts", None)
        report.pop("interactions", None)
        report.pop("reciprocal_support_pairs", None)
    return report


def evaluate_technological_ecosystem_generalization(
    simulation: BioFoundrySimulation,
    seeds: Iterable[int],
    *,
    horizon: int | None = None,
    show_progress: bool = True,
    bootstrap_resamples: int = 10_000,
) -> dict[str, Any]:
    """Evaluate one frozen society under independent held-out stress schedules."""

    evaluator_seeds = [int(seed) for seed in seeds]
    if not evaluator_seeds:
        raise ValueError("at least one held-out evaluation seed is required")
    iterator: Iterable[int] = evaluator_seeds
    if show_progress:
        iterator = tqdm(
            evaluator_seeds,
            desc="Held-out disturbance schedules",
            unit="schedule",
        )
    episodes = [
        evaluate_technological_ecosystem(
            simulation,
            horizon=horizon,
            show_progress=False,
            detailed=False,
            disturbance_seed=seed,
            include_knockouts=False,
        )
        for seed in iterator
    ]
    values = np.asarray(
        [float(item["intact"]["resilience_auc"]) for item in episodes],
        dtype=np.float64,
    )
    if len(values) == 1:
        low = high = float(values[0])
    else:
        rng = np.random.default_rng(20260812)
        indices = rng.integers(
            0, len(values), size=(int(bootstrap_resamples), len(values))
        )
        low, high = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
    return {
        "evaluation_unit": "held-out disturbance schedule",
        "discovery_state_frozen": True,
        "agent_actions_per_schedule": 0,
        "seeds": evaluator_seeds,
        "n": len(episodes),
        "resilience_auc": {
            "mean": round(float(np.mean(values)), 6),
            "ci95_low": round(float(low), 6),
            "ci95_high": round(float(high), 6),
            "standard_deviation": round(
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 6
            ),
            "worst_case": round(float(np.min(values)), 6),
        },
        "schedules": [
            {
                "seed": seed,
                "disturbance_schedule": episode["disturbance_schedule"],
                "resilience_auc": episode["intact"]["resilience_auc"],
                "service_auc": episode["intact"]["service_auc"],
                "disturbance_count": episode["intact"]["disturbance_count"],
            }
            for seed, episode in zip(evaluator_seeds, episodes, strict=True)
        ],
    }


def snapshot_dynamics(path: str | Path) -> list[dict[str, Any]]:
    """Extract compact spatial and technological history from recorded snapshots."""

    points: list[dict[str, Any]] = []
    for record in read_records(path):
        if record.get("type") != "snapshot":
            continue
        snapshot = dict(record.get("snapshot", {}))
        agents = dict(snapshot.get("agents", {}))
        count = int(agents.get("display_count", agents.get("count", 0)))
        x = np.asarray(agents.get("x", [])[:count], dtype=np.int64)
        y = np.asarray(agents.get("y", [])[:count], dtype=np.int64)
        if count <= 1 or x.size != count or y.size != count:
            concentration = 1.0 if count == 1 else 0.0
        else:
            width = int(dict(snapshot.get("world", {})).get("width", 1))
            _, occupancy = np.unique(y * width + x, return_counts=True)
            shares = occupancy.astype(np.float64) / count
            hhi = float(np.sum(shares * shares))
            concentration = float(
                np.clip((hhi - 1.0 / count) / (1.0 - 1.0 / count), 0.0, 1.0)
            )
        distance = np.asarray(agents.get("distance_traveled", []), dtype=np.float64)
        visited = np.asarray(agents.get("distinct_cells_visited", []), dtype=np.float64)
        artifact_state = dict(snapshot.get("artifacts", {}))
        performance = np.asarray(
            artifact_state.get(
                "lifetime_peak_performance",
                artifact_state.get("peak_performance", []),
            ),
            dtype=np.float64,
        )
        last_actions = np.asarray(agents.get("last_action", []), dtype=np.int16)
        points.append(
            {
                "tick": int(snapshot.get("tick", 0)),
                "artifact_count": int(artifact_state.get("count", 0)),
                "best_artifact_performance": float(
                    np.max(performance) if performance.size else 0.0
                ),
                "portfolio_resilience": float(
                    dict(snapshot.get("research", {})).get(
                        "portfolio_resilience", 0.0
                    )
                ),
                "spatial_concentration": concentration,
                "regional_crowding": regional_crowding(
                    x,
                    y,
                    width=int(dict(snapshot.get("world", {})).get("width", 1)),
                    height=int(dict(snapshot.get("world", {})).get("height", 1)),
                ),
                "mean_distance_traveled": float(
                    np.mean(distance) if distance.size else 0.0
                ),
                "mean_distinct_cells_visited": float(
                    np.mean(visited) if visited.size else 0.0
                ),
                "moving_agent_fraction": float(
                    np.mean(last_actions == int(ActionType.MOVE))
                    if last_actions.size
                    else 0.0
                ),
                "_distance_by_agent": distance.tolist(),
            }
        )
    deduplicated = {int(point["tick"]): point for point in points}
    ordered = [deduplicated[tick] for tick in sorted(deduplicated)]
    previous_distance: np.ndarray | None = None
    for point in ordered:
        current_distance = np.asarray(point.pop("_distance_by_agent"), dtype=np.float64)
        point["relocating_agent_fraction"] = float(
            np.mean(current_distance > previous_distance)
            if previous_distance is not None
            and current_distance.size
            and current_distance.shape == previous_distance.shape
            else 0.0
        )
        previous_distance = current_distance
    return ordered


def render_technology_ecology_figures(
    simulation: BioFoundrySimulation,
    assay: dict[str, Any],
    dynamics: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    stem: str = "technology-ecology",
) -> dict[str, str]:
    """Render succession, inheritance, mobility, and knockout diagnostics."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(destination / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "technology-ecology figures require `python -m pip install -e '.[analysis]'`"
        ) from exc

    ecology = technology_ecology_summary(simulation)
    artifacts = list(ecology["artifacts"])
    colors = {
        "ink": "#16302b",
        "green": "#16866f",
        "gold": "#d39b32",
        "coral": "#d8664b",
        "blue": "#397aa8",
        "purple": "#8a5ea7",
        "muted": "#8aa29b",
        "grid": "#d9e4e0",
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelcolor": colors["ink"],
            "text.color": colors["ink"],
            "xtick.color": colors["ink"],
            "ytick.color": colors["ink"],
        }
    )

    overview, axes = plt.subplots(2, 2, figsize=(13.8, 8.4), constrained_layout=True)
    creation_ticks = np.asarray(
        [int(item["created_tick"]) for item in artifacts], dtype=np.float64
    )
    performances = np.asarray(
        [float(item["performance"]) for item in artifacts], dtype=np.float64
    )
    collaborative = np.asarray(
        [bool(item["collaborative"]) for item in artifacts], dtype=np.bool_
    )
    if artifacts:
        axes[0, 0].scatter(
            creation_ticks[~collaborative],
            performances[~collaborative],
            s=34,
            color=colors["muted"],
            alpha=0.75,
            label="single-agent lineage",
        )
        axes[0, 0].scatter(
            creation_ticks[collaborative],
            performances[collaborative],
            s=52,
            marker="D",
            color=colors["gold"],
            edgecolor=colors["ink"],
            linewidth=0.4,
            label="multi-agent lineage",
        )
        order = np.argsort(creation_ticks, kind="stable")
        axes[0, 0].step(
            creation_ticks[order],
            np.maximum.accumulate(performances[order]),
            where="post",
            color=colors["green"],
            linewidth=1.8,
            label="cumulative best",
        )
    axes[0, 0].set_title("A  Technological succession", loc="left")
    axes[0, 0].set_xlabel("Discovery tick")
    axes[0, 0].set_ylabel("Peak field performance")
    if artifacts:
        axes[0, 0].legend(frameon=False, fontsize=8)

    ticks = np.asarray([point["tick"] for point in dynamics], dtype=np.float64)
    if dynamics:
        axes[0, 1].plot(
            ticks,
            [point.get("regional_crowding", 0.0) for point in dynamics],
            color=colors["coral"],
            linewidth=1.8,
            label="regional crowding",
        )
        axes[0, 1].plot(
            ticks,
            [point.get("relocating_agent_fraction", 0.0) for point in dynamics],
            color=colors["purple"],
            linewidth=1.4,
            linestyle="--",
            label="agents relocating / interval",
        )
        distance = np.asarray(
            [point["mean_distance_traveled"] for point in dynamics], dtype=np.float64
        )
        maximum = max(1.0, float(np.max(distance)))
        axes[0, 1].plot(
            ticks,
            distance / maximum,
            color=colors["blue"],
            linewidth=1.8,
            label=f"mean travel / {maximum:.0f} steps",
        )
    axes[0, 1].set_title("B  Society mobility", loc="left")
    axes[0, 1].set_xlabel("Simulation tick")
    axes[0, 1].set_ylabel("Normalized population measure")
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].legend(frameon=False, fontsize=8)

    curve = np.asarray(assay["intact"]["resilience_curve"], dtype=np.float64)
    assay_ticks = simulation.tick + np.arange(1, len(curve) + 1)
    axes[1, 0].plot(
        assay_ticks,
        curve,
        color=colors["green"],
        linewidth=2.0,
        label="intact ecosystem",
    )
    for disturbance in assay["intact"].get("disturbances", []):
        tick = int(disturbance["tick"])
        axes[1, 0].axvline(tick, color=colors["coral"], alpha=0.38, linewidth=1.0)
        axes[1, 0].text(
            tick,
            max(0.005, float(np.max(curve, initial=0.0)) * 0.96),
            str(disturbance.get("kind", "stress")).replace("_", " "),
            rotation=90,
            va="top",
            ha="right",
            fontsize=7,
            color=colors["coral"],
        )
    axes[1, 0].set_title("C  Agent-free resilience assay", loc="left")
    axes[1, 0].set_xlabel("World tick after agents are frozen")
    axes[1, 0].set_ylabel("Balanced service coverage")

    niche_counts = {
        name: sum(item["dominant_service"] == name for item in artifacts)
        for name in SERVICE_NAMES
    }
    labels = [name.replace("_", "\n") for name in SERVICE_NAMES]
    axes[1, 1].bar(
        np.arange(len(labels)),
        [niche_counts[name] for name in SERVICE_NAMES],
        color=[
            colors["green"],
            colors["coral"],
            colors["blue"],
            colors["gold"],
            colors["muted"],
            colors["ink"],
        ],
        alpha=0.88,
    )
    axes[1, 1].set_title("D  Emergent technological niches", loc="left")
    axes[1, 1].set_ylabel("Artifacts dominated by service")
    axes[1, 1].set_xticks(np.arange(len(labels)), labels, fontsize=8)
    for axis in axes.ravel():
        axis.grid(axis="y", color=colors["grid"], linewidth=0.7, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    overview.suptitle(
        "Decentralized technological ecosystem: succession, movement, and resilience",
        fontsize=15,
        fontweight="bold",
    )
    overview_png = destination / f"{stem}-overview.png"
    overview_pdf = destination / f"{stem}-overview.pdf"
    overview.savefig(overview_png, dpi=240, facecolor="white")
    overview.savefig(overview_pdf, facecolor="white")
    plt.close(overview)

    lineage, (tree_axis, adoption_axis) = plt.subplots(
        1, 2, figsize=(14.2, 7.2), gridspec_kw={"width_ratios": [1.65, 1.0]},
        constrained_layout=True,
    )
    programs = simulation.program_library.programs
    edges = simulation.program_library.lineage_edges
    agent_ids = set(simulation.agent_ids)
    ordered_programs = sorted(
        programs,
        key=lambda program_id: (
            int(programs[program_id].get("first_tick", 0)), program_id
        ),
    )
    lane = {program_id: index for index, program_id in enumerate(ordered_programs)}
    for edge in edges:
        parent = str(edge["parent_program_id"])
        child = str(edge["child_program_id"])
        if parent not in lane or child not in lane:
            continue
        parent_record = programs[parent]
        child_record = programs[child]
        parent_authors = {
            str(author)
            for author in parent_record.get("authors", [])
            if str(author) in agent_ids
        }
        child_author = str(edge.get("author", ""))
        eligible = bool(parent_authors) and child_author in agent_ids
        cross_agent = eligible and child_author not in parent_authors
        tree_axis.plot(
            [parent_record.get("first_tick", 0), child_record.get("first_tick", 0)],
            [lane[parent], lane[child]],
            color=colors["gold"] if cross_agent else colors["muted"],
            linewidth=1.45 if cross_agent else 0.75,
            alpha=0.85 if cross_agent else 0.48,
            zorder=1,
        )
    for program_id in ordered_programs:
        record = programs[program_id]
        installs = len(record.get("installations", []))
        tree_axis.scatter(
            int(record.get("first_tick", 0)),
            lane[program_id],
            s=18 + 10 * np.sqrt(max(1, installs)),
            color=colors["green"],
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
    tree_axis.set_title("A  Executable program phylogeny", loc="left")
    tree_axis.set_xlabel("First appearance tick")
    tree_axis.set_ylabel("Distinct executable program")
    tree_axis.set_yticks([])
    tree_axis.legend(
        handles=[
            Line2D([0], [0], color=colors["muted"], label="same-agent descent"),
            Line2D([0], [0], color=colors["gold"], linewidth=2, label="cross-agent descent"),
        ],
        frameon=False,
        fontsize=8,
    )
    ranked_programs = sorted(
        programs.values(),
        key=lambda record: -len(record.get("installations", [])),
    )[:15]
    adoption_axis.barh(
        np.arange(len(ranked_programs)),
        [len(record.get("installations", [])) for record in ranked_programs],
        color=colors["blue"],
        alpha=0.85,
    )
    adoption_axis.set_yticks(
        np.arange(len(ranked_programs)),
        [str(record.get("name", "program"))[:32] for record in ranked_programs],
        fontsize=8,
    )
    adoption_axis.invert_yaxis()
    adoption_axis.set_xlabel("Artifact installations")
    adoption_axis.set_title("B  Differential cultural adoption", loc="left")
    for axis in (tree_axis, adoption_axis):
        axis.grid(axis="x", color=colors["grid"], linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    lineage.suptitle(
        "Technological inheritance is executable and attributable",
        fontsize=15,
        fontweight="bold",
    )
    lineage_png = destination / f"{stem}-lineage.png"
    lineage_pdf = destination / f"{stem}-lineage.pdf"
    lineage.savefig(lineage_png, dpi=240, facecolor="white")
    lineage.savefig(lineage_pdf, facecolor="white")
    plt.close(lineage)

    count = simulation.artifacts.count
    matrix = np.zeros((count, count), dtype=np.float64)
    for interaction in assay.get("interactions", []):
        source = int(str(interaction["source"]).removeprefix("artifact_"))
        target = int(str(interaction["target"]).removeprefix("artifact_"))
        if source < count and target < count:
            matrix[target, source] = float(interaction["effect"])
    knockout_effect = {
        str(item["artifact_id"]): float(item["system_effect"])
        for item in assay.get("knockouts", [])
    }
    order = sorted(
        range(count),
        key=lambda index: -abs(
            knockout_effect.get(f"artifact_{index:08d}", 0.0)
        ),
    )
    display = order[: min(40, count)]
    interaction_figure, (matrix_axis, knockout_axis) = plt.subplots(
        1, 2, figsize=(13.8, 7.4), gridspec_kw={"width_ratios": [1.65, 1.0]},
        constrained_layout=True,
    )
    if display:
        visible = matrix[np.ix_(display, display)]
        scale = max(float(np.max(np.abs(visible), initial=0.0)), 1e-6)
        image = matrix_axis.imshow(
            visible,
            cmap="PiYG",
            vmin=-scale,
            vmax=scale,
            interpolation="nearest",
            aspect="auto",
        )
        interaction_figure.colorbar(
            image, ax=matrix_axis, label="Target AUC lost when source is removed"
        )
        labels = [f"A{index}" for index in display]
        matrix_axis.set_xticks(np.arange(len(display)), labels, rotation=90, fontsize=6)
        matrix_axis.set_yticks(np.arange(len(display)), labels, fontsize=6)
    matrix_axis.set_title("A  Ecological technology interactions", loc="left")
    matrix_axis.set_xlabel("Knocked-out source technology")
    matrix_axis.set_ylabel("Responding target technology")

    top_knockouts = sorted(
        assay.get("knockouts", []),
        key=lambda item: -abs(float(item["system_effect"])),
    )[:20]
    effects = [float(item["system_effect"]) for item in top_knockouts]
    knockout_axis.barh(
        np.arange(len(top_knockouts)),
        effects,
        color=[colors["green"] if value >= 0 else colors["coral"] for value in effects],
        alpha=0.88,
    )
    knockout_axis.axvline(0.0, color=colors["ink"], linewidth=0.8)
    knockout_axis.set_yticks(
        np.arange(len(top_knockouts)),
        [str(item["artifact_id"]).replace("artifact_000000", "A") for item in top_knockouts],
        fontsize=8,
    )
    knockout_axis.invert_yaxis()
    knockout_axis.set_xlabel("Intact − knockout resilience AUC")
    knockout_axis.set_title("B  Causal importance to the habitat", loc="left")
    knockout_axis.grid(axis="x", color=colors["grid"], linewidth=0.7)
    knockout_axis.spines[["top", "right"]].set_visible(False)
    interaction_figure.suptitle(
        "Artifact-removal experiments reveal the technological ecology",
        fontsize=15,
        fontweight="bold",
    )
    interactions_png = destination / f"{stem}-interactions.png"
    interactions_pdf = destination / f"{stem}-interactions.pdf"
    interaction_figure.savefig(interactions_png, dpi=240, facecolor="white")
    interaction_figure.savefig(interactions_pdf, facecolor="white")
    plt.close(interaction_figure)

    from matplotlib.colors import ListedColormap

    map_figure, map_axis = plt.subplots(figsize=(12.8, 8.2), constrained_layout=True)
    terrain_cmap = ListedColormap(
        [
            "#153b52",
            "#397f87",
            "#7ea46e",
            "#476f52",
            "#988264",
            "#b6a862",
            "#91a9aa",
            "#756d78",
            "#8e7657",
        ]
    )
    map_axis.imshow(
        simulation.world.terrain,
        cmap=terrain_cmap,
        vmin=0,
        vmax=8,
        interpolation="nearest",
        alpha=0.88,
        origin="upper",
    )
    station_y, station_x = np.nonzero(simulation.world.stations)
    map_axis.scatter(
        station_x,
        station_y,
        marker="s",
        s=70,
        facecolor="white",
        edgecolor=colors["ink"],
        linewidth=1.0,
        label="distributed laboratory",
        zorder=4,
    )
    map_axis.scatter(
        simulation.population.x,
        simulation.population.y,
        marker=".",
        s=16,
        color=colors["ink"],
        alpha=0.45,
        label="agent at freeze",
        zorder=3,
    )
    niche_color = dict(
        zip(
            SERVICE_NAMES,
            [
                colors["blue"],
                colors["coral"],
                colors["ink"],
                colors["gold"],
                colors["muted"],
                colors["green"],
            ],
            strict=True,
        )
    )
    positions = {
        item["artifact_id"]: tuple(map(float, item["position"])) for item in artifacts
    }
    for interaction in assay.get("interactions", [])[:80]:
        source = positions.get(str(interaction["source"]))
        target = positions.get(str(interaction["target"]))
        effect = float(interaction["effect"])
        if source is None or target is None or abs(effect) < INTERACTION_EPSILON:
            continue
        map_axis.annotate(
            "",
            xy=target,
            xytext=source,
            arrowprops={
                "arrowstyle": "->",
                "color": colors["green"] if effect > 0 else colors["coral"],
                "alpha": 0.36,
                "linewidth": 0.7 + min(2.0, abs(effect) * 40.0),
            },
            zorder=4,
        )
    for item in artifacts:
        x, y = item["position"]
        effect = abs(knockout_effect.get(str(item["artifact_id"]), 0.0))
        map_axis.scatter(
            x,
            y,
            marker="D" if item["collaborative"] else "o",
            s=42 + 1800 * effect,
            color=niche_color.get(str(item["dominant_service"]), colors["muted"]),
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
    for item in sorted(
        artifacts,
        key=lambda value: -abs(
            knockout_effect.get(str(value["artifact_id"]), 0.0)
        ),
    )[:6]:
        x, y = item["position"]
        map_axis.text(
            x + 0.7,
            y - 0.7,
            str(item["name"])[:28],
            fontsize=7,
            color=colors["ink"],
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
                "pad": 1.2,
            },
            zorder=6,
        )
    map_axis.set_title(
        "Spatial technological ecology at the moment agents are frozen",
        loc="left",
        fontsize=15,
        fontweight="bold",
    )
    map_axis.set_xlabel("World x")
    map_axis.set_ylabel("World y")
    map_axis.legend(frameon=False, loc="upper right", fontsize=8)
    map_png = destination / f"{stem}-map.png"
    map_pdf = destination / f"{stem}-map.pdf"
    map_figure.savefig(map_png, dpi=240, facecolor="white")
    map_figure.savefig(map_pdf, facecolor="white")
    plt.close(map_figure)

    return {
        "overview_png": str(overview_png.resolve()),
        "overview_pdf": str(overview_pdf.resolve()),
        "lineage_png": str(lineage_png.resolve()),
        "lineage_pdf": str(lineage_pdf.resolve()),
        "interactions_png": str(interactions_png.resolve()),
        "interactions_pdf": str(interactions_pdf.resolve()),
        "map_png": str(map_png.resolve()),
        "map_pdf": str(map_pdf.resolve()),
    }

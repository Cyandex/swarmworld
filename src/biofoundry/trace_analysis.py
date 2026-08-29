"""Mechanistic diagnostics for a recorded BioFoundry JSONL episode."""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .events import read_records
from .scenario_analysis import analyze_scenario_trace
from .types import ActionType


def _quantiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "median": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(np.mean(array)), 6),
        "median": round(float(np.median(array)), 6),
        "p90": round(float(np.quantile(array, 0.9)), 6),
        "max": int(np.max(array)),
    }


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9]+", left.lower()))
    right_tokens = set(re.findall(r"[a-z0-9]+", right.lower()))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def analyze_trace(path: str | Path) -> dict[str, Any]:
    """Summarize behavior, failures, exploration, and causal information flow."""

    source = Path(path)
    record_types: Counter[str] = Counter()
    events: Counter[str] = Counter()
    first_event_tick: dict[str, int] = {}
    executed_actions: Counter[str] = Counter()
    planned_actions: Counter[str] = Counter()
    per_agent_actions: dict[str, Counter[str]] = defaultdict(Counter)
    rejection_reasons: Counter[str] = Counter()
    successful_outcomes: Counter[str] = Counter()
    model_errors: Counter[str] = Counter()
    usage: Counter[str] = Counter()
    plan_lengths: list[int] = []
    queued_replacements: list[int] = []
    rolled_back_plans = 0
    inspected_sites: dict[str, list[tuple[int, int]]] = defaultdict(list)
    publications: dict[str, dict[str, Any]] = {}
    microbatches: list[dict[str, Any]] = []
    material_tests: list[dict[str, Any]] = []
    task_claims: list[dict[str, Any]] = []
    artifact_builds: list[dict[str, Any]] = []
    program_targets: Counter[str] = Counter()
    research_states: dict[str, list[dict[str, Any]]] = defaultdict(list)
    messages: list[dict[str, Any]] = []
    fulfillment_events: list[dict[str, Any]] = []
    legacy_fulfillment_events: list[dict[str, Any]] = []
    retrieval_selected = 0
    retrieval_used = 0
    retrieval_selected_ids: dict[str, set[str]] = defaultdict(set)
    retrieval_outcome_attempts = 0
    retrieval_successful_outcomes = 0
    final_snapshot: dict[str, Any] | None = None
    header: dict[str, Any] = {}
    last_tick = 0

    for record in read_records(source):
        record_type = str(record.get("type", "unknown"))
        record_types[record_type] += 1
        tick = int(record.get("tick", 0))
        last_tick = max(last_tick, tick)
        if record_type == "header":
            header = dict(record.get("metadata", {}))
        elif record_type == "actions":
            for agent, action in dict(record.get("actions", {})).items():
                try:
                    name = ActionType(int(action.get("verb", 0))).name
                except (TypeError, ValueError):
                    name = "INVALID"
                executed_actions[name] += 1
                per_agent_actions[str(agent)][name] += 1
        elif record_type == "model_trace":
            actions = list(record.get("actions", []))
            planning_committed = record.get("planning_committed") is not False
            if planning_committed:
                plan_lengths.append(len(actions))
                queued_replacements.append(
                    int(record.get("queued_actions_replaced", 0))
                )
                for action in actions:
                    try:
                        name = ActionType(int(action.get("verb", 0))).name
                    except (TypeError, ValueError):
                        name = "INVALID"
                    planned_actions[name] += 1
            else:
                rolled_back_plans += 1
            error = str(record.get("error", "")).strip()
            if error:
                category = error.split(":", 1)[0][:120]
                model_errors[category] += 1
            for key, value in dict(record.get("usage", {})).items():
                if isinstance(value, int):
                    usage[str(key)] += value
            state = record.get("research_state")
            if planning_committed and isinstance(state, dict):
                research_states[str(record.get("agent", "unknown"))].append(
                    {"tick": tick, **state}
                )
            retrieval = record.get("retrieval")
            if planning_committed and isinstance(retrieval, dict):
                selected_items = list(retrieval.get("selected", []))
                retrieval_selected += len(selected_items)
                retrieval_used += len(retrieval.get("used_as_causal_parent", []))
                agent = str(record.get("agent", "unknown"))
                retrieval_selected_ids[agent].update(
                    str(item.get("record_id"))
                    for item in selected_items
                    if isinstance(item, dict) and item.get("record_id")
                )
        elif record_type == "snapshot":
            final_snapshot = dict(record.get("snapshot", {}))
        elif record_type == "event":
            kind = str(record.get("kind", "unknown"))
            payload = dict(record.get("payload", {}))
            events[kind] += 1
            first_event_tick.setdefault(kind, tick)
            if kind == "action_rejected":
                rejection_reasons[str(payload.get("reason", "unknown"))] += 1
            elif kind == "action_result" and payload.get("success"):
                successful_outcomes[str(payload.get("action", "unknown"))] += 1
            if kind == "action_result":
                agent = str(payload.get("agent", "unknown"))
                linked = retrieval_selected_ids[agent].intersection(
                    map(str, payload.get("causal_parents", []))
                )
                if linked:
                    retrieval_outcome_attempts += 1
                    retrieval_successful_outcomes += int(bool(payload.get("success")))
            elif kind == "sample_inspected":
                inspected_sites[str(payload.get("agent", "unknown"))].append(
                    (int(payload.get("x", -1)), int(payload.get("y", -1)))
                )
            elif kind == "insight_deposited" and payload.get("published"):
                publications[str(payload.get("record_id"))] = payload
            elif kind == "microbatch_fabricated":
                microbatches.append({"tick": tick, **payload})
            elif kind == "material_tested":
                material_tests.append({"tick": tick, **payload})
            elif kind == "task_claimed":
                task_claims.append({"tick": tick, **payload})
            elif kind == "artifact_built":
                artifact_builds.append({"tick": tick, **payload})
            elif kind == "artifact_program_installed":
                program_targets[str(payload.get("artifact_id", "unknown"))] += 1
            elif kind == "message_delivered":
                messages.append({"tick": tick, **payload})
            elif kind == "request_fulfilled":
                fulfillment_events.append({"tick": tick, "kind": kind, **payload})
            elif kind in {"knowledge_taught", "resource_traded"} and payload.get(
                "reply_to"
            ):
                legacy_fulfillment_events.append({"tick": tick, "kind": kind, **payload})

    citation_edges = 0
    cross_agent_edges = 0
    for publication in publications.values():
        child_author = publication.get("author")
        for parent_id in publication.get("causal_parents", []):
            parent = publications.get(str(parent_id))
            if parent is None:
                continue
            citation_edges += 1
            cross_agent_edges += int(parent.get("author") != child_author)

    peak_performance_by_artifact: dict[str, float] = {}
    if final_snapshot is not None:
        artifact_state = dict(final_snapshot.get("artifacts", {}))
        lifetime_peaks = artifact_state.get(
            "lifetime_peak_performance",
            artifact_state.get("peak_performance", []),
        )
        for artifact_id, value in zip(
            artifact_state.get("ids", []),
            lifetime_peaks,
            strict=False,
        ):
            peak_performance_by_artifact[str(artifact_id)] = float(value)

    artifact_lineages: list[dict[str, Any]] = []
    for artifact in artifact_builds:
        batch = dict(artifact.get("batch", {}))
        direct_parents = [str(item) for item in batch.get("causal_parents", [])]
        visited: set[str] = set()
        authors: set[str] = set()
        max_depth = 0
        stack = [(parent, 1) for parent in direct_parents]
        while stack:
            record_id, depth = stack.pop()
            if record_id in visited:
                continue
            publication = publications.get(record_id)
            if publication is None:
                continue
            visited.add(record_id)
            authors.add(str(publication.get("author", "unknown")))
            max_depth = max(max_depth, depth)
            stack.extend(
                (str(parent), depth + 1)
                for parent in publication.get("causal_parents", [])
                if str(parent) not in visited
            )
        creator = str(artifact.get("agent", "unknown"))
        external_authors = sorted(author for author in authors if author != creator)
        artifact_id = str(artifact.get("artifact_id", "unknown"))
        artifact_lineages.append(
            {
                "artifact_id": artifact_id,
                "artifact_name": str(artifact.get("artifact_name", "Untitled")),
                "creator": creator,
                "created_tick": int(artifact.get("tick", 0)),
                "direct_parent_ids": direct_parents,
                "ancestry_record_ids": sorted(visited),
                "ancestry_authors": sorted(authors),
                "external_ancestry_authors": external_authors,
                "citation_depth": max_depth,
                "cross_agent_epistemic_lineage": bool(external_authors),
                "peak_performance": peak_performance_by_artifact.get(artifact_id),
            }
        )

    total_outcomes = sum(successful_outcomes.values()) + sum(rejection_reasons.values())
    fabricated_at = {
        str(batch.get("batch_id")): int(batch["tick"])
        for batch in microbatches
        if batch.get("batch_id") is not None
    }
    tested_batch_ids = {
        str(result.get("batch_id"))
        for result in material_tests
        if result.get("batch_id") is not None
    }
    test_latencies = [
        int(result["tick"]) - fabricated_at[str(result["batch_id"])]
        for result in material_tests
        if str(result.get("batch_id")) in fabricated_at
    ]
    exploration: dict[str, Any] = {}
    for agent, sites in sorted(inspected_sites.items()):
        unique = len(set(sites))
        exploration[agent] = {
            "inspections": len(sites),
            "unique_sites": unique,
            "revisit_fraction": round(1.0 - unique / len(sites), 6) if sites else 0.0,
        }

    final_research = (
        dict(final_snapshot.get("research", {})) if final_snapshot is not None else {}
    )
    program_lineage = (
        list(final_snapshot.get("program_lineage", []))
        if final_snapshot is not None
        else []
    )
    program_catalog = (
        list(final_snapshot.get("program_catalog", []))
        if final_snapshot is not None
        else []
    )
    program_depths: dict[str, int] = {}
    for edge in program_lineage:
        parent = str(edge.get("parent_program_id", ""))
        child = str(edge.get("child_program_id", ""))
        program_depths[child] = max(
            program_depths.get(child, 0), program_depths.get(parent, 0) + 1
        )
    state_continuity: dict[str, Any] = {}
    all_goal_similarities: list[float] = []
    for agent, states in sorted(research_states.items()):
        goals = [str(state.get("goal", "")) for state in states]
        similarities = [
            _token_jaccard(previous, current)
            for previous, current in zip(goals, goals[1:], strict=False)
        ]
        all_goal_similarities.extend(similarities)
        state_continuity[agent] = {
            "updates": len(states),
            "initial_goal": goals[0] if goals else "",
            "final_goal": goals[-1] if goals else "",
            "mean_consecutive_goal_similarity": (
                round(float(np.mean(similarities)), 6) if similarities else None
            ),
            "final_hypothesis": str(states[-1].get("hypothesis", "")) if states else "",
            "final_next_checkpoint": (
                str(states[-1].get("next_checkpoint", "")) if states else ""
            ),
            "final_collaboration_need": (
                str(states[-1].get("collaboration_need", "")) if states else ""
            ),
        }
    if not fulfillment_events:
        fulfillment_events = legacy_fulfillment_events
    return {
        "path": str(source.resolve()),
        "metadata": header,
        "scenario_analysis": analyze_scenario_trace(source),
        "records": dict(sorted(record_types.items())),
        "last_recorded_tick": last_tick,
        "actions": {
            "executed": dict(sorted(executed_actions.items())),
            "planned": dict(sorted(planned_actions.items())),
            "non_wait_executed": sum(executed_actions.values())
            - executed_actions.get("WAIT", 0),
            "per_agent": {
                agent: dict(sorted(counts.items()))
                for agent, counts in sorted(per_agent_actions.items())
            },
        },
        "model": {
            "calls": record_types.get("model_trace", 0),
            "errors": sum(model_errors.values()),
            "provider_outage_attempts": record_types.get("provider_outage", 0),
            "rolled_back_plans": rolled_back_plans,
            "error_categories": dict(model_errors.most_common()),
            "usage": dict(sorted(usage.items())),
            "plan_length": _quantiles(plan_lengths),
            "queued_actions_replaced": {
                **_quantiles(queued_replacements),
                "total": sum(queued_replacements),
            },
            "research_state": {
                "updates": sum(len(states) for states in research_states.values()),
                "agents": len(research_states),
                "mean_consecutive_goal_similarity": (
                    round(float(np.mean(all_goal_similarities)), 6)
                    if all_goal_similarities
                    else None
                ),
                "per_agent": state_continuity,
            },
        },
        "execution": {
            "outcomes": total_outcomes,
            "rejections": sum(rejection_reasons.values()),
            "rejection_rate": round(
                sum(rejection_reasons.values()) / total_outcomes, 6
            )
            if total_outcomes
            else None,
            "rejection_reasons": dict(rejection_reasons.most_common()),
            "successful_actions": dict(sorted(successful_outcomes.items())),
        },
        "events": {
            "counts": dict(sorted(events.items())),
            "first_tick": dict(sorted(first_event_tick.items())),
        },
        "exploration": exploration,
        "information_flow": {
            "publications": len(publications),
            "publication_authors": len(
                {str(item.get("author")) for item in publications.values()}
            ),
            "citation_edges": citation_edges,
            "cross_agent_citation_edges": cross_agent_edges,
            "messages": len(messages),
            "addressed_messages": sum(
                bool(message.get("addressed")) for message in messages
            ),
            "message_replies": sum(bool(message.get("reply_to")) for message in messages),
            "request_fulfillments": len(fulfillment_events),
            "fulfillments": fulfillment_events,
        },
        "program_culture": {
            "programs": len(program_catalog),
            "forks": len(program_lineage),
            "maximum_lineage_depth": max(program_depths.values(), default=0),
            "lineage": program_lineage,
        },
        "retrieval": {
            "selected_records": retrieval_selected,
            "selected_records_later_cited": retrieval_used,
            "citation_rate": round(
                retrieval_used / max(1, retrieval_selected), 6
            ),
            "causal_outcome_attempts": retrieval_outcome_attempts,
            "causal_success_rate": round(
                retrieval_successful_outcomes / max(1, retrieval_outcome_attempts), 6
            ),
        },
        "experimental_cycle": {
            "microbatches_fabricated": len(microbatches),
            "microbatches_tested": len(material_tests),
            "test_completion_fraction": (
                round(len(material_tests) / len(microbatches), 6)
                if microbatches
                else None
            ),
            "fabrication_to_test_latency_ticks": _quantiles(test_latencies),
            "untested_batch_ids": sorted(set(fabricated_at) - tested_batch_ids),
            "batches": microbatches,
            "tests": material_tests,
        },
        "coordination": {
            "task_claims": task_claims,
            "distinct_task_claimants": len(
                {str(item.get("agent")) for item in task_claims}
            ),
        },
        "artifacts": {
            "built": len(artifact_builds),
            "named": [
                str(item.get("artifact_name", "Untitled")) for item in artifact_builds
            ],
            "program_installs_by_artifact": dict(sorted(program_targets.items())),
            "cross_agent_epistemic_lineage_count": sum(
                int(item["cross_agent_epistemic_lineage"])
                for item in artifact_lineages
            ),
            "best_cross_agent_lineage_peak_performance": max(
                (
                    float(item["peak_performance"])
                    for item in artifact_lineages
                    if item["cross_agent_epistemic_lineage"]
                    and item["peak_performance"] is not None
                ),
                default=None,
            ),
            "lineages": artifact_lineages,
        },
        "final_research": final_research,
    }


def render_trace_lineage_figure(
    path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    """Render a deterministic citation-to-artifact diagnostic for one episode."""

    source = Path(path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(destination / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:  # pragma: no cover - exercised without analysis extra
        raise RuntimeError(
            "lineage figures require `python -m pip install -e '.[analysis]'`"
        ) from exc

    report = analyze_trace(source)
    publications: dict[str, dict[str, Any]] = {}
    for record in read_records(source):
        if record.get("type") != "event" or record.get("kind") != "insight_deposited":
            continue
        payload = dict(record.get("payload", {}))
        if not payload.get("published"):
            continue
        publications[str(payload.get("record_id"))] = {
            **payload,
            "tick": int(record.get("tick", 0)),
        }

    lineages = list(report["artifacts"]["lineages"])
    authors = sorted(
        {
            str(item.get("author", "unknown"))
            for item in publications.values()
        }
        | {str(item["creator"]) for item in lineages}
    )
    author_y = {author: index for index, author in enumerate(authors)}
    cmap = plt.get_cmap("tab20")
    author_color = {
        author: cmap(index % 20) for index, author in enumerate(authors)
    }
    artifact_ancestry = {
        record_id
        for lineage in lineages
        for record_id in lineage["ancestry_record_ids"]
    }

    figure, (network_axis, performance_axis) = plt.subplots(
        1,
        2,
        figsize=(14.5, 6.8),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )
    for record_id, publication in publications.items():
        child_x = int(publication["tick"])
        child_y = author_y[str(publication.get("author", "unknown"))]
        for parent_id in publication.get("causal_parents", []):
            parent = publications.get(str(parent_id))
            if parent is None:
                continue
            parent_x = int(parent["tick"])
            parent_y = author_y[str(parent.get("author", "unknown"))]
            highlighted = record_id in artifact_ancestry and str(parent_id) in artifact_ancestry
            network_axis.plot(
                [parent_x, child_x],
                [parent_y, child_y],
                color="#334155" if highlighted else "#cbd5e1",
                alpha=0.58 if highlighted else 0.28,
                linewidth=1.2 if highlighted else 0.65,
                zorder=1,
            )
    for record_id, publication in publications.items():
        author = str(publication.get("author", "unknown"))
        highlighted = record_id in artifact_ancestry
        network_axis.scatter(
            int(publication["tick"]),
            author_y[author],
            s=34 if highlighted else 18,
            color=author_color[author],
            edgecolor="#0f172a" if highlighted else "none",
            linewidth=0.55,
            alpha=0.95 if highlighted else 0.42,
            zorder=3,
        )
    for lineage in lineages:
        if not lineage["direct_parent_ids"]:
            continue
        artifact_x = int(lineage["created_tick"])
        artifact_y = author_y[str(lineage["creator"])]
        for parent_id in lineage["direct_parent_ids"]:
            parent = publications.get(str(parent_id))
            if parent is None:
                continue
            network_axis.plot(
                [int(parent["tick"]), artifact_x],
                [author_y[str(parent.get("author", "unknown"))], artifact_y],
                color="#e11d48",
                linewidth=1.35,
                alpha=0.72,
                linestyle="--",
                zorder=2,
            )
        network_axis.scatter(
            artifact_x,
            artifact_y,
            marker="*",
            s=145,
            color="#e11d48",
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
    network_axis.set_title("A  Emergent citation flow into built artifacts", loc="left")
    network_axis.set_xlabel("Simulation tick")
    network_axis.set_yticks(
        range(len(authors)),
        [author.replace("agent_", "A") for author in authors],
    )
    network_axis.set_ylabel("Publication author")
    network_axis.grid(axis="x", color="#e2e8f0", linewidth=0.7)
    network_axis.spines[["top", "right"]].set_visible(False)
    network_axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#64748b",
                label="publication",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markerfacecolor="#e11d48",
                markersize=12,
                label="built artifact",
            ),
        ],
        frameon=False,
        loc="upper left",
    )

    artifact_names = [str(item["artifact_name"]) for item in lineages]
    performances = [
        float(item["peak_performance"] or 0.0) for item in lineages
    ]
    external_counts = [len(item["external_ancestry_authors"]) for item in lineages]
    bar_colors = [plt.get_cmap("viridis")(min(count, 4) / 4) for count in external_counts]
    positions = np.arange(len(lineages))
    performance_axis.barh(positions, performances, color=bar_colors, height=0.72)
    performance_axis.set_yticks(
        positions,
        [name if len(name) <= 34 else name[:31] + "..." for name in artifact_names],
        fontsize=8,
    )
    performance_axis.invert_yaxis()
    performance_axis.set_xlabel("Peak measured field performance")
    performance_axis.set_title("B  Lineage depth does not imply function", loc="left")
    target = (
        report.get("metadata", {})
        .get("config", {})
        .get("science", {})
        .get("target_artifact_performance")
    )
    if isinstance(target, (int, float)):
        performance_axis.axvline(
            float(target), color="#dc2626", linestyle=":", linewidth=1.3, label="target"
        )
    for position, (value, external_count) in enumerate(
        zip(performances, external_counts, strict=True)
    ):
        performance_axis.text(
            value + 0.002,
            position,
            f"{external_count} external author{'s' if external_count != 1 else ''}",
            va="center",
            fontsize=7.5,
            color="#334155",
        )
    performance_axis.grid(axis="x", color="#e2e8f0", linewidth=0.7)
    performance_axis.spines[["top", "right"]].set_visible(False)
    if isinstance(target, (int, float)):
        performance_axis.legend(frameon=False, loc="lower right")

    figure.suptitle(
        "Epistemic emergence can reach physical artifacts without a performance advantage",
        fontsize=14,
        y=1.01,
    )
    figure.tight_layout()
    stem = source.stem + "-lineage"
    png = destination / f"{stem}.png"
    pdf = destination / f"{stem}.pdf"
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return {"png": str(png.resolve()), "pdf": str(pdf.resolve())}

"""Objective collective-behavior metrics for preregistered swarm comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .simulation import BioFoundrySimulation
from .types import ActionType


def _specialization(counts: NDArray[np.float64]) -> tuple[float, float]:
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0, 0.0
    joint = counts / total
    agent_marginal = joint.sum(axis=1, keepdims=True)
    action_marginal = joint.sum(axis=0, keepdims=True)
    expected = agent_marginal @ action_marginal
    mask = joint > 0.0
    mutual_information = float(
        np.sum(joint[mask] * np.log2(joint[mask] / expected[mask]))
    )
    action_probs = action_marginal.ravel()
    action_probs = action_probs[action_probs > 0.0]
    action_entropy = float(-np.sum(action_probs * np.log2(action_probs)))
    normalized = mutual_information / action_entropy if action_entropy > 1e-12 else 0.0
    return mutual_information, normalized


def swarm_metrics(
    simulation: BioFoundrySimulation,
    action_histogram: NDArray[np.int64],
) -> dict[str, Any]:
    """Summarize specialization and causal composition without an LLM judge."""
    scientific_actions = [
        ActionType.INSPECT,
        ActionType.HARVEST,
        ActionType.DEPOSIT,
        ActionType.OPERATE,
        ActionType.TEST,
        ActionType.PROPOSE_RECIPE,
        ActionType.BUILD,
        ActionType.COMMUNICATE,
        ActionType.PUBLISH,
        ActionType.WRITE_PROGRAM,
        ActionType.FORK_PROGRAM,
        ActionType.CLAIM_TASK,
        ActionType.TEACH,
        ActionType.TRADE,
        ActionType.COMBINE_DESIGN,
    ]
    scientific = action_histogram[:, [int(action) for action in scientific_actions]].astype(
        np.float64
    )
    total = float(scientific.sum())
    mutual_information, normalized_specialization = _specialization(scientific)
    null_values: list[float] = []
    if total > 0.0:
        action_labels = np.repeat(
            np.arange(scientific.shape[1]), scientific.sum(axis=0).astype(np.int64)
        )
        agent_totals = scientific.sum(axis=1).astype(np.int64)
        rng = np.random.default_rng(simulation.config.simulation.seed + 918_273)
        for _ in range(500):
            shuffled = rng.permutation(action_labels)
            null_counts = np.zeros_like(scientific)
            cursor = 0
            for agent, count in enumerate(agent_totals):
                assigned = shuffled[cursor : cursor + count]
                null_counts[agent] = np.bincount(
                    assigned, minlength=scientific.shape[1]
                )
                cursor += count
            null_values.append(_specialization(null_counts)[1])
    null_mean = float(np.mean(null_values)) if null_values else 0.0
    null_high = float(np.quantile(null_values, 0.95)) if null_values else 0.0
    permutation_p = (
        (1.0 + sum(value >= normalized_specialization for value in null_values))
        / (1.0 + len(null_values))
        if null_values
        else 1.0
    )

    dominant_actions: dict[str, str] = {}
    for index, agent_id in enumerate(simulation.agent_ids):
        row = action_histogram[index]
        non_wait_row = row.copy()
        non_wait_row[int(ActionType.WAIT)] = 0
        if int(non_wait_row.sum()) > 0:
            dominant_actions[agent_id] = ActionType(int(np.argmax(non_wait_row))).name

    max_contributors = 0
    max_causal_parents = 0
    causal_composition_success = False
    for provenance in simulation.artifacts.provenance.values():
        contributors = len(set(provenance.get("contributors", [])))
        parents = len(set(provenance.get("causal_parents", [])))
        max_contributors = max(max_contributors, contributors)
        max_causal_parents = max(max_causal_parents, parents)
        causal_composition_success |= contributors >= 2 and parents >= 2

    if simulation.research is not None:
        card = simulation.research.scorecard(
            simulation.artifacts, len(simulation.archive.records)
        )

    depths: dict[str, int] = {}
    for edge in simulation.program_library.lineage_edges:
        parent = str(edge["parent_program_id"])
        child = str(edge["child_program_id"])
        depths[child] = max(depths.get(child, 0), depths.get(parent, 0) + 1)
    adoption_counts: dict[str, int] = {}
    verified_adoption_counts: dict[str, int] = {}
    for skills in simulation.program_library.skills.values():
        for program_id, entry in skills.items():
            adoption_counts[program_id] = adoption_counts.get(program_id, 0) + 1
            if entry.get("verified"):
                verified_adoption_counts[program_id] = (
                    verified_adoption_counts.get(program_id, 0) + 1
                )
    messages = list(simulation.message_records.values())
    retrieval_selected = 0
    retrieval_used = 0
    retrieval_outcome_attempts = 0
    retrieval_successful_outcomes = 0
    for memory in simulation.memories:
        retrieval_selected += sum(
            int(stats.get("selected", 0)) for stats in memory.retrieval_stats.values()
        )
        retrieval_used += sum(
            int(stats.get("used", 0)) for stats in memory.retrieval_stats.values()
        )
        retrieval_outcome_attempts += sum(
            int(stats.get("outcome_attempts", 0))
            for stats in memory.retrieval_stats.values()
        )
        retrieval_successful_outcomes += sum(
            int(stats.get("successful_outcomes", 0))
            for stats in memory.retrieval_stats.values()
        )
        causal_composition_success = bool(
            card["milestones"]["causally_composed_artifact"]
            and card["milestones"]["distributed_contribution"]
        )

    return {
        "scientific_actions": int(total),
        "specialization_mutual_information_bits": round(mutual_information, 6),
        "normalized_specialization": round(normalized_specialization, 6),
        "specialization_null_mean": round(null_mean, 6),
        "specialization_null_95": round(null_high, 6),
        "excess_normalized_specialization": round(
            normalized_specialization - null_mean, 6
        ),
        "specialization_permutation_p": round(permutation_p, 6),
        "dominant_action_by_agent": dominant_actions,
        "max_artifact_contributors": max_contributors,
        "max_artifact_causal_parents": max_causal_parents,
        "causal_composition_success": causal_composition_success,
        "combined_designs": (
            sum(1 for proposal in simulation.research.proposals if proposal["combined"])
            if simulation.research is not None
            else 0
        ),
        "tested_recipes": (
            len({test["recipe_id"] for test in simulation.research.tests})
            if simulation.research is not None
            else 0
        ),
        "programs_created": len(simulation.program_library.programs),
        "program_forks": len(simulation.program_library.lineage_edges),
        "maximum_program_lineage_depth": max(depths.values(), default=0),
        "maximum_program_adoption": max(adoption_counts.values(), default=0),
        "maximum_verified_program_adoption": max(
            verified_adoption_counts.values(), default=0
        ),
        "messages_delivered": len(messages),
        "addressed_message_fraction": round(
            sum(bool(message.get("addressed")) for message in messages)
            / max(1, len(messages)),
            6,
        ),
        "message_reply_fraction": round(
            sum(bool(message.get("reply_to")) for message in messages)
            / max(1, len(messages)),
            6,
        ),
        "request_fulfillment_fraction": round(
            sum(bool(message.get("fulfillments")) for message in messages)
            / max(1, len(messages)),
            6,
        ),
        "retrieval_records_selected": retrieval_selected,
        "retrieval_records_cited": retrieval_used,
        "retrieval_citation_rate": round(
            retrieval_used / max(1, retrieval_selected), 6
        ),
        "retrieval_causal_outcome_attempts": retrieval_outcome_attempts,
        "retrieval_causal_success_rate": round(
            retrieval_successful_outcomes / max(1, retrieval_outcome_attempts), 6
        ),
        "active_agents": int(np.count_nonzero(simulation.population.active)),
        "maximum_agent_generation": int(
            np.max(simulation.population.generation, initial=0)
        ),
    }

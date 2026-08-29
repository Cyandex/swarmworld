"""Deterministic open-loop interventions on recorded multi-agent action traces."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tqdm import tqdm

from . import ENGINE_REVISION
from .config import config_from_dict
from .events import read_records
from .memory import MemoryRecord
from .simulation import BioFoundrySimulation
from .types import AgentAction


def load_action_trace(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header: dict[str, Any] | None = None
    actions: list[dict[str, Any]] = []
    model_traces: dict[int, list[dict[str, Any]]] = {}
    for record in read_records(path):
        if record.get("type") == "header":
            header = record
        elif record.get("type") == "model_trace":
            model_traces.setdefault(int(record.get("tick", -1)), []).append(record)
        elif record.get("type") == "actions":
            action_record = dict(record)
            action_record["model_traces"] = model_traces.pop(
                int(record.get("tick", -1)), []
            )
            actions.append(action_record)
    if header is None:
        raise ValueError("action trace has no replay header")
    metadata = header.get("metadata", {})
    recorded_revision = metadata.get("engine_revision")
    if recorded_revision is None:
        raise ValueError(
            "counterfactual replay requires an engine-revision marker; this legacy "
            "trace supports snapshot-integrity verification only"
        )
    if recorded_revision != ENGINE_REVISION:
        raise ValueError(
            "counterfactual replay requires the generating engine revision: "
            f"trace={recorded_revision}, current={ENGINE_REVISION}"
        )
    config_data = metadata.get("config")
    if not isinstance(config_data, dict):
        raise ValueError("replay header does not contain a complete configuration")
    policy = str(metadata.get("policy", ""))
    for record in actions:
        record.setdefault("recorded_policy", policy)
    return config_data, actions


def apply_recorded_model_trace(
    simulation: BioFoundrySimulation, record: dict[str, Any]
) -> None:
    """Replay authoritative private-memory writes made by the LLM policy."""

    if record.get("planning_committed") is False:
        return
    research_state = record.get("research_state")
    agent_id = str(record.get("agent", ""))
    index = simulation.population.id_to_index.get(agent_id)
    if not isinstance(research_state, dict) or index is None:
        return
    tick = int(record.get("tick", simulation.tick))
    simulation.memories[index].remember(
        MemoryRecord(
            record_id=f"research_state_{agent_id}_{tick:08d}",
            tick=tick,
            kind="research_state",
            content=json.dumps(research_state, sort_keys=True),
            salience=0.95,
            causal_parents=tuple(
                sorted(map(str, research_state.get("evidence_ids", [])))
            ),
        ),
        notebook=True,
    )


def acknowledge_recorded_macroturns(
    simulation: BioFoundrySimulation,
    record: dict[str, Any],
    *,
    policy: str | None = None,
) -> None:
    """Replay policy-side scheduling mutations that occur before each world step."""

    recorded = record.get("macroturn_agents")
    if isinstance(recorded, list):
        simulation.acknowledge_macroturn([str(agent) for agent in recorded])
        return
    policy_name = policy or str(record.get("recorded_policy", ""))
    if policy_name == "llm":
        # Backward compatibility for traces written before macroturn_agents was
        # explicit. The scheduler is deterministic from the pre-step state.
        simulation.acknowledge_macroturn(simulation.scheduled_macro_agents())


def replay_intervention(
    config_data: dict[str, Any],
    action_records: Iterable[dict[str, Any]],
    removed_agents: set[str] | None = None,
    *,
    description: str = "Counterfactual replay",
    show_progress: bool = True,
) -> BioFoundrySimulation:
    """Replay recorded actions after replacing selected agents with passive controls."""
    simulation = BioFoundrySimulation(config_from_dict(config_data))
    records = list(action_records)
    iterator: Iterable[dict[str, Any]] = records
    if show_progress:
        iterator = tqdm(records, desc=description, unit="tick", leave=False)
    removed = removed_agents or set()
    for record in iterator:
        expected_tick = int(record.get("tick", -1))
        if expected_tick != simulation.tick:
            raise ValueError(
                f"action trace tick {expected_tick} does not match {simulation.tick}"
            )
        raw_actions = dict(record.get("actions", {}))
        for model_trace in record.get("model_traces", []):
            apply_recorded_model_trace(simulation, model_trace)
        acknowledge_recorded_macroturns(simulation, record)
        for agent_id in removed:
            if agent_id in simulation.population.id_to_index:
                raw_actions[agent_id] = AgentAction().as_dict()
        simulation.step(raw_actions)
    return simulation


def outcome_summary(simulation: BioFoundrySimulation) -> dict[str, Any]:
    result: dict[str, Any] = {
        "tick": simulation.tick,
        "artifacts": simulation.artifacts.count,
        "artifact_score": round(simulation.artifacts.summary_score(), 6),
        "state_digest": simulation.state_digest(),
    }
    if simulation.research is not None:
        result["research"] = simulation.research.scorecard(
            simulation.artifacts, len(simulation.archive.records)
        )
    return result


def claimed_contributors(simulation: BioFoundrySimulation) -> list[str]:
    """Return material contributors and cited public-record authors."""
    contributors: set[str] = set()
    for provenance in simulation.artifacts.provenance.values():
        contributors.update(str(agent) for agent in provenance.get("contributors", []))
        for parent in provenance.get("causal_parents", []):
            if (
                simulation.research is not None
                and parent in simulation.research.publications
            ):
                contributors.add(
                    str(simulation.research.publications[parent]["author"])
                )
                continue
            record = simulation.archive.by_id.get(parent)
            if record is None or "|" not in record.content:
                continue
            author = record.content.split("|", 1)[0].strip()
            if author.startswith("agent_"):
                contributors.add(author)
    return sorted(contributors)


def compare_outcomes(
    factual: dict[str, Any], counterfactual: dict[str, Any]
) -> dict[str, Any]:
    factual_research = factual.get("research", {})
    counterfactual_research = counterfactual.get("research", {})
    return {
        "outcome_lost": bool(
            factual_research.get("outcome_success", False)
            and not counterfactual_research.get("outcome_success", False)
        ),
        "emergent_outcome_lost": bool(
            factual_research.get("emergent_success", False)
            and not counterfactual_research.get("emergent_success", False)
        ),
        "artifact_performance_loss": round(
            float(factual_research.get("best_artifact_performance", 0.0))
            - float(counterfactual_research.get("best_artifact_performance", 0.0)),
            6,
        ),
        "material_utility_loss": round(
            float(factual_research.get("best_material_utility", 0.0))
            - float(counterfactual_research.get("best_material_utility", 0.0)),
            6,
        ),
        "behavioral_novelty_loss": round(
            float(factual_research.get("best_behavioral_novelty", 0.0))
            - float(counterfactual_research.get("best_behavioral_novelty", 0.0)),
            6,
        ),
        "portfolio_resilience_loss": round(
            float(factual_research.get("portfolio_resilience", 0.0))
            - float(counterfactual_research.get("portfolio_resilience", 0.0)),
            6,
        ),
        "service_breadth_loss": int(factual_research.get("service_breadth", 0))
        - int(counterfactual_research.get("service_breadth", 0)),
        "self_organized_portfolio_lost": bool(
            factual_research.get("self_organized_portfolio", False)
            and not counterfactual_research.get("self_organized_portfolio", False)
        ),
    }

"""Command-line entry points for validation, simulation, serving, and scaling."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from . import ENGINE_REVISION, __version__
from .analysis import analyze_study
from .capabilities import (
    addressing_available,
    available_action_types,
    replies_available,
)
from .config import GameConfig, config_from_dict, load_config
from .counterfactuals import (
    acknowledge_recorded_macroturns,
    apply_recorded_model_trace,
    claimed_contributors,
    compare_outcomes,
    load_action_trace,
    outcome_summary,
    replay_intervention,
)
from .dynamics import SocietyDynamicsTracker, regional_crowding
from .events import EventRecorder, digest_snapshot, read_records
from .metrics import swarm_metrics
from .policies.llm import (
    SCENARIO_ACTION_MANUAL,
    SCENARIO_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    LLMPolicy,
    capability_action_manual,
)
from .policies.scripted import (
    OracleResearchSocietyPolicy,
    ScalableForagerPolicy,
    ScenarioIndustryPolicy,
    ScriptedBioFoundryPolicy,
)
from .providers.openai_compatible import OpenAICompatibleProvider
from .scenario_analysis import analyze_scenario_trace
from .scenarios import load_scenario, scenario_json, scenario_metadata, validate_scenario
from .simulation import BioFoundrySimulation
from .structured_output import bounded_action_plan_text_format
from .technology_ecology import (
    evaluate_technological_ecosystem,
    evaluate_technological_ecosystem_generalization,
    render_technology_ecology_figures,
    snapshot_dynamics,
    technology_ecology_summary,
)
from .trace_analysis import analyze_trace, render_trace_lineage_figure
from .types import ActionType


def _device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _load(path: str) -> GameConfig:
    config = load_config(path)
    config.validate()
    return config


def doctor(args: argparse.Namespace) -> int:
    config = _load(args.config)
    scenario = load_scenario(config.world.scenario_package)
    modules = [
        "numpy",
        "yaml",
        "tqdm",
        "jsonschema",
        "gymnasium",
        "pettingzoo",
        "fastapi",
        "uvicorn",
        "websockets",
        "torch",
        "transformers",
    ]
    found = {module: bool(importlib.util.find_spec(module)) for module in modules}
    godot = shutil.which("godot") or shutil.which("godot4")
    report = {
        "biofoundry_version": __version__,
        "engine_revision": ENGINE_REVISION,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "accelerator_for_local_llm": _device(),
        "modules": found,
        "godot_executable": godot,
        "config_valid": True,
        "agents": config.population.agents,
        "world": [config.world.width, config.world.height],
        "scenario": scenario_metadata(scenario),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    required = ("numpy", "yaml", "tqdm", "jsonschema", "gymnasium", "pettingzoo")
    return 0 if all(found[name] for name in required) else 1


def _policy(name: str, config: GameConfig):
    scenario = load_scenario(config.world.scenario_package)
    if name == "scripted":
        if scenario is not None:
            return ScenarioIndustryPolicy()
        return ScriptedBioFoundryPolicy()
    if name == "scalable":
        return ScalableForagerPolicy()
    if name == "research-oracle":
        return OracleResearchSocietyPolicy()
    return LLMPolicy(
        OpenAICompatibleProvider(
            config.llm,
            allowed_action_types=available_action_types(config),
            allow_addressing=addressing_available(config),
            allow_replies=replies_available(config),
            resource_names=scenario.resource_names if scenario is not None else None,
            operation_names=scenario.operation_names if scenario is not None else None,
        )
    )


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_commit() -> str | None:
    """Return immutable source provenance without making git a runtime dependency."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def _git_dirty() -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(completed.stdout.strip())


def _reproducibility_metadata(config: GameConfig) -> dict[str, Any]:
    capabilities = available_action_types(config)
    scenario = load_scenario(config.world.scenario_package)
    schema = bounded_action_plan_text_format(
        config.llm.max_plan_actions,
        grammar_safe_numbers=config.llm.grammar_safe_numbers,
        allowed_action_types=capabilities,
        allow_addressing=addressing_available(config),
        allow_replies=replies_available(config),
        resource_names=scenario.resource_names if scenario is not None else None,
        operation_names=scenario.operation_names if scenario is not None else None,
    )["schema"]
    prompt_text = (
        SCENARIO_SYSTEM_PROMPT.format(
            scenario_name=scenario.name,
            max_plan_actions=config.llm.max_plan_actions,
            scenario_instructions=scenario.agent_prompt,
        )
        + "\n"
        + SCENARIO_ACTION_MANUAL
        if scenario is not None
        else SYSTEM_PROMPT.format(max_plan_actions=config.llm.max_plan_actions)
        + "\n"
        + capability_action_manual(capabilities)
    )
    return {
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "config_sha256": _sha256_json(config.as_dict()),
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "action_schema_sha256": _sha256_json(schema),
        "capabilities": [action.name for action in capabilities],
        "scenario": scenario_metadata(scenario),
    }


def _discovery_frontier(
    points: list[dict[str, Any]],
    *,
    horizon: int,
    target: float,
) -> dict[str, Any]:
    """Summarize the best measured artifact attained through simulation time."""

    if not points:
        return {
            "final_performance": 0.0,
            "normalized_auc": 0.0,
            "time_to_target": None,
            "target": float(target),
        }
    ticks = np.asarray([int(point["tick"]) for point in points], dtype=np.float64)
    values = np.maximum.accumulate(
        np.asarray(
            [float(point.get("best_artifact_performance", 0.0)) for point in points],
            dtype=np.float64,
        )
    )
    if ticks[-1] < horizon:
        ticks = np.append(ticks, float(horizon))
        values = np.append(values, values[-1])
    auc = float(np.trapezoid(values, ticks) / max(1.0, float(horizon)))
    reached = np.nonzero(values >= float(target))[0]
    return {
        "final_performance": round(float(values[-1]), 6),
        "normalized_auc": round(auc, 6),
        "time_to_target": int(ticks[reached[0]]) if reached.size else None,
        "target": round(float(target), 6),
    }


def _normalize_held_out_checkpoints(
    checkpoints: list[int] | None,
    *,
    ticks: int,
) -> list[int]:
    """Return unique, ordered assay ticks after validating the discovery horizon."""

    values = sorted({int(value) for value in checkpoints or []})
    invalid = [value for value in values if value <= 0 or value > int(ticks)]
    if invalid:
        raise ValueError(
            "held-out evaluation checkpoints must be positive and no greater than "
            f"the {ticks}-tick discovery horizon; invalid values: {invalid}"
        )
    return values


def _checkpoint_evaluation(
    simulation: BioFoundrySimulation,
    *,
    held_out_evaluation_seeds: list[int],
    dynamics: SocietyDynamicsTracker,
    action_histogram: np.ndarray,
    model_calls: int,
    model_usage: dict[str, int],
) -> dict[str, Any]:
    """Measure one frozen discovery state without advancing or modifying it."""

    if not dynamics.points or int(dynamics.points[-1]["tick"]) != simulation.tick:
        dynamics.record(simulation)
    before = simulation.state_digest()
    held_out = evaluate_technological_ecosystem_generalization(
        simulation,
        held_out_evaluation_seeds,
        horizon=simulation.config.evaluation.horizon,
    )
    after = simulation.state_digest()
    if after != before:
        raise RuntimeError("held-out checkpoint evaluation modified discovery state")

    research: dict[str, Any] = {}
    if simulation.research is not None:
        scorecard = simulation.research.scorecard(
            simulation.artifacts, len(simulation.archive.records)
        )
        research_keys = (
            "outcome_success",
            "emergent_success",
            "best_artifact_performance",
            "best_current_program_peak_performance",
            "best_final_state_artifact_performance",
            "portfolio_resilience",
            "service_breadth",
            "self_organized_portfolio",
            "best_behavioral_novelty",
            "best_material_utility",
        )
        research = {key: scorecard.get(key) for key in research_keys}
        research["validated_inventions"] = dict(scorecard.get("counts", {})).get(
            "validated_inventions", 0
        )

    ecology = technology_ecology_summary(simulation)
    ecology_keys = (
        "artifact_count",
        "collaborative_artifact_fraction",
        "participating_agents",
        "publication_reuse_fraction",
        "program_forks",
        "cross_agent_program_fork_fraction",
        "maximum_program_lineage_depth",
        "maximum_verified_program_adoption",
        "causal_closure_fraction",
        "causal_closure_gain_fraction",
    )
    compact_ecology = {key: ecology.get(key) for key in ecology_keys if key in ecology}
    return {
        "tick": simulation.tick,
        "state_digest": before,
        "discovery_state_frozen": True,
        "agent_actions_per_schedule": 0,
        "artifacts": simulation.artifacts.count,
        "artifact_score": simulation.artifacts.summary_score(),
        "model_calls": int(model_calls),
        "model_usage": dict(sorted(model_usage.items())),
        "discovery_frontier": _discovery_frontier(
            dynamics.points,
            horizon=simulation.tick,
            target=simulation.config.science.target_artifact_performance,
        ),
        "research": research,
        "technology_ecology": compact_ecology,
        "swarm_metrics": swarm_metrics(simulation, action_histogram),
        "held_out_generalization": held_out,
    }


async def _simulate_episode(
    config: GameConfig,
    ticks: int,
    policy_name: str,
    output: str,
    *,
    held_out_evaluation_seeds: list[int] | None = None,
    held_out_evaluation_checkpoints: list[int] | None = None,
) -> dict[str, Any]:
    checkpoint_ticks = _normalize_held_out_checkpoints(held_out_evaluation_checkpoints, ticks=ticks)
    if checkpoint_ticks and not held_out_evaluation_seeds:
        raise ValueError("held-out evaluation checkpoints require held-out evaluation seeds")
    if checkpoint_ticks and not config.evaluation.enabled:
        raise ValueError("held-out evaluation checkpoints require evaluation.enabled")
    checkpoint_tick_set = set(checkpoint_ticks)
    checkpoint_evaluations: list[dict[str, Any]] = []
    simulation = BioFoundrySimulation(config)
    initial_positions = [
        [int(simulation.population.x[index]), int(simulation.population.y[index])]
        for index in range(simulation.population.size)
    ]
    policy = _policy(policy_name, config)
    dynamics = SocietyDynamicsTracker()
    dynamics_interval = max(1, config.simulation.snapshot_interval // 4)
    dynamics.record(simulation)
    action_histogram = np.zeros((simulation.population.size, len(ActionType)), dtype=np.int64)
    event_counts: dict[str, int] = {}
    model_calls = 0
    decision_opportunities = 0
    model_errors = 0
    model_usage: dict[str, int] = {}
    metadata = {
        "config": config.as_dict(),
        "engine_revision": ENGINE_REVISION,
        "ticks_requested": ticks,
        "policy": policy_name,
        "version": __version__,
        "reproducibility": _reproducibility_metadata(config),
        "scenario": scenario_metadata(simulation.scenario),
    }
    if checkpoint_ticks:
        metadata["held_out_evaluation_seeds"] = list(held_out_evaluation_seeds or [])
        metadata["held_out_evaluation_checkpoints"] = checkpoint_ticks
    with EventRecorder(
        output,
        metadata,
        compression=config.trace.compression,
        deduplicate_prompts=config.trace.deduplicate_prompts,
    ) as recorder:
        recorder.write_snapshot(
            simulation.snapshot(display_limit=None),
            state_digest=simulation.state_digest(),
        )
        for _ in tqdm(range(ticks), desc="BioFoundry ticks", unit="tick"):
            macroturn_agents: list[str] = []
            if isinstance(policy, LLMPolicy):
                while True:
                    macroturn_agents = simulation.scheduled_macro_agents()
                    committed = await policy.refresh_plans(simulation)
                    for trace in policy.last_traces:
                        recorder.write_record({"type": "model_trace", **trace})
                        model_calls += 1
                        model_errors += int(bool(trace.get("error")))
                        for key, value in trace.get("usage", {}).items():
                            if isinstance(value, int):
                                model_usage[key] = model_usage.get(key, 0) + value
                    if committed:
                        decision_opportunities += len(policy.last_traces)
                        actions = policy.next_actions(simulation)
                        break
                    recorder.write_record(
                        {
                            "type": "provider_outage",
                            "tick": simulation.tick,
                            "macroturn_agents": macroturn_agents,
                            "errors": [
                                {
                                    "agent": trace.get("agent", ""),
                                    "error": str(trace.get("error", ""))[:280],
                                }
                                for trace in policy.last_traces
                            ],
                            "world_advanced": False,
                        }
                    )
                    await asyncio.sleep(config.llm.provider_retry_seconds)
            else:
                actions = policy.actions(simulation)
            for agent_id, action in actions.items():
                index = simulation.population.id_to_index[agent_id]
                action_histogram[index, int(action.verb)] += 1
            recorder.write_record(
                {
                    "type": "actions",
                    "tick": simulation.tick,
                    "macroturn_agents": macroturn_agents,
                    "actions": {agent_id: action.as_dict() for agent_id, action in actions.items()},
                }
            )
            result = simulation.step(actions)
            for event in result.events:
                event_counts[event.kind] = event_counts.get(event.kind, 0) + 1
            recorder.write_events(result.events)
            if simulation.tick % dynamics_interval == 0:
                dynamics.record(simulation)
            periodic_snapshot = simulation.tick % config.simulation.snapshot_interval == 0
            if periodic_snapshot:
                recorder.write_snapshot(
                    simulation.snapshot(display_limit=None),
                    state_digest=simulation.state_digest(),
                )
            if simulation.tick in checkpoint_tick_set:
                if not periodic_snapshot:
                    recorder.write_snapshot(
                        simulation.snapshot(display_limit=None),
                        state_digest=simulation.state_digest(),
                    )
                checkpoint_evaluations.append(
                    _checkpoint_evaluation(
                        simulation,
                        held_out_evaluation_seeds=list(held_out_evaluation_seeds or []),
                        dynamics=dynamics,
                        action_histogram=action_histogram,
                        model_calls=model_calls,
                        model_usage=model_usage,
                    )
                )
        digest = simulation.state_digest()
        recorder.write_snapshot(
            simulation.snapshot(display_limit=None),
            state_digest=digest,
        )
    summary = {
        "biofoundry_version": __version__,
        "engine_revision": ENGINE_REVISION,
        "scenario": scenario_metadata(simulation.scenario),
        "position_summary_version": 2,
        "ticks": simulation.tick,
        "agents": simulation.population.size,
        "initial_positions": initial_positions,
        "final_positions": [
            [int(simulation.population.x[index]), int(simulation.population.y[index])]
            for index in range(simulation.population.size)
        ],
        "macroturn_phases": [
            int((config.population.macro_phase_offset + index) % config.population.macro_interval)
            for index in range(simulation.population.size)
        ],
        "artifacts": simulation.artifacts.count,
        "artifact_score": simulation.artifacts.summary_score(),
        "state_digest": digest,
        "output": str(Path(output).resolve()),
        "event_counts": dict(sorted(event_counts.items())),
        "action_counts": {
            action.name: int(action_histogram[:, int(action)].sum()) for action in ActionType
        },
        "model_calls": model_calls,
        "decision_opportunities": (
            decision_opportunities if isinstance(policy, LLMPolicy) else None
        ),
        "model_errors": model_errors,
        "model_usage": dict(sorted(model_usage.items())),
        "swarm_metrics": swarm_metrics(simulation, action_histogram),
        "technology_ecology": technology_ecology_summary(simulation),
        "mobility": {
            "successful_moves": int(np.sum(simulation.distance_traveled)),
            "mean_distance_traveled": round(float(np.mean(simulation.distance_traveled)), 6),
            "moving_agent_fraction": round(float(np.mean(simulation.distance_traveled > 0)), 6),
            "mean_distinct_cells_visited": round(
                float(np.mean([len(cells) for cells in simulation.visited_cells])),
                6,
            ),
            "regional_crowding": round(
                regional_crowding(
                    simulation.population.x[simulation.population.active],
                    simulation.population.y[simulation.population.active],
                    width=simulation.world.width,
                    height=simulation.world.height,
                ),
                6,
            ),
        },
        "action_budget": {
            "configured": config.simulation.action_attempt_budget,
            "used": simulation.action_attempts_used,
            "blocked": simulation.action_attempts_blocked,
            "exhausted_tick": simulation.action_budget_exhausted_tick,
        },
    }
    if not dynamics.points or dynamics.points[-1]["tick"] != simulation.tick:
        dynamics.record(simulation)
    summary["discovery_frontier"] = _discovery_frontier(
        dynamics.points,
        horizon=simulation.tick,
        target=config.science.target_artifact_performance,
    )
    summary["dynamics_history"] = dynamics.packet(sample_interval=dynamics_interval)
    if checkpoint_ticks:
        summary["held_out_checkpoints"] = checkpoint_evaluations
    if simulation.research is not None:
        summary["research"] = simulation.research.scorecard(
            simulation.artifacts, len(simulation.archive.records)
        )
    if config.evaluation.enabled:
        summary["ecosystem_assay"] = evaluate_technological_ecosystem(
            simulation,
            horizon=config.evaluation.horizon,
            detailed=False,
        )
        if held_out_evaluation_seeds:
            final_checkpoint = next(
                (item for item in checkpoint_evaluations if int(item["tick"]) == simulation.tick),
                None,
            )
            summary["held_out_generalization"] = (
                copy.deepcopy(final_checkpoint["held_out_generalization"])
                if final_checkpoint is not None
                else evaluate_technological_ecosystem_generalization(
                    simulation,
                    held_out_evaluation_seeds,
                    horizon=config.evaluation.horizon,
                )
            )
    if simulation.scenario is not None:
        summary["scenario_analysis"] = analyze_scenario_trace(output)
    return summary


def simulate(args: argparse.Namespace) -> int:
    config = _load(args.config)
    if args.seed is not None:
        config = replace(config, simulation=replace(config.simulation, seed=args.seed))
    if args.agents is not None:
        config = replace(config, population=replace(config.population, agents=args.agents))
    if args.model_call_budget is not None:
        config = replace(
            config,
            llm=replace(config.llm, call_budget=args.model_call_budget),
        )
    if args.action_attempt_budget is not None:
        config = replace(
            config,
            simulation=replace(
                config.simulation,
                action_attempt_budget=args.action_attempt_budget,
            ),
        )
    if args.temperature is not None:
        config = replace(
            config,
            llm=replace(config.llm, temperature=args.temperature),
        )
        config.validate()
    ticks = args.ticks or config.simulation.max_ticks
    summary = asyncio.run(_simulate_episode(config, ticks, args.policy, args.output))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


RESEARCH_CONDITIONS: dict[str, dict[str, bool]] = {
    "full": {},
    # Special runner: N isolated one-agent worlds with the same total action/model
    # opportunity. The population maximum is the independent-search envelope.
    "independent-search": {"communication": False},
    "no-feedback": {"action_feedback": False, "immediate_replanning": False},
    "no-communication": {"communication": False},
    "no-social-awareness": {"social_awareness": False},
    "no-depot": {"shared_depot": False},
    "no-infrastructure-map": {"public_infrastructure_map": False},
    "global-landmarks": {"global_landmarks": True},
    "no-addressing": {"addressed_communication": False},
    "no-program-forking": {"program_forking": False},
    "no-explicit-culture": {
        "communication": False,
        "program_forking": False,
        "skill_library": False,
    },
    "no-skill-library": {"skill_library": False},
    "no-retrieval-diagnostics": {"retrieval_diagnostics": False},
    "legacy-transcript": {
        "experience_reuse": False,
        "request_tracking": False,
        "message_interrupts": True,
        "selective_replanning": False,
    },
    "bounded-context-only": {},
    "no-experience-reuse": {"experience_reuse": False},
    "no-request-tracking": {"request_tracking": False},
}

RESEARCH_LLM_CONDITIONS: dict[str, dict[str, Any]] = {
    "legacy-transcript": {
        "experience_attention": False,
        "retrieval_feedback_weight": 0.0,
        "retrieval_exploration_weight": 0.0,
    },
    "bounded-context-only": {
        "experience_attention": True,
        "retrieval_feedback_weight": 0.0,
        "retrieval_exploration_weight": 0.0,
    },
}


def _population_config(
    base: GameConfig,
    population_size: int,
    *,
    reference_population: int,
    world_scaling: str,
) -> GameConfig:
    if population_size < 1:
        raise ValueError("population sizes must be positive")
    world = base.world
    if world_scaling == "constant-area-per-agent":
        factor = np.sqrt(population_size / max(1, reference_population))
        world = replace(
            world,
            width=max(16, int(round(base.world.width * factor))),
            height=max(16, int(round(base.world.height * factor))),
        )
    config = replace(
        base,
        world=world,
        population=replace(base.population, agents=population_size),
    )
    config.validate()
    return config


def _condition_config(base: GameConfig, condition: str, seed: int) -> GameConfig:
    science = replace(
        base.science,
        enabled=True,
        **RESEARCH_CONDITIONS[condition],
    )
    llm = replace(base.llm, **RESEARCH_LLM_CONDITIONS.get(condition, {}))
    config = replace(
        base,
        llm=llm,
        science=science,
        simulation=replace(base.simulation, seed=seed),
    )
    config.validate()
    return config


def _member_budget(total: int | None, members: int, index: int) -> int | None:
    if total is None:
        return None
    if total < members:
        raise ValueError(
            "independent-search requires each isolated member to receive at least "
            "one unit of every configured global budget"
        )
    quotient, remainder = divmod(total, members)
    return quotient + int(index < remainder)


def _sum_mappings(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for record in records:
        for name, value in dict(record.get(key, {})).items():
            if isinstance(value, int):
                result[str(name)] = result.get(str(name), 0) + value
    return dict(sorted(result.items()))


async def _independent_search_episode(
    config: GameConfig,
    *,
    population_size: int,
    ticks: int,
    policy_name: str,
    output_dir: Path,
    seed: int,
    held_out_evaluation_seeds: list[int] | None = None,
    held_out_evaluation_checkpoints: list[int] | None = None,
) -> dict[str, Any]:
    """Run N isolated agents and return their predeclared maximum-performance envelope."""

    members: list[dict[str, Any]] = []
    # Draw the corresponding swarm once without advancing it. Each isolated member
    # receives the same position and decision phase as swarm agent i.
    matched_population = BioFoundrySimulation(config).population
    matched_positions = [
        [int(matched_population.x[index]), int(matched_population.y[index])]
        for index in range(population_size)
    ]
    suffix = ".jsonl.gz" if config.trace.compression == "gzip" else ".jsonl"
    iterator = tqdm(
        range(population_size),
        desc=f"Independent envelope N={population_size}, seed={seed}",
        unit="agent",
        leave=False,
    )
    for member in iterator:
        member_config = replace(
            config,
            population=replace(
                config.population,
                agents=1,
                macro_phase_offset=config.population.macro_phase_offset + member,
                initial_positions=[matched_positions[member]],
            ),
            llm=replace(
                config.llm,
                call_budget=_member_budget(config.llm.call_budget, population_size, member),
            ),
            simulation=replace(
                config.simulation,
                action_attempt_budget=_member_budget(
                    config.simulation.action_attempt_budget,
                    population_size,
                    member,
                ),
            ),
        )
        member_config.validate()
        log_path = output_dir / (
            f"{policy_name}-independent-search-n-{population_size}-seed-{seed}"
            f"-member-{member:03d}{suffix}"
        )
        result = await _simulate_episode(
            member_config,
            ticks,
            policy_name,
            str(log_path),
            held_out_evaluation_seeds=held_out_evaluation_seeds,
            held_out_evaluation_checkpoints=held_out_evaluation_checkpoints,
        )
        result["independent_member"] = member
        members.append(result)

    selected_index = max(
        range(len(members)),
        key=lambda index: (
            float(dict(members[index].get("discovery_frontier", {})).get("normalized_auc", 0.0)),
            float(dict(members[index].get("discovery_frontier", {})).get("final_performance", 0.0)),
            -index,
        ),
    )
    aggregate = copy.deepcopy(members[selected_index])
    endpoint_winners: dict[str, int] = {"discovery_frontier_auc": selected_index}

    performance_winner = max(
        range(len(members)),
        key=lambda index: (
            float(dict(members[index].get("research", {})).get("best_artifact_performance", 0.0)),
            -index,
        ),
    )
    endpoint_winners["artifact_performance"] = performance_winner
    if "research" in members[performance_winner]:
        aggregate["research"] = copy.deepcopy(members[performance_winner]["research"])
    for field, endpoint_name in (
        ("best_current_program_peak_performance", "current_program_peak_performance"),
        ("best_final_state_artifact_performance", "final_state_artifact_performance"),
    ):
        candidates = [
            (float(dict(item.get("research", {})).get(field, 0.0)), index)
            for index, item in enumerate(members)
        ]
        _, winner = max(candidates, key=lambda item: (item[0], -item[1]))
        endpoint_winners[endpoint_name] = winner
        aggregate.setdefault("research", {})[field] = copy.deepcopy(
            dict(members[winner].get("research", {})).get(field, 0.0)
        )

    ecosystem_candidates = [
        (
            float(
                dict(dict(item.get("ecosystem_assay", {})).get("intact", {})).get(
                    "resilience_auc", float("-inf")
                )
            ),
            index,
        )
        for index, item in enumerate(members)
        if dict(item.get("ecosystem_assay", {})).get("intact")
    ]
    if ecosystem_candidates:
        _, ecosystem_winner = max(ecosystem_candidates, key=lambda item: (item[0], -item[1]))
        endpoint_winners["ecosystem_resilience_auc"] = ecosystem_winner
        aggregate["ecosystem_assay"] = copy.deepcopy(members[ecosystem_winner]["ecosystem_assay"])

    held_out_candidates = [
        (
            float(
                dict(dict(item.get("held_out_generalization", {})).get("resilience_auc", {})).get(
                    "mean", float("-inf")
                )
            ),
            index,
        )
        for index, item in enumerate(members)
        if dict(item.get("held_out_generalization", {})).get("resilience_auc")
    ]
    if held_out_candidates:
        _, held_out_winner = max(held_out_candidates, key=lambda item: (item[0], -item[1]))
        endpoint_winners["held_out_resilience_auc"] = held_out_winner
        aggregate["held_out_generalization"] = copy.deepcopy(
            members[held_out_winner]["held_out_generalization"]
        )

    checkpoint_evaluations: list[dict[str, Any]] = []
    checkpoint_ticks = _normalize_held_out_checkpoints(held_out_evaluation_checkpoints, ticks=ticks)
    for checkpoint_tick in checkpoint_ticks:
        member_checkpoints: list[tuple[int, dict[str, Any]]] = []
        for member_index, item in enumerate(members):
            checkpoint = next(
                (
                    record
                    for record in item.get("held_out_checkpoints", [])
                    if int(record.get("tick", -1)) == checkpoint_tick
                ),
                None,
            )
            if isinstance(checkpoint, dict):
                member_checkpoints.append((member_index, checkpoint))
        candidates = [
            (
                float(
                    dict(
                        dict(checkpoint.get("held_out_generalization", {})).get(
                            "resilience_auc", {}
                        )
                    ).get("mean", float("-inf"))
                ),
                member_index,
                checkpoint,
            )
            for member_index, checkpoint in member_checkpoints
        ]
        if not candidates:
            continue
        _, winner, winning_checkpoint = max(candidates, key=lambda item: (item[0], -item[1]))
        aggregate_checkpoint = copy.deepcopy(winning_checkpoint)
        aggregate_checkpoint.update(
            {
                "independent_search": True,
                "selected_member": winner,
                "selection_rule": (
                    "maximum held-out resilience AUC across isolated members at this checkpoint"
                ),
                "model_calls": sum(
                    int(checkpoint.get("model_calls", 0)) for _, checkpoint in member_checkpoints
                ),
                "model_usage": _sum_mappings(
                    [checkpoint for _, checkpoint in member_checkpoints],
                    "model_usage",
                ),
                "member_held_out_resilience_auc": [
                    {
                        "member": member_index,
                        "value": dict(
                            dict(checkpoint.get("held_out_generalization", {})).get(
                                "resilience_auc", {}
                            )
                        ).get("mean"),
                    }
                    for member_index, checkpoint in member_checkpoints
                ],
            }
        )
        checkpoint_evaluations.append(aggregate_checkpoint)
        endpoint_winners[f"held_out_resilience_auc_at_tick_{checkpoint_tick}"] = winner
    aggregate.update(
        {
            "agents": population_size,
            "population_size": population_size,
            "independent_search": True,
            "independent_member_count": population_size,
            "position_summary_version": 2,
            "initial_positions": matched_positions,
            "final_positions": [
                list((item.get("final_positions") or [matched_positions[member_index]])[0])
                for member_index, item in enumerate(members)
            ],
            "macroturn_phases": [
                int(
                    (config.population.macro_phase_offset + member)
                    % config.population.macro_interval
                )
                for member in range(population_size)
            ],
            "matched_initial_positions": matched_positions,
            "selected_member": selected_index,
            "selection_rule": (
                "endpoint-wise maximum across N isolated searches; selected_member "
                "identifies the discovery-frontier-AUC winner"
            ),
            "endpoint_winners": endpoint_winners,
            "member_outputs": [str(item["output"]) for item in members],
            "member_endpoints": [
                {
                    "member": index,
                    "final_performance": dict(item.get("discovery_frontier", {})).get(
                        "final_performance", 0.0
                    ),
                    "frontier_auc": dict(item.get("discovery_frontier", {})).get(
                        "normalized_auc", 0.0
                    ),
                    "current_program_peak_performance": dict(item.get("research", {})).get(
                        "best_current_program_peak_performance", 0.0
                    ),
                    "final_state_artifact_performance": dict(item.get("research", {})).get(
                        "best_final_state_artifact_performance", 0.0
                    ),
                    "ecosystem_resilience_auc": dict(
                        dict(item.get("ecosystem_assay", {})).get("intact", {})
                    ).get("resilience_auc"),
                    "held_out_resilience_auc": dict(
                        dict(item.get("held_out_generalization", {})).get("resilience_auc", {})
                    ).get("mean"),
                }
                for index, item in enumerate(members)
            ],
            "model_calls": sum(int(item.get("model_calls", 0)) for item in members),
            "decision_opportunities": (
                sum(int(item["decision_opportunities"]) for item in members)
                if all(item.get("decision_opportunities") is not None for item in members)
                else None
            ),
            "model_errors": sum(int(item.get("model_errors", 0)) for item in members),
            "model_usage": _sum_mappings(members, "model_usage"),
            "action_counts": _sum_mappings(members, "action_counts"),
            "event_counts": _sum_mappings(members, "event_counts"),
            "artifacts": sum(int(item.get("artifacts", 0)) for item in members),
            "artifact_score": sum(float(item.get("artifact_score", 0.0)) for item in members),
            "mobility": {
                "successful_moves": sum(
                    int(dict(item.get("mobility", {})).get("successful_moves", 0))
                    for item in members
                ),
                "mean_distance_traveled": float(
                    np.mean(
                        [
                            float(dict(item.get("mobility", {})).get("mean_distance_traveled", 0.0))
                            for item in members
                        ]
                    )
                ),
                "moving_agent_fraction": float(
                    np.mean(
                        [
                            float(dict(item.get("mobility", {})).get("moving_agent_fraction", 0.0))
                            for item in members
                        ]
                    )
                ),
                "mean_distinct_cells_visited": float(
                    np.mean(
                        [
                            float(
                                dict(item.get("mobility", {})).get(
                                    "mean_distinct_cells_visited", 0.0
                                )
                            )
                            for item in members
                        ]
                    )
                ),
                # Isolated worlds have no joint spatial crowding statistic.
                "regional_crowding": None,
            },
            "action_budget": {
                "configured": config.simulation.action_attempt_budget,
                "used": sum(
                    int(dict(item.get("action_budget", {})).get("used", 0)) for item in members
                ),
                "blocked": sum(
                    int(dict(item.get("action_budget", {})).get("blocked", 0)) for item in members
                ),
                "exhausted_tick": None,
                "member_exhausted_ticks": [
                    dict(item.get("action_budget", {})).get("exhausted_tick") for item in members
                ],
            },
        }
    )
    if checkpoint_ticks:
        aggregate["held_out_checkpoints"] = checkpoint_evaluations
    aggregate.pop("independent_member", None)
    return aggregate


async def _run_research_study(args: argparse.Namespace) -> list[dict[str, Any]]:
    base = _load(args.config)
    held_out_evaluation_seeds = list(getattr(args, "held_out_evaluation_seeds", None) or [])
    held_out_evaluation_checkpoints = _normalize_held_out_checkpoints(
        getattr(args, "held_out_evaluation_checkpoints", None),
        ticks=args.ticks,
    )
    if held_out_evaluation_checkpoints and not held_out_evaluation_seeds:
        raise ValueError("--held-out-evaluation-checkpoints requires --held-out-evaluation-seeds")
    if held_out_evaluation_checkpoints and not base.evaluation.enabled:
        raise ValueError(
            "--held-out-evaluation-checkpoints requires evaluation.enabled in the experiment config"
        )
    decision_schedule = getattr(args, "decision_schedule", "fixed")
    base = replace(
        base,
        science=replace(base.science, decision_schedule=decision_schedule),
    )
    if args.agents is not None:
        base = replace(base, population=replace(base.population, agents=args.agents))
    if args.macro_interval is not None:
        base = replace(
            base,
            population=replace(base.population, macro_interval=args.macro_interval),
        )
    if args.model_call_budget is not None:
        base = replace(
            base,
            llm=replace(base.llm, call_budget=args.model_call_budget),
        )
    if args.action_attempt_budget is not None:
        base = replace(
            base,
            simulation=replace(
                base.simulation,
                action_attempt_budget=args.action_attempt_budget,
            ),
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    population_sizes = list(getattr(args, "population_sizes", None) or [base.population.agents])
    reference_population = base.population.agents
    world_scaling = getattr(args, "world_scaling", "fixed")
    manifest = {
        "experiment": "open_invention_habitat_resilience_v4",
        "biofoundry_version": __version__,
        "engine_revision": ENGINE_REVISION,
        "config": base.as_dict(),
        "reproducibility": _reproducibility_metadata(base),
        "policy": args.policy,
        "conditions": args.conditions,
        "condition_overrides": {
            name: {
                "science": RESEARCH_CONDITIONS[name],
                "llm": RESEARCH_LLM_CONDITIONS.get(name, {}),
            }
            for name in args.conditions
        },
        "seeds": args.seeds,
        "population_sizes": population_sizes,
        "world_scaling": world_scaling,
        "decision_schedule": decision_schedule,
        "control_definitions": {
            "independent-search": (
                "endpoint-wise maximum among N isolated "
                "one-agent worlds initialized at the corresponding swarm agents' "
                "positions and macroturn phases; total configured model-call and "
                "action budgets are partitioned across members"
            ),
            "no-communication": (
                "removes public messaging, publishing, teaching, task claims, trade, "
                "and publication-dependent design composition from the advertised "
                "and executable model action contract"
            ),
            "no-explicit-culture": (
                "no-communication plus program forking and measured skill-library "
                "inheritance disabled; physical artifact stigmergy remains"
            ),
        },
        "ticks": args.ticks,
        "model_call_budget": base.llm.call_budget,
        "action_attempt_budget": base.simulation.action_attempt_budget,
        "held_out_evaluation_seeds": held_out_evaluation_seeds,
        "primary_endpoints": [
            "ecosystem_assay.intact.resilience_auc",
            "held_out_generalization.resilience_auc.mean",
            "research.outcome_success",
            "research.best_artifact_performance",
            "research.best_current_program_peak_performance",
            "research.best_final_state_artifact_performance",
            "research.portfolio_resilience",
            "research.best_behavioral_novelty",
            "discovery_frontier.normalized_auc",
        ],
        "mechanism_endpoints": [
            "technology_ecology.collaborative_artifact_fraction",
            "technology_ecology.cross_agent_program_fork_fraction",
            "technology_ecology.causal_closure_gain_fraction",
            "ecosystem_assay.mean_absolute_ecological_coupling",
            "ecosystem_assay.reciprocal_support_pair_count",
            "ecosystem_assay.maximum_knockout_effect",
            "research.emergent_success",
            "research.self_organized_portfolio",
            "swarm_metrics.causal_composition_success",
            "swarm_metrics.normalized_specialization",
            "swarm_metrics.program_forks",
            "swarm_metrics.maximum_program_lineage_depth",
            "swarm_metrics.maximum_verified_program_adoption",
            "swarm_metrics.message_reply_fraction",
            "swarm_metrics.request_fulfillment_fraction",
            "swarm_metrics.retrieval_citation_rate",
        ],
    }
    if held_out_evaluation_checkpoints:
        manifest["held_out_evaluation_checkpoints"] = held_out_evaluation_checkpoints
        manifest["checkpoint_evaluation"] = {
            "discovery_state_frozen": True,
            "agent_actions_per_schedule": 0,
            "selection_rule_for_independent_search": (
                "endpoint-wise maximum held-out resilience AUC across isolated "
                "members at each checkpoint"
            ),
        }
    (output_dir / "study-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    combinations = [
        (condition, population_size, seed)
        for condition in args.conditions
        for population_size in population_sizes
        for seed in args.seeds
    ]
    results: list[dict[str, Any]] = []
    for condition, population_size, seed in tqdm(
        combinations, desc="Research conditions", unit="run"
    ):
        population_config = _population_config(
            base,
            population_size,
            reference_population=reference_population,
            world_scaling=world_scaling,
        )
        config = _condition_config(population_config, condition, seed)
        trace_suffix = ".jsonl.gz" if config.trace.compression == "gzip" else ".jsonl"
        size_token = f"-n-{population_size}" if getattr(args, "population_sizes", None) else ""
        log_path = output_dir / (f"{args.policy}-{condition}{size_token}-seed-{seed}{trace_suffix}")
        if condition == "independent-search":
            summary = await _independent_search_episode(
                config,
                population_size=population_size,
                ticks=args.ticks,
                policy_name=args.policy,
                output_dir=output_dir,
                seed=seed,
                held_out_evaluation_seeds=getattr(args, "held_out_evaluation_seeds", None),
                held_out_evaluation_checkpoints=getattr(
                    args, "held_out_evaluation_checkpoints", None
                ),
            )
        else:
            summary = await _simulate_episode(
                config,
                args.ticks,
                args.policy,
                str(log_path),
                held_out_evaluation_seeds=getattr(args, "held_out_evaluation_seeds", None),
                held_out_evaluation_checkpoints=getattr(
                    args, "held_out_evaluation_checkpoints", None
                ),
            )
        summary["condition"] = condition
        summary["seed"] = seed
        summary["population_size"] = population_size
        summary["world_scaling"] = world_scaling
        results.append(summary)
    summary_path = output_dir / "study-summary.json"
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return results


def research_study(args: argparse.Namespace) -> int:
    results = asyncio.run(_run_research_study(args))
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def analyze_research_study(args: argparse.Namespace) -> int:
    analysis = analyze_study(
        args.summary,
        args.output_dir,
        resamples=args.bootstrap_resamples,
        source_labels=args.labels,
    )
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0


def analyze_movement(args: argparse.Namespace) -> int:
    """Analyze seed-level movement and spatial self-organization."""

    from .mobility_analysis import analyze_mobility

    analysis = analyze_mobility(
        args.summary,
        args.output_dir,
        resamples=args.bootstrap_resamples,
        proximity_radius=args.artifact_proximity_radius,
        encounter_radius=args.encounter_radius,
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).resolve()),
                "episodes": len(analysis["episodes"]),
                "unit_of_analysis": analysis["unit_of_analysis"],
                "figures": analysis["figures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def technology_dossiers(args: argparse.Namespace) -> int:
    """Catalog trace-derived technologies and optionally generate concept images."""

    from .technology_visualization import build_technology_dossiers

    manifest = build_technology_dossiers(
        args.path,
        args.output_dir,
        styles=args.styles,
        generate=args.generate,
        model=args.model,
        size=args.size,
        quality=args.quality,
        top_k=args.top_k,
        include_retired=not args.active_only,
        minimum_peak_performance=args.minimum_peak_performance,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).resolve()),
                "technology_count": manifest["technology_count"],
                "generated": manifest["generated"],
                "model": manifest["model"],
                "manifest": str(
                    (Path(args.output_dir) / "technology-render-manifest.json").resolve()
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def technology_atlas(args: argparse.Namespace) -> int:
    """Build a semantic atlas and distinct, illustrated technology shortlist."""

    from .technology_atlas import build_technology_atlas

    audit = build_technology_atlas(
        args.summary,
        args.output_dir,
        conditions=args.conditions,
        include_independent_members=not args.selected_independent_only,
        featured_count=args.featured_count,
        maximum_cosine_similarity=args.maximum_cosine_similarity,
        maximum_per_cluster=args.maximum_per_cluster,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_dimensions=args.embedding_dimensions,
        embedding_batch_size=args.embedding_batch_size,
        generate_images=args.generate,
        image_model=args.image_model,
        image_size=args.image_size,
        image_quality=args.image_quality,
        image_workers=args.image_workers,
        image_styles=args.image_styles,
        overwrite_embeddings=args.overwrite_embeddings,
        overwrite_images=args.overwrite_images,
        random_state=args.random_state,
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).resolve()),
                "technology_count": audit["collection"]["technology_count"],
                "featured_count": audit["selection"]["featured_count"],
                "generated_image_count": audit["generated_image_count"],
                "explorer": audit["explorer"],
                "audit": str((Path(args.output_dir) / "technology-atlas-audit.json").resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def trace_report(args: argparse.Namespace) -> int:
    report = analyze_trace(args.path)
    if args.figures_dir:
        report["figures"] = render_trace_lineage_figure(args.path, args.figures_dir)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


def ecosystem_report(args: argparse.Namespace) -> int:
    """Replay a society and run the frozen, agent-free technology assay."""

    config_data, action_records = load_action_trace(args.path)
    simulation = replay_intervention(
        config_data,
        action_records,
        description="Reconstruct technological society",
    )
    assay = evaluate_technological_ecosystem(
        simulation,
        horizon=args.horizon,
    )
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report = {
        "path": str(Path(args.path).resolve()),
        "technology_ecology": technology_ecology_summary(simulation),
        "ecosystem_assay": assay,
        "mobility_history": snapshot_dynamics(args.path),
        "trace_diagnostics": analyze_trace(args.path),
    }
    if args.evaluation_seeds:
        report["held_out_generalization"] = evaluate_technological_ecosystem_generalization(
            simulation,
            args.evaluation_seeds,
            horizon=args.horizon,
        )
    report["figures"] = render_technology_ecology_figures(
        simulation,
        assay,
        report["mobility_history"],
        destination,
        stem=Path(args.path).name.split(".jsonl", 1)[0],
    )
    output = destination / "ecosystem-report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "artifacts": simulation.artifacts.count,
                "intact_resilience_auc": assay["intact"]["resilience_auc"],
                "mean_absolute_ecological_coupling": assay["mean_absolute_ecological_coupling"],
                "reciprocal_support_pair_count": assay["reciprocal_support_pair_count"],
                "figures": report["figures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def counterfactual_replay(args: argparse.Namespace) -> int:
    config_data, action_records = load_action_trace(args.path)
    factual_simulation = replay_intervention(
        config_data,
        action_records,
        description="Factual replay",
    )
    factual = outcome_summary(factual_simulation)
    if args.all_claimed:
        targets = claimed_contributors(factual_simulation)
    else:
        targets = list(args.remove_agent or [])
    if not targets:
        raise ValueError("no intervention targets were selected or found")
    interventions: list[dict[str, Any]] = []
    for agent_id in tqdm(targets, desc="Agent removals", unit="agent"):
        counterfactual_simulation = replay_intervention(
            config_data,
            action_records,
            {agent_id},
            description=f"Remove {agent_id}",
        )
        counterfactual = outcome_summary(counterfactual_simulation)
        interventions.append(
            {
                "removed_agent": agent_id,
                "comparison": compare_outcomes(factual, counterfactual),
                "counterfactual": counterfactual,
            }
        )
    report = {
        "path": str(Path(args.path).resolve()),
        "intervention": "replace the selected agent's recorded actions with WAIT",
        "factual": factual,
        "interventions": interventions,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


def benchmark(args: argparse.Namespace) -> int:
    base = _load(args.config)
    counts = args.agents or [16, 64, 256, 1024, 4096]
    results: list[dict[str, Any]] = []
    for count in tqdm(counts, desc="Population sizes", unit="size"):
        config = replace(base, population=replace(base.population, agents=count))
        simulation = BioFoundrySimulation(config)
        policy = ScalableForagerPolicy()
        started = time.perf_counter()
        event_count = 0
        for _ in tqdm(
            range(args.ticks),
            desc=f"N={count}",
            unit="tick",
            leave=False,
        ):
            result = simulation.step(policy.actions(simulation))
            event_count += len(result.events)
        elapsed = time.perf_counter() - started
        results.append(
            {
                "agents": count,
                "ticks": args.ticks,
                "seconds": elapsed,
                "ticks_per_second": args.ticks / max(elapsed, 1e-9),
                "agent_steps_per_second": count * args.ticks / max(elapsed, 1e-9),
                "events": event_count,
                "state_digest": simulation.state_digest(),
            }
        )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def replay(args: argparse.Namespace) -> int:
    path = Path(args.path)
    types: dict[str, int] = {}
    final_digest = None
    snapshots = 0
    deterministic_snapshots = 0
    simulation: BioFoundrySimulation | None = None
    recorded_policy = ""
    deterministic_skip_reason: str | None = None
    for record in tqdm(read_records(path), desc="Reading replay", unit="record"):
        record_type = str(record.get("type", "unknown"))
        types[record_type] = types.get(record_type, 0) + 1
        if record_type == "header":
            metadata = record.get("metadata", {})
            recorded_policy = str(metadata.get("policy", ""))
            config_data = metadata.get("config")
            recorded_revision = metadata.get("engine_revision")
            if recorded_revision is None:
                deterministic_skip_reason = (
                    "trace predates explicit engine revisions; snapshot integrity "
                    "can be verified, but current-code action replay is not valid"
                )
            elif recorded_revision != ENGINE_REVISION:
                deterministic_skip_reason = (
                    f"trace engine revision {recorded_revision} differs from current "
                    f"revision {ENGINE_REVISION}"
                )
            elif isinstance(config_data, dict):
                replay_config = config_from_dict(config_data)
                replay_scenario = load_scenario(replay_config.world.scenario_package)
                recorded_scenario = metadata.get("scenario", {})
                if replay_scenario is not None and isinstance(recorded_scenario, dict):
                    recorded_hash = recorded_scenario.get("package_hash")
                    if recorded_hash and recorded_hash != replay_scenario.package_hash:
                        raise ValueError(
                            "scenario package differs from the recorded trace: "
                            f"trace={recorded_hash}, current={replay_scenario.package_hash}"
                        )
                simulation = BioFoundrySimulation(replay_config)
            else:
                deterministic_skip_reason = "trace header has no complete configuration"
        elif record_type == "model_trace" and simulation is not None:
            apply_recorded_model_trace(simulation, record)
        elif record_type == "actions" and simulation is not None:
            if int(record.get("tick", -1)) != simulation.tick:
                raise ValueError(
                    f"action trace tick {record.get('tick')} does not match {simulation.tick}"
                )
            acknowledge_recorded_macroturns(
                simulation,
                record,
                policy=recorded_policy,
            )
            simulation.step(record.get("actions", {}))
        elif record_type == "snapshot":
            snapshots += 1
            observed = digest_snapshot(record["snapshot"])
            if observed != record.get("digest"):
                raise ValueError(f"snapshot {snapshots} has a digest mismatch")
            final_digest = str(record.get("state_digest", observed))
            if simulation is not None and record.get("state_digest") is not None:
                replayed = simulation.state_digest()
                if replayed != record["state_digest"]:
                    raise ValueError(
                        f"snapshot {snapshots} deterministic-state mismatch: "
                        f"expected {record['state_digest']}, got {replayed}"
                    )
                deterministic_snapshots += 1
    print(
        json.dumps(
            {
                "path": str(path.resolve()),
                "records": sum(types.values()),
                "types": types,
                "snapshots_verified": snapshots,
                "deterministic_snapshots_verified": deterministic_snapshots,
                "action_trace_replayed": simulation is not None and types.get("actions", 0) > 0,
                "deterministic_replay_skip_reason": deterministic_skip_reason,
                "final_digest": final_digest,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app

    config = _load(args.config)
    app = create_app(config, record_path=args.record)
    uvicorn.run(app, host=config.server.host, port=config.server.port, log_level="info")
    return 0


def playback(args: argparse.Namespace) -> int:
    """Serve a completed trace to the observatory without making model calls."""

    import uvicorn

    from .playback import create_playback_app

    app = create_playback_app(
        args.path,
        ticks_per_second=args.ticks_per_second,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def world_validate(args: argparse.Namespace) -> int:
    descriptor = validate_scenario(args.path)
    print(
        json.dumps(
            {
                "valid": True,
                "scenario": {
                    key: descriptor[key]
                    for key in ("id", "name", "version", "package_hash", "format_version")
                },
                "catalog_limits": descriptor["catalog_limits"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def world_describe(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.path)
    if scenario is None:  # pragma: no cover - argparse always supplies a path
        raise ValueError("a scenario path is required")
    print(scenario_json(scenario))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="biofoundry", description="BioFoundry World simulation and research tools"
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    world_parser = subparsers.add_parser(
        "world", help="validate or describe a declarative world package"
    )
    world_subparsers = world_parser.add_subparsers(dest="world_command", required=True)
    world_validate_parser = world_subparsers.add_parser(
        "validate", help="validate package files and cross-references"
    )
    world_validate_parser.add_argument("path")
    world_validate_parser.set_defaults(function=world_validate)
    world_describe_parser = world_subparsers.add_parser(
        "describe", help="print the resolved machine-readable world contract"
    )
    world_describe_parser.add_argument("path")
    world_describe_parser.set_defaults(function=world_describe)

    doctor_parser = subparsers.add_parser("doctor", help="validate runtime and configuration")
    doctor_parser.add_argument("--config", default="configs/demo.yaml")
    doctor_parser.set_defaults(function=doctor)

    simulate_parser = subparsers.add_parser("simulate", help="run a bounded headless episode")
    simulate_parser.add_argument("--config", default="configs/demo.yaml")
    simulate_parser.add_argument("--ticks", type=int)
    simulate_parser.add_argument("--agents", type=int)
    simulate_parser.add_argument("--seed", type=int)
    simulate_parser.add_argument("--model-call-budget", type=int)
    simulate_parser.add_argument(
        "--action-attempt-budget",
        type=int,
        help="cap all admitted non-WAIT action attempts, including failed attempts",
    )
    simulate_parser.add_argument(
        "--temperature",
        type=float,
        help="override model sampling temperature for an explicitly labeled run",
    )
    simulate_parser.add_argument(
        "--policy",
        choices=["scripted", "research-oracle", "scalable", "llm"],
        default="scripted",
    )
    simulate_parser.add_argument("--output", default="runs/demo.jsonl")
    simulate_parser.set_defaults(function=simulate)

    study_parser = subparsers.add_parser(
        "research-study", help="run preregistered collective-science conditions"
    )
    study_parser.add_argument("--config", default="configs/demo.yaml")
    study_parser.add_argument(
        "--policy",
        choices=["research-oracle", "scripted", "scalable", "llm"],
        default="research-oracle",
    )
    study_parser.add_argument(
        "--conditions",
        nargs="+",
        choices=sorted(RESEARCH_CONDITIONS),
        default=["full"],
    )
    study_parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18, 19, 20, 21])
    study_parser.add_argument("--ticks", type=int, default=256)
    population_group = study_parser.add_mutually_exclusive_group()
    population_group.add_argument("--agents", type=int)
    population_group.add_argument(
        "--population-sizes",
        type=int,
        nargs="+",
        help="run a preregistered population-size sweep, for example 1 4 16 50",
    )
    study_parser.add_argument(
        "--world-scaling",
        choices=["fixed", "constant-area-per-agent"],
        default="fixed",
        help="hold the world fixed or approximately preserve area per agent",
    )
    study_parser.add_argument("--macro-interval", type=int)
    study_parser.add_argument(
        "--decision-schedule",
        choices=["fixed", "event-driven"],
        default="fixed",
        help=(
            "fixed gives every agent treatment-invariant decision opportunities; "
            "event-driven also replans after salient events"
        ),
    )
    study_parser.add_argument(
        "--held-out-evaluation-seeds",
        type=int,
        nargs="+",
        help=(
            "after discovery, freeze agents and evaluate every final society under "
            "these unseen disturbance schedules"
        ),
    )
    study_parser.add_argument(
        "--held-out-evaluation-checkpoints",
        type=int,
        nargs="+",
        help=(
            "freeze and evaluate society copies at these discovery ticks using "
            "the held-out disturbance seeds; does not advance the live society"
        ),
    )
    study_parser.add_argument("--model-call-budget", type=int)
    study_parser.add_argument(
        "--action-attempt-budget",
        type=int,
        help="use the same global non-WAIT action-attempt budget in every condition",
    )
    study_parser.add_argument("--output-dir", default="runs/research-study")
    study_parser.set_defaults(function=research_study)

    analysis_parser = subparsers.add_parser(
        "analyze-study", help="analyze seed-level study results and render figures"
    )
    analysis_parser.add_argument("summary", nargs="+")
    analysis_parser.add_argument(
        "--labels",
        nargs="+",
        help="explicit condition label for each summary, in the same order",
    )
    analysis_parser.add_argument("--output-dir")
    analysis_parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    analysis_parser.set_defaults(function=analyze_research_study)

    mobility_parser = subparsers.add_parser(
        "analyze-mobility",
        help="analyze seed-level movement and spatial self-organization",
    )
    mobility_parser.add_argument("summary", nargs="+")
    mobility_parser.add_argument("--output-dir", required=True)
    mobility_parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    mobility_parser.add_argument(
        "--artifact-proximity-radius",
        type=float,
        default=3.0,
        help="agent-to-artifact proximity radius in world cells",
    )
    mobility_parser.add_argument(
        "--encounter-radius",
        type=float,
        default=2.0,
        help="agent-to-agent encounter radius in world cells",
    )
    mobility_parser.set_defaults(function=analyze_movement)

    dossier_parser = subparsers.add_parser(
        "technology-dossiers",
        help="catalog technologies and optionally render trace-grounded concept images",
    )
    dossier_parser.add_argument("path", help="recorded JSONL or JSONL.GZ trace")
    dossier_parser.add_argument("--output-dir", required=True)
    dossier_parser.add_argument(
        "--styles",
        nargs="+",
        choices=["engineering", "photorealistic", "overview", "mechanism"],
        default=["engineering", "photorealistic"],
    )
    dossier_parser.add_argument("--model", default="gpt-image-2")
    dossier_parser.add_argument("--size", default="1536x1024")
    dossier_parser.add_argument("--quality", default="high")
    dossier_parser.add_argument(
        "--top-k",
        type=int,
        help="rank by recorded lifetime peak performance and keep only K technologies",
    )
    dossier_parser.add_argument(
        "--minimum-peak-performance",
        type=float,
        default=0.0,
    )
    dossier_parser.add_argument(
        "--active-only",
        action="store_true",
        help="exclude artifacts marked retired at the final snapshot",
    )
    dossier_parser.add_argument(
        "--generate",
        action="store_true",
        help="call the OpenAI Image API; without this flag only prompts and sheets are written",
    )
    dossier_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="regenerate image files that already exist",
    )
    dossier_parser.set_defaults(function=technology_dossiers)

    atlas_parser = subparsers.add_parser(
        "technology-atlas",
        help="rank, embed, and explore technologies across completed studies",
    )
    atlas_parser.add_argument(
        "summary",
        nargs="+",
        help="one or more canonical study-summary.json files",
    )
    atlas_parser.add_argument("--output-dir", required=True)
    atlas_parser.add_argument(
        "--conditions",
        nargs="+",
        choices=[
            "full",
            "no-explicit-culture",
            "no-communication",
            "independent-search",
        ],
        default=[
            "full",
            "no-explicit-culture",
            "no-communication",
            "independent-search",
        ],
    )
    atlas_parser.add_argument("--featured-count", type=int, default=16)
    atlas_parser.add_argument("--maximum-cosine-similarity", type=float, default=0.82)
    atlas_parser.add_argument("--maximum-per-cluster", type=int, default=4)
    atlas_parser.add_argument(
        "--embedding-provider",
        choices=["embeddinggemma", "openai", "tfidf"],
        default="embeddinggemma",
    )
    atlas_parser.add_argument("--embedding-model", default="google/embeddinggemma-300m")
    atlas_parser.add_argument("--embedding-dimensions", type=int, default=768)
    atlas_parser.add_argument("--embedding-batch-size", type=int, default=16)
    atlas_parser.add_argument("--image-model", default="gpt-image-2")
    atlas_parser.add_argument("--image-size", default="1536x1024")
    atlas_parser.add_argument("--image-quality", default="high")
    atlas_parser.add_argument(
        "--image-styles",
        nargs="+",
        choices=["engineering", "photorealistic", "overview", "mechanism"],
        default=["engineering", "photorealistic"],
        help="image concepts to generate for each featured technology",
    )
    atlas_parser.add_argument(
        "--image-workers",
        type=int,
        default=4,
        help="bounded concurrent Image API requests (default: 4)",
    )
    atlas_parser.add_argument("--random-state", type=int, default=42)
    atlas_parser.add_argument(
        "--selected-independent-only",
        action="store_true",
        help="include only each independent-search cell's selected member trace",
    )
    atlas_parser.add_argument(
        "--generate",
        action="store_true",
        help="generate the selected GPT Image concepts for each featured technology",
    )
    atlas_parser.add_argument("--overwrite-embeddings", action="store_true")
    atlas_parser.add_argument("--overwrite-images", action="store_true")
    atlas_parser.set_defaults(function=technology_atlas)

    trace_parser = subparsers.add_parser(
        "trace-report", help="summarize behavioral and causal diagnostics from one JSONL run"
    )
    trace_parser.add_argument("path")
    trace_parser.add_argument("--output")
    trace_parser.add_argument("--figures-dir")
    trace_parser.set_defaults(function=trace_report)

    ecosystem_parser = subparsers.add_parser(
        "ecosystem-report",
        help="run agent-free artifact knockouts and render technology-ecology figures",
    )
    ecosystem_parser.add_argument("path")
    ecosystem_parser.add_argument("--output-dir", required=True)
    ecosystem_parser.add_argument(
        "--horizon",
        type=int,
        help="override the configured agent-free disturbance-cycle horizon",
    )
    ecosystem_parser.add_argument(
        "--evaluation-seeds",
        type=int,
        nargs="+",
        help=(
            "predeclared held-out disturbance seeds; evaluates the frozen final "
            "society without additional agent actions"
        ),
    )
    ecosystem_parser.set_defaults(function=ecosystem_report)

    counterfactual_parser = subparsers.add_parser(
        "counterfactual-replay",
        help="replace selected agents with WAIT in a deterministic action trace",
    )
    counterfactual_parser.add_argument("path")
    target_group = counterfactual_parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--remove-agent", nargs="+")
    target_group.add_argument("--all-claimed", action="store_true")
    counterfactual_parser.add_argument("--output")
    counterfactual_parser.set_defaults(function=counterfactual_replay)

    benchmark_parser = subparsers.add_parser("benchmark", help="measure headless scaling")
    benchmark_parser.add_argument("--config", default="configs/demo.yaml")
    benchmark_parser.add_argument("--ticks", type=int, default=100)
    benchmark_parser.add_argument("--agents", type=int, nargs="+")
    benchmark_parser.set_defaults(function=benchmark)

    replay_parser = subparsers.add_parser("replay", help="verify a JSONL replay")
    replay_parser.add_argument("path")
    replay_parser.set_defaults(function=replay)

    serve_parser = subparsers.add_parser("serve", help="run live simulation and WebSocket API")
    serve_parser.add_argument("--config", default="configs/demo.yaml")
    serve_parser.add_argument("--record")
    serve_parser.set_defaults(function=serve)

    playback_parser = subparsers.add_parser(
        "playback",
        help="serve an exact completed trace to the web observatory without model calls",
    )
    playback_parser.add_argument("path", help="recorded JSONL or JSONL.GZ trace")
    playback_parser.add_argument("--host", default="127.0.0.1")
    playback_parser.add_argument("--port", type=int, default=8765)
    playback_parser.add_argument(
        "--ticks-per-second",
        type=float,
        default=8.0,
        help="base playback rate before the observatory speed multiplier",
    )
    playback_parser.set_defaults(function=playback)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.function(args))

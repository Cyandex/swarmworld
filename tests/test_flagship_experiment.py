import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

import biofoundry.cli as cli
from biofoundry.events import read_records
from biofoundry.simulation import BioFoundrySimulation

ROOT = Path(__file__).resolve().parents[1]


def test_episode_summary_distinguishes_initial_and_final_positions(tmp_path) -> None:
    config = cli._load(str(ROOT / "configs" / "demo.yaml"))
    output = tmp_path / "episode.jsonl"

    summary = asyncio.run(
        cli._simulate_episode(config, 3, "scripted", str(output))
    )
    snapshots = [
        record["snapshot"]
        for record in read_records(output)
        if record.get("type") == "snapshot"
    ]
    initial_agents = snapshots[0]["agents"]
    final_agents = snapshots[-1]["agents"]

    assert summary["position_summary_version"] == 2
    assert summary["initial_positions"] == [
        list(position)
        for position in zip(initial_agents["x"], initial_agents["y"], strict=True)
    ]
    assert summary["final_positions"] == [
        list(position)
        for position in zip(final_agents["x"], final_agents["y"], strict=True)
    ]


def test_episode_evaluates_frozen_held_out_checkpoints(tmp_path) -> None:
    base = cli._load(str(ROOT / "configs" / "demo.yaml"))
    config = replace(
        base,
        evaluation=replace(base.evaluation, enabled=True, horizon=2),
    )
    output = tmp_path / "checkpoint-episode.jsonl"
    baseline_output = tmp_path / "baseline-episode.jsonl"

    baseline = asyncio.run(
        cli._simulate_episode(config, 3, "scripted", str(baseline_output))
    )

    summary = asyncio.run(
        cli._simulate_episode(
            config,
            3,
            "scripted",
            str(output),
            held_out_evaluation_seeds=[901, 902],
            held_out_evaluation_checkpoints=[1, 3],
        )
    )

    checkpoints = summary["held_out_checkpoints"]
    assert [item["tick"] for item in checkpoints] == [1, 3]
    assert all(item["discovery_state_frozen"] for item in checkpoints)
    assert all(item["agent_actions_per_schedule"] == 0 for item in checkpoints)
    assert summary["held_out_generalization"] == checkpoints[-1][
        "held_out_generalization"
    ]
    assert summary["state_digest"] == baseline["state_digest"]
    assert summary["final_positions"] == baseline["final_positions"]
    assert summary["action_counts"] == baseline["action_counts"]
    assert summary["research"] == baseline["research"]
    snapshot_ticks = [
        int(record["snapshot"]["tick"])
        for record in read_records(output)
        if record.get("type") == "snapshot"
    ]
    assert 1 in snapshot_ticks
    assert 3 in snapshot_ticks


def test_checkpoint_ticks_must_fall_within_discovery_horizon() -> None:
    with pytest.raises(ValueError, match="no greater than"):
        cli._normalize_held_out_checkpoints([0, 9], ticks=8)


def _summary(
    config, output: str, performance: float, frontier_auc: float | None = None
) -> dict[str, object]:
    return {
        "agents": config.population.agents,
        "output": output,
        "model_calls": 3,
        "decision_opportunities": config.population.agents * 2,
        "model_errors": 0,
        "model_usage": {"input_tokens": 10},
        "action_counts": {"MOVE": 2},
        "event_counts": {"agent_moved": 2},
        "discovery_frontier": {
            "final_performance": performance,
            "normalized_auc": performance / 2 if frontier_auc is None else frontier_auc,
        },
        "research": {"best_artifact_performance": performance},
        "ecosystem_assay": {"intact": {"resilience_auc": performance}},
    }


def test_flagship_runner_builds_paired_swarm_and_independent_envelope(
    tmp_path, monkeypatch
) -> None:
    member = 0
    isolated_configs = []

    async def fake_simulate(
        config,
        ticks,
        policy_name,
        output,
        *,
        held_out_evaluation_seeds=None,
        held_out_evaluation_checkpoints=None,
    ):
        nonlocal member
        assert held_out_evaluation_seeds is None
        assert held_out_evaluation_checkpoints is None
        assert ticks == 8
        assert policy_name == "scripted"
        if config.population.agents == 1:
            isolated_configs.append(config)
            performance = (0.4, 0.3)[member]
            frontier_auc = (0.1, 0.2)[member]
            member += 1
        else:
            performance = 0.4
            frontier_auc = None
        return _summary(config, output, performance, frontier_auc)

    monkeypatch.setattr(cli, "_simulate_episode", fake_simulate)
    args = argparse.Namespace(
        config=str(ROOT / "configs" / "demo.yaml"),
        agents=None,
        population_sizes=[2],
        world_scaling="fixed",
        macro_interval=None,
        model_call_budget=None,
        action_attempt_budget=None,
        output_dir=str(tmp_path),
        conditions=["full", "independent-search"],
        seeds=[3001],
        ticks=8,
        policy="scripted",
    )

    results = asyncio.run(cli._run_research_study(args))

    assert len(results) == 2
    independent = next(
        record for record in results if record["condition"] == "independent-search"
    )
    assert independent["selected_member"] == 1
    assert independent["endpoint_winners"]["artifact_performance"] == 0
    assert independent["research"]["best_artifact_performance"] == 0.4
    assert independent["discovery_frontier"]["normalized_auc"] == 0.2
    assert independent["independent_member_count"] == 2
    assert independent["model_calls"] == 6
    assert independent["decision_opportunities"] == 4
    assert independent["action_counts"]["MOVE"] == 4
    assert [config.population.initial_positions[0] for config in isolated_configs] == (
        independent["matched_initial_positions"]
    )
    assert [config.population.macro_phase_offset for config in isolated_configs] == [
        0,
        1,
    ]
    manifest = json.loads((tmp_path / "study-manifest.json").read_text())
    assert manifest["population_sizes"] == [2]
    assert "independent-search" in manifest["control_definitions"]
    assert manifest["decision_schedule"] == "fixed"
    assert "held_out_evaluation_checkpoints" not in manifest


def test_independent_search_selects_held_out_winner_at_each_checkpoint(
    tmp_path, monkeypatch
) -> None:
    base = cli._load(str(ROOT / "configs" / "demo.yaml"))
    config = replace(base, population=replace(base.population, agents=2))
    member = 0

    async def fake_simulate(
        member_config,
        ticks,
        policy_name,
        output,
        *,
        held_out_evaluation_seeds=None,
        held_out_evaluation_checkpoints=None,
    ):
        nonlocal member
        assert held_out_evaluation_seeds == [901]
        assert held_out_evaluation_checkpoints == [4, 8]
        values = ((0.1, 0.4), (0.2, 0.3))[member]
        result = _summary(member_config, output, values[-1])
        result["held_out_generalization"] = {
            "resilience_auc": {"mean": values[-1]}
        }
        result["held_out_checkpoints"] = [
            {
                "tick": tick,
                "model_calls": tick,
                "model_usage": {"total_tokens": tick * 10},
                "held_out_generalization": {
                    "resilience_auc": {"mean": value}
                },
            }
            for tick, value in zip((4, 8), values, strict=True)
        ]
        result["initial_positions"] = [member_config.population.initial_positions[0]]
        result["final_positions"] = [member_config.population.initial_positions[0]]
        result["macroturn_phases"] = [member_config.population.macro_phase_offset]
        result["mobility"] = {}
        result["action_budget"] = {}
        member += 1
        return result

    monkeypatch.setattr(cli, "_simulate_episode", fake_simulate)
    result = asyncio.run(
        cli._independent_search_episode(
            config,
            population_size=2,
            ticks=8,
            policy_name="scripted",
            output_dir=tmp_path,
            seed=3001,
            held_out_evaluation_seeds=[901],
            held_out_evaluation_checkpoints=[4, 8],
        )
    )

    checkpoints = result["held_out_checkpoints"]
    assert [item["selected_member"] for item in checkpoints] == [1, 0]
    assert [item["model_calls"] for item in checkpoints] == [8, 16]
    assert result["endpoint_winners"]["held_out_resilience_auc_at_tick_4"] == 1
    assert result["endpoint_winners"]["held_out_resilience_auc_at_tick_8"] == 0


def test_population_sweep_has_nested_spawns_and_population_invariant_rng_state() -> None:
    base = cli._load(str(ROOT / "configs" / "demo.yaml"))
    small = BioFoundrySimulation(
        cli._population_config(base, 4, reference_population=12, world_scaling="fixed")
    )
    large = BioFoundrySimulation(
        cli._population_config(base, 16, reference_population=12, world_scaling="fixed")
    )

    assert list(zip(small.population.x, small.population.y, strict=True)) == list(
        zip(large.population.x[:4], large.population.y[:4], strict=True)
    )
    assert small.rng.bit_generator.state == large.rng.bit_generator.state


def test_fixed_schedule_matches_each_isolated_members_swarm_phase() -> None:
    base = cli._load(str(ROOT / "configs" / "demo.yaml"))
    swarm_config = replace(
        base,
        population=replace(base.population, agents=4, macro_interval=4),
        science=replace(
            base.science,
            enabled=True,
            immediate_replanning=True,
            decision_schedule="fixed",
        ),
    )
    swarm = BioFoundrySimulation(swarm_config)
    swarm.needs_replan[:] = True

    for tick in range(4):
        swarm.tick = tick
        expected_index = (-tick) % 4
        assert swarm.scheduled_macro_agents() == [f"agent_{expected_index:06d}"]
        member_config = replace(
            swarm_config,
            population=replace(
                swarm_config.population,
                agents=1,
                macro_phase_offset=expected_index,
                initial_positions=[
                    [
                        int(swarm.population.x[expected_index]),
                        int(swarm.population.y[expected_index]),
                    ]
                ],
            ),
        )
        member = BioFoundrySimulation(member_config)
        member.tick = tick
        assert member.scheduled_macro_agents() == ["agent_000000"]

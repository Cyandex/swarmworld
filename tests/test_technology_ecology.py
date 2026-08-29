import copy
from dataclasses import replace
from pathlib import Path

import numpy as np

from biofoundry.config import load_config
from biofoundry.materials import DEFAULT_RECIPES, required_inputs
from biofoundry.programs import ArtifactProgram
from biofoundry.science import ResearchMission
from biofoundry.simulation import BioFoundrySimulation
from biofoundry.technology_ecology import (
    evaluate_technological_ecosystem,
    evaluate_technological_ecosystem_generalization,
    technology_ecology_summary,
)
from biofoundry.types import (
    ActionType,
    AgentAction,
    ArtifactType,
    Direction,
    Resource,
    Terrain,
)

ROOT = Path(__file__).resolve().parents[1]


def test_cross_agent_fork_metric_excludes_system_starter_ancestry() -> None:
    base = load_config(ROOT / "configs" / "demo.yaml")
    config = replace(base, population=replace(base.population, agents=2))
    simulation = BioFoundrySimulation(config)
    first_agent, second_agent = simulation.agent_ids
    starter = ArtifactProgram(
        name="starter",
        instructions=({"op": "collect_water", "value": 0.01},),
    )
    own_parent = ArtifactProgram(
        name="own parent",
        instructions=({"op": "collect_water", "value": 0.02},),
        author=first_agent,
    )
    same_agent_child = ArtifactProgram(
        name="same-agent child",
        instructions=({"op": "collect_water", "value": 0.03},),
        author=first_agent,
    )
    cross_agent_child = ArtifactProgram(
        name="cross-agent child",
        instructions=({"op": "collect_water", "value": 0.04},),
        author=second_agent,
    )
    library = simulation.program_library
    library.register(starter, tick=0, artifact_id="starter")
    library.register(
        own_parent, tick=1, artifact_id="artifact_1", parent=starter
    )
    library.register(
        same_agent_child, tick=2, artifact_id="artifact_2", parent=own_parent
    )
    library.register(
        cross_agent_child, tick=3, artifact_id="artifact_3", parent=own_parent
    )

    summary = technology_ecology_summary(simulation)

    assert summary["program_forks"] == 3
    assert summary["agent_parent_program_forks"] == 2
    assert summary["cross_agent_program_forks"] == 1
    assert summary["cross_agent_program_fork_fraction"] == 0.5
    assert summary["program_fork_authorship_version"] == 2


def test_recipe_identity_uses_composition_and_process_not_scale_or_prose() -> None:
    recipe = copy.deepcopy(DEFAULT_RECIPES[ArtifactType.MATERIAL_SYSTEM])
    scaled_and_renamed = copy.deepcopy(recipe)
    for item in scaled_and_renamed["inputs"]:
        item["mass"] *= 0.25
    scaled_and_renamed["output_form"] = "a completely different invented form"
    scaled_and_renamed["design_principles"] = ["different prose"]

    changed_composition = copy.deepcopy(recipe)
    changed_composition["inputs"][0]["mass"] *= 1.2
    changed_process = copy.deepcopy(recipe)
    changed_process["steps"][0]["intensity"] += 0.05

    assert ResearchMission.recipe_id(recipe) == ResearchMission.recipe_id(
        scaled_and_renamed
    )
    assert ResearchMission.recipe_id(recipe) != ResearchMission.recipe_id(
        changed_composition
    )
    assert ResearchMission.recipe_id(recipe) != ResearchMission.recipe_id(
        changed_process
    )


def test_global_action_attempt_budget_counts_failures_and_blocks_excess() -> None:
    config = load_config(ROOT / "configs" / "demo.yaml")
    config = replace(
        config,
        population=replace(config.population, agents=2),
        simulation=replace(config.simulation, action_attempt_budget=1),
    )
    simulation = BioFoundrySimulation(config)
    actions = {
        agent_id: AgentAction(verb=ActionType.MOVE, direction=Direction.NORTH)
        for agent_id in simulation.agent_ids
    }

    first = simulation.step(actions)
    second = simulation.step(actions)

    assert simulation.action_attempts_used == 1
    assert simulation.action_attempts_blocked == 3
    assert simulation.action_budget_exhausted_tick == 0
    assert sum(event.kind == "action_budget_blocked" for event in first.events) == 1
    assert sum(event.kind == "action_budget_blocked" for event in second.events) == 2


def test_distributed_workspace_profile_has_fifty_agents_and_six_hubs() -> None:
    config = load_config(
        ROOT / "configs" / "openai-gpt-5.6-luna-technology-ecology-50.yaml"
    )
    simulation = BioFoundrySimulation(config)

    station_y, station_x = np.nonzero(simulation.world.stations)
    assert simulation.population.size == 50
    assert config.world.workspace_layout == "distributed"
    assert len(station_x) == 6
    assert len(set(zip(station_x.tolist(), station_y.tolist(), strict=True))) == 6
    assert np.count_nonzero(simulation.world.terrain == int(Terrain.FOUNDRY)) <= 25
    assert config.science.public_infrastructure_map is False
    assert config.evaluation.enabled is True
    assert config.simulation.action_attempt_budget is None
    assert config.llm.call_budget is None


def test_agent_free_ecosystem_assay_is_deterministic_and_nonmutating() -> None:
    config = load_config(ROOT / "configs" / "demo.yaml")
    config = replace(
        config,
        evaluation=replace(config.evaluation, enabled=True, horizon=4),
    )
    simulation = BioFoundrySimulation(config)
    recipe = DEFAULT_RECIPES[ArtifactType.MATERIAL_SYSTEM]
    for resource, mass in required_inputs(recipe).items():
        simulation.population.inventory[0, int(resource)] = np.float32(mass)
    simulation.step(
        {
            simulation.agent_ids[0]: AgentAction(
                verb=ActionType.BUILD,
                artifact=ArtifactType.MATERIAL_SYSTEM,
                recipe=recipe,
                artifact_spec={
                    "name": "Assay artifact",
                    "claimed_function": "autonomous field service",
                    "architecture": "branching porous body",
                    "bio_inspiration": ["vascular tissue"],
                    "predicted_effects": ["field regulation"],
                    "geometry": {},
                },
                program={
                    "name": "assay_program",
                    "instructions": [{"op": "collect_water", "value": 0.02}],
                },
            )
        }
    )
    before = simulation.state_digest()

    first = evaluate_technological_ecosystem(
        simulation, horizon=4, show_progress=False
    )
    second = evaluate_technological_ecosystem(
        simulation, horizon=4, show_progress=False
    )

    assert simulation.state_digest() == before
    assert first == second
    assert first["agent_actions_during_assay"] == 0
    assert len(first["knockouts"]) == 1
    assert len(first["intact"]["resilience_curve"]) == 4


def test_held_out_assay_changes_schedule_but_never_discovery_state() -> None:
    config = load_config(ROOT / "configs" / "demo.yaml")
    simulation = BioFoundrySimulation(config)
    before = simulation.state_digest()

    first = evaluate_technological_ecosystem(
        simulation,
        horizon=100,
        show_progress=False,
        detailed=False,
        disturbance_seed=901,
        include_knockouts=False,
    )
    second = evaluate_technological_ecosystem(
        simulation,
        horizon=100,
        show_progress=False,
        detailed=False,
        disturbance_seed=902,
        include_knockouts=False,
    )
    generalization = evaluate_technological_ecosystem_generalization(
        simulation,
        [901, 902],
        horizon=100,
        show_progress=False,
        bootstrap_resamples=20,
    )

    assert simulation.state_digest() == before
    assert first["disturbance_schedule"] != second["disturbance_schedule"]
    assert generalization["discovery_state_frozen"] is True
    assert generalization["agent_actions_per_schedule"] == 0
    assert generalization["n"] == 2


def test_no_communication_treatment_rejects_material_trade_at_engine_boundary() -> None:
    base = load_config(ROOT / "configs" / "demo.yaml")
    config = replace(
        base,
        population=replace(base.population, agents=2),
        science=replace(base.science, enabled=True, communication=False),
    )
    simulation = BioFoundrySimulation(config)
    simulation.population.x[:] = 10
    simulation.population.y[:] = 10
    simulation.population.inventory[0, int(Resource.KELP)] = np.float32(1.0)
    result = simulation.step(
        {
            simulation.agent_ids[0]: AgentAction(
                verb=ActionType.TRADE,
                resource=Resource.KELP,
                amount=0.25,
            )
        }
    )

    rejection = next(event for event in result.events if event.kind == "action_rejected")
    assert "disabled by this experimental treatment" in str(rejection.payload["reason"])
    assert simulation.population.inventory[0, int(Resource.KELP)] == 1.0

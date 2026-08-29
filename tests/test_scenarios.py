from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from biofoundry.config import GameConfig, ScienceConfig, load_config
from biofoundry.events import read_records
from biofoundry.policies.scripted import ScenarioIndustryPolicy
from biofoundry.scenarios import ScenarioMaterialLab, load_scenario, validate_scenario
from biofoundry.science import ResearchMission
from biofoundry.server import LiveGame
from biofoundry.simulation import BioFoundrySimulation
from biofoundry.structured_output import bounded_action_plan_text_format
from biofoundry.types import ActionType, Resource

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "worlds" / "ashen_realms"
CONFIG = PACKAGE / "configs" / "demo.yaml"


def test_legacy_configuration_serialization_is_unchanged() -> None:
    config = GameConfig()

    assert "scenario_package" not in config.as_dict()["world"]
    assert BioFoundrySimulation(config).scenario is None


def test_ashen_realms_package_validates_and_has_stable_catalogs() -> None:
    descriptor = validate_scenario(PACKAGE)

    assert descriptor["id"] == "ashen_realms"
    assert descriptor["catalog_limits"] == {
        "terrains": 9,
        "resources": 9,
        "facilities": 7,
        "operations": 10,
    }
    assert descriptor["catalogs"]["resources"][3]["id"] == "IRON_ORE"
    assert descriptor["catalogs"]["operations"][3]["id"] == "SMELT"


def test_scenario_generation_and_ticks_are_deterministic() -> None:
    config = load_config(CONFIG)
    first = BioFoundrySimulation(config)
    second = BioFoundrySimulation(config)
    policy = ScenarioIndustryPolicy()

    assert first.state_digest() == second.state_digest()
    assert first.world.snapshot()["scenario"]["id"] == "ashen_realms"
    for _ in range(12):
        first.step(policy.actions(first))
        second.step(policy.actions(second))
    assert first.state_digest() == second.state_digest()


def test_scenario_material_lab_accepts_external_vocabulary() -> None:
    scenario = load_scenario(PACKAGE)
    assert scenario is not None
    lab = ScenarioMaterialLab(scenario)
    recipe = scenario.externalize_recipe(scenario.default_recipe)
    inventory = np.zeros(len(Resource), dtype=np.float32)
    for resource, mass in lab.required_inputs(recipe).items():
        inventory[int(resource)] = np.float32(mass)

    batch = lab.execute(recipe, inventory, tick=7, contributors=["agent_000000"])
    encoded = batch.as_dict()

    assert encoded["recipe"]["steps"][2]["operation"] == "SMELT"
    assert set(encoded["composition"]) == set(lab.component_names)
    assert encoded["process_state"]["purity"] > 0.5
    assert batch.properties["stiffness"] > 0.0

    mission = ResearchMission(ScienceConfig(), scenario=scenario)
    staged = mission.stage_batch("agent_000000", recipe, batch)
    tested = mission.test_batch(mission.batches[0], tick=8)
    assert staged["source_recipe"]["inputs"][0]["resource"] == "IRON_ORE"
    assert tested["recipe_id"] == staged["recipe_id"]


def test_scenario_vocabulary_is_advertised_to_structured_agents() -> None:
    scenario = load_scenario(PACKAGE)
    assert scenario is not None
    contract = bounded_action_plan_text_format(
        4,
        resource_names=scenario.resource_names,
        operation_names=scenario.operation_names,
    )["schema"]

    assert contract["$defs"]["recipe_input"]["properties"]["resource"]["enum"] == [
        name for name in scenario.resource_names if name != "NONE"
    ]
    assert "SMELT" in contract["$defs"]["recipe_step"]["properties"]["operation"][
        "enum"
    ]


def test_scenario_observation_and_disturbance_use_scenario_meanings() -> None:
    config = load_config(CONFIG)
    config = replace(
        config,
        world=replace(config.world, disturbance_interval=2),
        population=replace(config.population, agents=2),
    )
    simulation = BioFoundrySimulation(config)
    observation = simulation.semantic_observation(0)

    assert '"id":"ashen_realms"' in observation
    assert "IRON_ORE" in observation
    simulation.step({})
    simulation.step({})
    result = simulation.step({})
    assert any(event.kind == "environmental_disturbance" for event in result.events)
    assert simulation.last_actions.tolist() == [int(ActionType.WAIT)] * 2


def test_live_server_selects_scenario_policy_and_normalizes_manual_actions(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "scenario-live.jsonl"
    live = LiveGame(load_config(CONFIG), record_path=trace)
    try:
        assert isinstance(live.policy, ScenarioIndustryPolicy)
        assert live.simulation.snapshot()["scenario"]["id"] == "ashen_realms"
        agent_id = live.simulation.agent_ids[0]
        result = live.handle_command(
            {
                "command": "manual_action",
                "agent": agent_id,
                "action": {"verb": "DEPOSIT", "resource": "IRON_ORE", "amount": 0.1},
            }
        )
        assert result["detail"] == ""
        assert live.manual_actions[agent_id].resource == Resource[
            live.simulation.scenario.slot("resources", "IRON_ORE")
        ]
    finally:
        live.close()
    header = next(read_records(trace))
    assert header["metadata"]["scenario"]["id"] == "ashen_realms"
    assert header["metadata"]["scenario"]["package_hash"]

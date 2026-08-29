import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from biofoundry.config import load_config
from biofoundry.counterfactuals import (
    compare_outcomes,
    outcome_summary,
    replay_intervention,
)
from biofoundry.knowledge import EmpiricalKnowledge
from biofoundry.materials import DEFAULT_RECIPES, required_inputs
from biofoundry.memory import MemoryRecord
from biofoundry.policies.scripted import (
    OracleResearchSocietyPolicy,
    ScriptedBioFoundryPolicy,
)
from biofoundry.simulation import BioFoundrySimulation
from biofoundry.types import ActionType, AgentAction, ArtifactType, Resource, Station, Terrain

ROOT = Path(__file__).resolve().parents[1]


def _config():
    return load_config(ROOT / "configs" / "demo.yaml")


ARTIFACT_SPEC = {
    "name": "Test lamellar collector",
    "claimed_function": "capture local moisture under wet conditions",
    "architecture": "branching lamellae feeding a reinforced collection edge",
    "bio_inspiration": ["leaf vasculature"],
    "predicted_effects": ["increased stored water"],
    "geometry": {
        "layers": 2,
        "surface_area": 1.4,
        "channel_density": 0.5,
        "anisotropy": 0.4,
        "branching": 0.7,
        "connectivity": 0.65,
        "curvature": 0.3,
        "modularity": 0.5,
    },
}


def test_seed_and_action_trace_are_deterministic() -> None:
    first = BioFoundrySimulation(_config())
    second = BioFoundrySimulation(_config())
    first_policy = ScriptedBioFoundryPolicy()
    second_policy = ScriptedBioFoundryPolicy()
    for _ in range(40):
        first.step(first_policy.actions(first))
        second.step(second_policy.actions(second))
    assert first.state_digest() == second.state_digest()


def test_agent_builds_programmed_artifact() -> None:
    simulation = BioFoundrySimulation(_config())
    artifact = ArtifactType.MATERIAL_SYSTEM
    recipe = DEFAULT_RECIPES[artifact]
    for resource, mass in required_inputs(recipe).items():
        simulation.population.inventory[0, int(resource)] = np.float32(mass)
    custom_program = {
        "name": "aggressive_capture",
        "instructions": [{"op": "collect_water", "value": 0.03}],
    }
    simulation.step(
        {
            simulation.agent_ids[0]: AgentAction(
                verb=ActionType.BUILD,
                artifact=artifact,
                recipe=recipe,
                program=custom_program,
                artifact_spec=ARTIFACT_SPEC,
            )
        }
    )
    assert simulation.artifacts.count == 1
    assert simulation.artifacts.programs[0].name == "aggressive_capture"
    snapshot = simulation.snapshot()
    assert snapshot["artifacts"]["count"] == 1
    assert isinstance(snapshot["causal_evidence"], list)
    initial = float(simulation.artifacts.storage[0])
    simulation.step({})
    assert float(simulation.artifacts.storage[0]) > initial


def test_agent_installs_program_that_executes_on_later_ticks() -> None:
    simulation = BioFoundrySimulation(_config())
    artifact = ArtifactType.MATERIAL_SYSTEM
    recipe = DEFAULT_RECIPES[artifact]
    for resource, mass in required_inputs(recipe).items():
        simulation.population.inventory[0, int(resource)] = np.float32(mass)
    agent_id = simulation.agent_ids[0]
    simulation.step(
        {
            agent_id: AgentAction(
                verb=ActionType.BUILD,
                artifact=artifact,
                recipe=recipe,
                artifact_spec=ARTIFACT_SPEC,
            )
        }
    )
    before_install = float(simulation.artifacts.storage[0])
    install_result = simulation.step(
        {
            agent_id: AgentAction(
                verb=ActionType.WRITE_PROGRAM,
                target_x=int(simulation.population.x[0]),
                target_y=int(simulation.population.y[0]),
                program={
                    "name": "agent_authored_capture",
                    "instructions": [{"op": "collect_water", "value": 0.05}],
                },
            )
        }
    )
    assert simulation.artifacts.programs[0].name == "agent_authored_capture"
    assert any(event.kind == "artifact_program_installed" for event in install_result.events)
    after_install = float(simulation.artifacts.storage[0])
    assert after_install > before_install
    simulation.step({})
    assert float(simulation.artifacts.storage[0]) > after_install


def test_artifact_id_disambiguates_colocated_program_targets() -> None:
    simulation = BioFoundrySimulation(_config())
    recipe = DEFAULT_RECIPES[ArtifactType.MATERIAL_SYSTEM]
    for resource, mass in required_inputs(recipe).items():
        simulation.population.inventory[0, int(resource)] = np.float32(2.0 * mass)
    agent_id = simulation.agent_ids[0]
    for suffix in ("first", "second"):
        simulation.step(
            {
                agent_id: AgentAction(
                    verb=ActionType.BUILD,
                    artifact=ArtifactType.MATERIAL_SYSTEM,
                    recipe=recipe,
                    artifact_spec={**ARTIFACT_SPEC, "name": suffix},
                )
            }
        )

    result = simulation.step(
        {
            agent_id: AgentAction(
                verb=ActionType.WRITE_PROGRAM,
                target_artifact_id="artifact_00000001",
                program={
                    "name": "second_artifact_program",
                    "instructions": [{"op": "collect_water", "value": 0.05}],
                },
            )
        }
    )

    assert simulation.artifacts.programs[0].name == "passive_material_system"
    assert simulation.artifacts.programs[1].name == "second_artifact_program"
    installed = [
        event for event in result.events if event.kind == "artifact_program_installed"
    ]
    assert installed[0].payload["artifact_id"] == "artifact_00000001"


def test_thousand_agent_core_step() -> None:
    config = _config()
    config = replace(config, population=replace(config.population, agents=1000))
    simulation = BioFoundrySimulation(config)
    simulation.step({})
    assert simulation.tick == 1
    assert simulation.population.x.shape == (1000,)


def test_agent_can_publish_persistent_insight() -> None:
    simulation = BioFoundrySimulation(_config())
    simulation.step(
        {
            simulation.agent_ids[0]: AgentAction(
                verb=ActionType.PUBLISH,
                insight={"kind": "observation", "content": "lamellae redirect cracks"},
            )
        }
    )
    assert len(simulation.archive.records) == 1
    assert len(simulation.deposited_insights) == 1
    assert simulation.deposited_insights[0]["published"] is True


def test_spatial_insight_is_visible_nearby_without_public_archive() -> None:
    config = _config()
    config.science.communication = False
    simulation = BioFoundrySimulation(config)
    simulation.population.x[1] = simulation.population.x[0]
    simulation.population.y[1] = simulation.population.y[0]
    simulation.step(
        {
            simulation.agent_ids[0]: AgentAction(
                verb=ActionType.DEPOSIT_INSIGHT,
                insight={"kind": "local_marker", "content": "Inspect this material site."},
            )
        }
    )

    observation = json.loads(simulation.semantic_observation(1))
    assert observation["public_archive"] == []
    assert observation["nearby_insights"][0]["kind"] == "local_marker"
    assert observation["nearby_insights"][0]["published"] is False


def test_public_communication_ablation_hides_task_board_and_global_scoreboard() -> None:
    config = _config()
    config.science.communication = False
    simulation = BioFoundrySimulation(config)
    assert simulation.research is not None
    simulation.research.claim(simulation.agent_ids[1], "an otherwise hidden task")

    result = simulation.step(
        {
            simulation.agent_ids[0]: AgentAction(
                verb=ActionType.CLAIM_TASK,
                message="broadcast this task",
            )
        }
    )

    assert any(event.kind == "action_rejected" for event in result.events)
    observation = json.loads(simulation.semantic_observation(0))
    mission = observation["research_mission"]
    assert mission["task_claims"] == {}
    assert mission["public_evidence_catalog"] == []
    assert "public_progress" not in mission
    assert "completed" not in mission
    assert "success_thresholds" not in mission
    assert "world_counts" not in observation


def test_publication_exposes_structured_evidence_without_hidden_evaluation() -> None:
    simulation = BioFoundrySimulation(_config())
    simulation.population.inventory[0, int(Resource.KELP)] = np.float32(0.2)
    simulation.step(
        {
            simulation.agent_ids[0]: AgentAction(
                verb=ActionType.PUBLISH,
                insight={
                    "kind": "material_evidence",
                    "content": "I directly sampled KELP at this site.",
                },
            )
        }
    )

    observation = json.loads(simulation.semantic_observation(1))
    catalog = observation["research_mission"]["public_evidence_catalog"]
    assert catalog[0]["author"] == simulation.agent_ids[0]
    assert catalog[0]["grounded_resources"] == ["KELP"]
    assert "evaluation_utility" not in catalog[0]


def test_social_awareness_ablation_retains_evidence_but_hides_peer_profiles() -> None:
    config = _config()
    config.science.social_awareness = False
    simulation = BioFoundrySimulation(config)
    simulation.population.inventory[0, int(Resource.KELP)] = np.float32(0.2)
    simulation.step(
        {
            simulation.agent_ids[0]: AgentAction(
                verb=ActionType.PUBLISH,
                insight={"kind": "evidence", "content": "Measured KELP sample."},
            ),
            simulation.agent_ids[1]: AgentAction(
                verb=ActionType.CLAIM_TASK,
                message="study water capture",
            ),
        }
    )

    mission = json.loads(simulation.semantic_observation(2))["research_mission"]
    assert len(mission["public_evidence_catalog"]) == 1
    assert mission["public_agent_profiles"] == {}
    assert mission["task_claims"] == {}


def test_snapshot_exposes_each_agents_latest_action() -> None:
    simulation = BioFoundrySimulation(_config())
    simulation.step(
        {
            simulation.agent_ids[0]: AgentAction(
                verb=ActionType.MOVE,
                direction=2,
            )
        }
    )
    actions = simulation.snapshot()["agents"]["last_action"]
    assert actions[0] == int(ActionType.MOVE)
    assert actions[1] == int(ActionType.WAIT)


def test_resources_are_discovered_locally_and_preserved_by_inspection() -> None:
    simulation = BioFoundrySimulation(_config())
    ys, xs = np.nonzero(simulation.world.resource_kind == int(Resource.KELP))
    distance_to_nearest = np.full(
        (simulation.world.height, simulation.world.width), 10_000, dtype=np.int32
    )
    for resource_x, resource_y in zip(xs, ys, strict=True):
        grid_y, grid_x = np.indices(distance_to_nearest.shape)
        distance_to_nearest = np.minimum(
            distance_to_nearest,
            np.abs(grid_x - resource_x) + np.abs(grid_y - resource_y),
        )
    far_y, far_x = np.unravel_index(np.argmax(distance_to_nearest), distance_to_nearest.shape)
    simulation.population.x[0] = far_x
    simulation.population.y[0] = far_y
    hidden = json.loads(simulation.semantic_observation(0))
    assert "KELP" not in hidden["nearest_resources"]

    simulation.population.x[0] = int(xs[0])
    simulation.population.y[0] = int(ys[0])
    discovered = json.loads(simulation.semantic_observation(0))
    assert discovered["nearest_resources"]["KELP"]["distance"] == 0
    simulation.step({simulation.agent_ids[0]: AgentAction(verb=ActionType.INSPECT)})
    notebook = simulation.memories[0].notebook[-1]
    assert '"resource": "KELP"' in notebook.content
    assert f'"x": {int(xs[0])}' in notebook.content


def test_every_executable_material_is_reachable_in_the_generated_world() -> None:
    simulation = BioFoundrySimulation(_config())
    present = {
        Resource(int(value)) for value in np.unique(simulation.world.resource_kind)
    }
    assert present == set(Resource)


def test_disturbances_are_deterministic_dynamic_world_events() -> None:
    first = BioFoundrySimulation(_config())
    second = BioFoundrySimulation(_config())
    event_kinds: list[str] = []
    for _ in range(98):
        first_result = first.step({})
        second.step({})
        event_kinds.extend(event.kind for event in first_result.events)

    assert "environmental_disturbance" in event_kinds
    assert float(first.world.contamination.max()) > 0.30
    assert first.state_digest() == second.state_digest()


def test_station_discovery_persists_and_interrupts_a_stale_plan() -> None:
    simulation = BioFoundrySimulation(_config())
    station_y, station_x = np.argwhere(simulation.world.stations == int(Station.TESTER))[0]
    simulation.population.x[0] = int(station_x)
    simulation.population.y[0] = int(station_y)
    simulation.empirical_knowledge[0] = EmpiricalKnowledge(simulation.world.width)

    result = simulation.step(
        {simulation.agent_ids[0]: AgentAction(verb=ActionType.INSPECT)}
    )
    discoveries = [event for event in result.events if event.kind == "salient_discovery"]
    assert any("TESTER" in event.payload["stations"] for event in discoveries)
    assert simulation.needs_replan[0]
    assert simulation.agent_ids[0] in simulation.scheduled_macro_agents()

    simulation.population.x[0] = 0
    simulation.population.y[0] = 0
    observation = json.loads(simulation.semantic_observation(0))
    remembered = observation["empirical_knowledge"]["observed_stations"]["TESTER"]
    assert remembered["nearest_observed_site"]["position"] == [
        int(station_x),
        int(station_y),
    ]


def test_terrain_landmark_coordinates_persist_after_agent_moves_away() -> None:
    simulation = BioFoundrySimulation(_config())
    foundry_y, foundry_x = np.argwhere(
        simulation.world.terrain == int(Terrain.FOUNDRY)
    )[0]
    simulation.population.x[0] = int(foundry_x)
    simulation.population.y[0] = int(foundry_y)
    simulation.empirical_knowledge[0] = EmpiricalKnowledge(simulation.world.width)
    simulation.step({simulation.agent_ids[0]: AgentAction(verb=ActionType.INSPECT)})

    simulation.population.x[0] = 0
    simulation.population.y[0] = 0
    observation = json.loads(simulation.semantic_observation(0))
    remembered = observation["empirical_knowledge"]["observed_terrains"]["FOUNDRY"]

    assert remembered["nearest_observed_site"]["position"] == [
        int(foundry_x),
        int(foundry_y),
    ]


def test_local_affordances_expose_legality_without_prescribing_science() -> None:
    simulation = BioFoundrySimulation(_config())
    foundry_y, foundry_x = np.argwhere(
        simulation.world.terrain == int(Terrain.FOUNDRY)
    )[0]
    simulation.population.x[0] = int(foundry_x)
    simulation.population.y[0] = int(foundry_y)

    observation = json.loads(simulation.semantic_observation(0))
    affordances = observation["local_affordances"]

    assert affordances["fabrication_workspace_here"] is True
    assert affordances["shared_depot_access_here"] is True
    assert set(affordances["walkable_directions"]).issubset(
        {"NORTH", "EAST", "SOUTH", "WEST"}
    )
    assert "recommended_action" not in affordances
    assert "recipe" not in affordances


def test_public_infrastructure_map_exposes_labs_but_not_resources() -> None:
    simulation = BioFoundrySimulation(_config())
    observation = json.loads(simulation.semantic_observation(0))
    infrastructure = observation["public_infrastructure"]

    assert "FOUNDRY_WORKSPACE" in infrastructure
    assert "TESTER" in infrastructure
    assert infrastructure["FOUNDRY_WORKSPACE"]["supports"] == ["DEPOSIT", "OPERATE"]
    assert all(resource.name not in infrastructure for resource in Resource)
    assert "recommended_action" not in infrastructure


def test_public_infrastructure_map_can_be_removed_as_navigation_control() -> None:
    config = _config()
    config.science.public_infrastructure_map = False
    simulation = BioFoundrySimulation(config)

    observation = json.loads(simulation.semantic_observation(0))

    assert observation["public_infrastructure"] == {}


def test_local_affordances_report_collectable_matter_and_test_readiness() -> None:
    simulation = BioFoundrySimulation(_config())
    resource_y, resource_x = np.argwhere(
        simulation.world.resource_kind == int(Resource.KELP)
    )[0]
    simulation.population.x[0] = int(resource_x)
    simulation.population.y[0] = int(resource_y)

    before = json.loads(simulation.semantic_observation(0))["local_affordances"]
    assert before["collectable_material_here"] == "KELP"
    assert before["untested_microbatch_ready"] is False


def test_harvest_samples_present_matter_without_a_named_material_request() -> None:
    simulation = BioFoundrySimulation(_config())
    ys, xs = np.nonzero(simulation.world.resource_kind == int(Resource.KELP))
    simulation.population.x[0] = int(xs[0])
    simulation.population.y[0] = int(ys[0])
    simulation.empirical_knowledge[0] = EmpiricalKnowledge(simulation.world.width)

    agent_id = simulation.agent_ids[0]
    result = simulation.step({agent_id: AgentAction(verb=ActionType.HARVEST)})

    assert float(simulation.population.inventory[0, int(Resource.KELP)]) > 0.0
    harvested = [event for event in result.events if event.kind == "resource_harvested"]
    assert len(harvested) == 1
    assert harvested[0].payload["resource"] == int(Resource.KELP)
    evidence = simulation.empirical_knowledge[0].view(
        x=int(xs[0]),
        y=int(ys[0]),
        inventory=simulation.population.inventory[0],
    )
    assert evidence["extraction_outcomes"]["KELP"] == {
        "attempts": 1,
        "successes": 1,
    }

    mass_before = float(simulation.world.resource_mass[int(ys[0]), int(xs[0])])
    rejected_result = simulation.step(
        {
            agent_id: AgentAction(
                verb=ActionType.HARVEST,
                resource=Resource.KELP,
            )
        }
    )
    assert any(event.kind == "action_rejected" for event in rejected_result.events)
    assert float(simulation.world.resource_mass[int(ys[0]), int(xs[0])]) >= mass_before


def test_executable_recipes_require_generic_empirical_grounding() -> None:
    simulation = BioFoundrySimulation(_config())
    ys, xs = np.nonzero(simulation.world.resource_kind == int(Resource.KELP))
    simulation.population.x[0] = int(xs[0])
    simulation.population.y[0] = int(ys[0])
    simulation.empirical_knowledge[0] = EmpiricalKnowledge(simulation.world.width)
    simulation.step({})

    grounded = simulation.grounded_resources(0)
    assert Resource.KELP in grounded
    unseen = next(
        resource for resource in Resource if resource != Resource.NONE and resource not in grounded
    )
    speculative_recipe = {
        "inputs": [{"resource": unseen.name, "mass": 0.2}],
        "steps": [{"operation": "PRESS", "intensity": 0.4}],
        "output_form": "empirical test coupon",
        "design_principles": ["compare an untested material hypothesis"],
    }
    result = simulation.step(
        {
            simulation.agent_ids[0]: AgentAction(
                verb=ActionType.PROPOSE_RECIPE,
                recipe=speculative_recipe,
            )
        }
    )

    rejected = [event for event in result.events if event.kind == "action_rejected"]
    assert len(rejected) == 1
    assert "lack empirical grounding" in rejected[0].payload["reason"]
    assert unseen.name in rejected[0].payload["reason"]
    assert not any(event.kind == "recipe_proposed" for event in result.events)

    assert simulation.research is not None
    simulation.research.deposit(simulation.agent_ids[1], unseen, 0.5)
    assert unseen not in simulation.grounded_resources(0)

    observation = json.loads(simulation.semantic_observation(0))
    knowledge = observation["empirical_knowledge"]
    assert unseen.name not in knowledge["grounded_materials"]
    assert unseen.name in observation["research_mission"]["shared_depot"]
    assert knowledge["scope_note"] == "Unlisted materials are unobserved, not proven absent."


def test_failed_action_is_observed_and_triggers_immediate_replanning() -> None:
    simulation = BioFoundrySimulation(_config())
    agent_id = simulation.agent_ids[0]
    result = simulation.step(
        {
            agent_id: AgentAction(
                verb=ActionType.BUILD,
                artifact=ArtifactType.MATERIAL_SYSTEM,
                recipe=DEFAULT_RECIPES[ArtifactType.MATERIAL_SYSTEM],
                artifact_spec=ARTIFACT_SPEC,
            )
        }
    )
    assert any(event.kind == "action_rejected" for event in result.events)
    assert any(event.kind == "action_result" for event in result.events)
    assert simulation.needs_replan[0]
    assert agent_id in simulation.scheduled_macro_agents()
    observation = simulation.semantic_observation(0)
    assert "combined personal and depot inventory is insufficient" in observation


def test_agent_authored_action_intent_persists_in_private_feedback() -> None:
    simulation = BioFoundrySimulation(_config())
    intent = "Establish a local baseline before returning to the foundry."

    simulation.step(
        {
            simulation.agent_ids[0]: AgentAction(
                verb=ActionType.INSPECT,
                message=intent,
            )
        }
    )

    observation = json.loads(simulation.semantic_observation(0))
    outcome = observation["recent_action_results"][-1]
    assert outcome["intent"] == intent
    assert intent in simulation.memories[0].episodic[-1].content


def test_recent_behavior_summary_is_factual_and_contains_no_recommendation() -> None:
    simulation = BioFoundrySimulation(_config())
    agent_id = simulation.agent_ids[0]
    simulation.step({agent_id: AgentAction(verb=ActionType.INSPECT)})
    simulation.step({agent_id: AgentAction(verb=ActionType.INSPECT)})

    observation = json.loads(simulation.semantic_observation(0))
    summary = observation["recent_behavior_summary"]

    assert summary["action_counts"] == {"INSPECT": 2}
    assert summary["successful_actions"] == 2
    assert summary["failed_actions"] == 0
    assert summary["latest_action_streak"] == 2
    assert "recommended_action" not in summary


def test_memory_latest_returns_newest_record_of_requested_kind() -> None:
    simulation = BioFoundrySimulation(_config())
    memory = simulation.memories[0]

    memory.remember(MemoryRecord("old", 2, "research_state", "old"), notebook=True)
    memory.remember(MemoryRecord("other", 4, "observation", "other"), notebook=True)
    memory.remember(MemoryRecord("new", 8, "research_state", "new"), notebook=True)

    assert memory.latest("research_state").content == "new"
    assert memory.latest("missing") is None


def test_unmeasured_publication_does_not_interrupt_open_loop_plan() -> None:
    simulation = BioFoundrySimulation(_config())
    simulation.needs_replan[0] = False

    result = simulation.step(
        {
            simulation.agent_ids[0]: AgentAction(
                verb=ActionType.PUBLISH,
                insight={
                    "content": "Local moisture varies across adjacent terrain.",
                    "kind": "observation",
                    "salience": 0.6,
                },
            )
        }
    )

    assert not simulation.needs_replan[0]
    requests = [event for event in result.events if event.kind == "replan_requested"]
    assert requests == []


def test_unmeasured_recipe_proposal_does_not_discard_open_loop_plan() -> None:
    simulation = BioFoundrySimulation(_config())
    ys, xs = np.nonzero(simulation.world.resource_kind == int(Resource.KELP))
    simulation.population.x[0] = int(xs[0])
    simulation.population.y[0] = int(ys[0])
    simulation._observe_visible_cells(0)  # Prime passive first-of-kind discovery.
    simulation.needs_replan[0] = False
    recipe = {
        "inputs": [{"resource": "KELP", "mass": 0.1}],
        "steps": [{"operation": "WASH", "intensity": 0.5}],
        "output_form": "washed fiber",
        "design_principles": ["controlled comparison"],
    }

    result = simulation.step(
        {
            simulation.agent_ids[0]: AgentAction(
                verb=ActionType.PROPOSE_RECIPE,
                recipe=recipe,
            )
        }
    )

    assert any(event.kind == "recipe_proposed" for event in result.events)
    assert not simulation.needs_replan[0]
    assert not any(event.kind == "replan_requested" for event in result.events)


def test_agent_observes_pending_batch_without_hidden_properties_until_test() -> None:
    simulation = BioFoundrySimulation(_config())
    assert "private_experiments" not in json.loads(simulation.semantic_observation(0))
    recipe = {
        "inputs": [{"resource": "KELP", "mass": 0.2}],
        "steps": [{"operation": "WASH", "intensity": 0.5}],
        "output_form": "washed kelp coupon",
        "design_principles": ["controlled comparison"],
    }
    ys, xs = np.nonzero(simulation.world.terrain == int(Terrain.FOUNDRY))
    simulation.population.x[0] = int(xs[0])
    simulation.population.y[0] = int(ys[0])
    simulation.population.inventory[0, int(Resource.KELP)] = np.float32(0.2)
    simulation._observe_visible_cells(0)
    simulation.needs_replan[0] = False

    simulation.step(
        {
            simulation.agent_ids[0]: AgentAction(
                verb=ActionType.OPERATE,
                recipe=recipe,
            )
        }
    )

    before_test = json.loads(simulation.semantic_observation(0))["private_experiments"]
    pending = before_test["pending_microbatches"]
    assert pending[0]["batch_id"] == "batch_00000000"
    assert pending[0]["source_recipe"] == recipe
    assert pending[0]["status"] == "UNTESTED"
    assert "properties" not in pending[0]
    assert "material_utility" not in pending[0]
    assert not simulation.needs_replan[0]
    assert not any(event.kind == "replan_requested" for event in simulation.last_events)

    simulation.step(
        {simulation.agent_ids[0]: AgentAction(verb=ActionType.TEST)}
    )

    after_test = json.loads(simulation.semantic_observation(0))["private_experiments"]
    assert after_test["pending_microbatches"] == []
    measured = after_test["recent_test_results"][0]
    assert measured["batch_id"] == "batch_00000000"
    assert "properties" in measured
    assert "material_utility" in measured
    assert measured["source_recipe"] == recipe
    assert "passes_material_target" not in measured
    assert "passes_material_target" not in simulation.memories[0].notebook[-1].content


def test_oracle_society_completes_grounded_collective_science_cycle() -> None:
    simulation = BioFoundrySimulation(_config())
    policy = OracleResearchSocietyPolicy()
    event_kinds: set[str] = set()
    for _ in range(120):
        result = simulation.step(policy.actions(simulation))
        event_kinds.update(event.kind for event in result.events)
        if simulation.research is not None:
            card = simulation.research.scorecard(
                simulation.artifacts, len(simulation.archive.records)
            )
            if card["completed"]:
                break
    assert simulation.research is not None
    card = simulation.research.scorecard(simulation.artifacts, len(simulation.archive.records))
    assert card["completed"] is True
    assert card["score"] == 1.0
    assert simulation.artifacts.count == 1
    assert len(simulation.artifacts.provenance[0]["contributors"]) >= 3
    assert len(simulation.artifacts.provenance[0]["program_history"]) == 2
    assert {
        "sample_inspected",
        "resource_deposited",
        "design_combined",
        "microbatch_fabricated",
        "material_tested",
        "artifact_built",
        "artifact_program_installed",
        "research_mission_completed",
    } <= event_kinds


def test_combination_rejects_two_records_from_the_same_agent() -> None:
    simulation = BioFoundrySimulation(_config())
    author = simulation.agent_ids[0]
    for content in ("first local observation", "second local observation"):
        simulation.step(
            {
                author: AgentAction(
                    verb=ActionType.PUBLISH,
                    insight={"kind": "observation", "content": content},
                )
            }
        )
    parents = [record.record_id for record in simulation.archive.records]
    result = simulation.step(
        {
            simulation.agent_ids[1]: AgentAction(
                verb=ActionType.COMBINE_DESIGN,
                recipe=DEFAULT_RECIPES[ArtifactType.MATERIAL_SYSTEM],
                causal_parents=parents,
            )
        }
    )
    rejected = [event for event in result.events if event.kind == "action_rejected"]
    assert len(rejected) == 1
    assert "distinct agents" in str(rejected[0].payload["reason"])


def test_combination_inherits_only_cited_empirical_material_grounding() -> None:
    simulation = BioFoundrySimulation(_config())
    first, second, integrator = 0, 1, 2
    simulation.population.inventory[first, int(Resource.KELP)] = np.float32(0.2)
    simulation.population.inventory[second, int(Resource.SHELL)] = np.float32(0.2)

    simulation.step(
        {
            simulation.agent_ids[first]: AgentAction(
                verb=ActionType.PUBLISH,
                insight={"kind": "material_evidence", "content": "Measured local KELP."},
            ),
            simulation.agent_ids[second]: AgentAction(
                verb=ActionType.PUBLISH,
                insight={"kind": "material_evidence", "content": "Measured local SHELL."},
            ),
        }
    )
    parent_ids = [record.record_id for record in simulation.archive.records]
    recipe = DEFAULT_RECIPES[ArtifactType.MATERIAL_SYSTEM]

    empty_y, empty_x = np.argwhere(
        simulation.world.resource_kind == int(Resource.NONE)
    )[0]
    simulation.population.x[integrator] = int(empty_x)
    simulation.population.y[integrator] = int(empty_y)
    simulation.empirical_knowledge[integrator] = EmpiricalKnowledge(simulation.world.width)
    independent = simulation.step(
        {
            simulation.agent_ids[integrator]: AgentAction(
                verb=ActionType.PROPOSE_RECIPE,
                recipe=recipe,
            )
        }
    )
    assert any(event.kind == "action_rejected" for event in independent.events)

    combined = simulation.step(
        {
            simulation.agent_ids[integrator]: AgentAction(
                verb=ActionType.COMBINE_DESIGN,
                recipe=recipe,
                causal_parents=parent_ids,
            )
        }
    )
    assert any(event.kind == "design_combined" for event in combined.events)
    assert simulation.research is not None
    proposal = simulation.research.proposals[-1]
    assert proposal["grounding_expanded"] is True
    assert set(proposal["inherited_resources"]) == {"KELP", "SHELL"}
    card = simulation.research.scorecard(
        simulation.artifacts, len(simulation.archive.records)
    )
    assert card["counts"]["grounding_expanding_combinations"] == 1
    assert card["composition_synergy"] == pytest.approx(-0.012873)
    assert card["composition_gain"] == card["composition_synergy"]
    assert card["best_independent_recipe_utility"] is None


def test_teach_persists_selected_publication_in_nearby_notebook() -> None:
    simulation = BioFoundrySimulation(_config())
    teacher, recipient = 0, 1
    simulation.population.x[recipient] = simulation.population.x[teacher]
    simulation.population.y[recipient] = simulation.population.y[teacher]
    simulation.step(
        {
            simulation.agent_ids[teacher]: AgentAction(
                verb=ActionType.PUBLISH,
                insight={
                    "kind": "controlled_observation",
                    "content": "Measured KELP hydration after washing.",
                    "salience": 0.8,
                },
            )
        }
    )
    record_id = simulation.archive.records[-1].record_id
    result = simulation.step(
        {
            simulation.agent_ids[teacher]: AgentAction(
                verb=ActionType.TEACH,
                causal_parents=[record_id],
            )
        }
    )
    assert any(event.kind == "knowledge_taught" for event in result.events)
    assert record_id in {
        record.record_id for record in simulation.memories[recipient].notebook
    }


def test_teach_can_transfer_private_empirical_record_without_publishing() -> None:
    simulation = BioFoundrySimulation(_config())
    teacher, recipient = 0, 1
    simulation.population.x[recipient] = simulation.population.x[teacher]
    simulation.population.y[recipient] = simulation.population.y[teacher]
    simulation.step(
        {
            simulation.agent_ids[teacher]: AgentAction(verb=ActionType.INSPECT),
        }
    )
    record_id = simulation.memories[teacher].notebook[-1].record_id
    assert record_id not in simulation.archive.by_id

    result = simulation.step(
        {
            simulation.agent_ids[teacher]: AgentAction(
                verb=ActionType.TEACH,
                causal_parents=[record_id],
            )
        }
    )

    assert any(event.kind == "knowledge_taught" for event in result.events)
    assert record_id in {
        record.record_id for record in simulation.memories[recipient].notebook
    }
    assert record_id not in simulation.archive.by_id


def test_snapshot_exposes_only_private_evidence_cited_by_public_outcomes() -> None:
    simulation = BioFoundrySimulation(_config())
    agent = simulation.agent_ids[0]
    simulation.step({agent: AgentAction(verb=ActionType.INSPECT)})
    observation_id = simulation.memories[0].notebook[-1].record_id
    uncited = MemoryRecord(
        record_id="private_uncited_note",
        tick=simulation.tick,
        kind="observation",
        content=json.dumps({"agent": agent, "x": 1, "y": 2}),
    )
    simulation.memories[0].remember(uncited, notebook=True)
    simulation.step(
        {
            agent: AgentAction(
                verb=ActionType.PUBLISH,
                insight={"kind": "field_note", "content": "A cited local observation."},
                causal_parents=[observation_id],
            )
        }
    )

    evidence = simulation.snapshot()["causal_evidence"]
    cited = next(item for item in evidence if item["record_id"] == observation_id)
    assert cited["author"] == agent
    assert cited["kind"] == "observation"
    assert cited["summary"].startswith("Observed ")
    assert not any(item["record_id"] == uncited.record_id for item in evidence)


def test_cross_agent_artifact_lineage_counts_without_ceremonial_combine_verb() -> None:
    config = _config()
    config.science.target_material_utility = 0.0
    config.science.target_artifact_performance = 0.0
    config.science.target_behavioral_novelty = 0.0
    simulation = BioFoundrySimulation(config)
    recipe = DEFAULT_RECIPES[ArtifactType.MATERIAL_SYSTEM]

    simulation.population.inventory[0, int(Resource.KELP)] = np.float32(0.2)
    simulation.population.inventory[1, int(Resource.SHELL)] = np.float32(0.2)
    simulation.step(
        {
            simulation.agent_ids[0]: AgentAction(
                verb=ActionType.PUBLISH,
                insight={"kind": "independent_test", "content": "Grounded KELP evidence."},
            )
        }
    )
    kelp_record = simulation.archive.records[-1].record_id
    simulation.step(
        {
            simulation.agent_ids[1]: AgentAction(
                verb=ActionType.PUBLISH,
                insight={"kind": "synthesis", "content": "Grounded SHELL evidence."},
                causal_parents=[kelp_record],
            )
        }
    )
    # The artifact cites one synthesis, whose ancestry contains grounded evidence
    # from both authors. Direct-parent count is not a collaboration ceremony.
    parents = [simulation.archive.records[-1].record_id]
    builder = 2
    station_y, station_x = np.argwhere(simulation.world.stations != 0)[0]
    simulation.population.x[builder] = int(station_x)
    simulation.population.y[builder] = int(station_y)
    for resource, mass in required_inputs(recipe).items():
        simulation.population.inventory[builder, int(resource)] = np.float32(
            mass * (1.0 + config.science.test_scale)
        )

    simulation.step(
        {
            simulation.agent_ids[builder]: AgentAction(
                verb=ActionType.OPERATE,
                recipe=recipe,
                causal_parents=parents,
            )
        }
    )
    simulation.step(
        {simulation.agent_ids[builder]: AgentAction(verb=ActionType.TEST)}
    )
    simulation.step(
        {
            simulation.agent_ids[builder]: AgentAction(
                verb=ActionType.BUILD,
                artifact=ArtifactType.MATERIAL_SYSTEM,
                recipe=recipe,
                causal_parents=parents,
                artifact_spec=ARTIFACT_SPEC,
                program={
                    "name": "lineage_test_program",
                    "instructions": [{"op": "collect_water", "value": 0.01}],
                },
            )
        }
    )

    assert simulation.research is not None
    card = simulation.research.scorecard(
        simulation.artifacts, len(simulation.archive.records)
    )
    assert card["milestones"]["combine_independent_evidence"] is False
    assert card["milestones"]["causally_composed_artifact"] is True
    assert card["milestones"]["distributed_contribution"] is True
    assert card["emergent_success"] is True
    assert card["inventions"][0]["causal_lineage"]["max_depth"] == 2
    assert card["inventions"][0]["causal_lineage"]["authors"] == [
        simulation.agent_ids[0],
        simulation.agent_ids[1],
    ]


def test_collective_science_build_requires_explicit_recipe() -> None:
    simulation = BioFoundrySimulation(_config())
    result = simulation.step(
        {
            simulation.agent_ids[0]: AgentAction(
                verb=ActionType.BUILD,
                artifact=ArtifactType.MATERIAL_SYSTEM,
            )
        }
    )
    rejected = [event for event in result.events if event.kind == "action_rejected"]
    assert len(rejected) == 1
    assert "explicit tested recipe" in str(rejected[0].payload["reason"])
    assert simulation.artifacts.count == 0


def test_agent_action_removal_replay_establishes_oracle_indispensability() -> None:
    config = _config()
    simulation = BioFoundrySimulation(config)
    policy = OracleResearchSocietyPolicy()
    action_records: list[dict[str, object]] = []
    for _ in range(100):
        actions = policy.actions(simulation)
        action_records.append(
            {
                "tick": simulation.tick,
                "actions": {agent_id: action.as_dict() for agent_id, action in actions.items()},
            }
        )
        simulation.step(actions)
    factual = replay_intervention(config.as_dict(), action_records, show_progress=False)
    counterfactual = replay_intervention(
        config.as_dict(),
        action_records,
        {simulation.agent_ids[0]},
        show_progress=False,
    )
    assert factual.state_digest() == simulation.state_digest()
    comparison = compare_outcomes(outcome_summary(factual), outcome_summary(counterfactual))
    assert comparison["outcome_lost"] is True
    assert comparison["artifact_performance_loss"] > 0.15

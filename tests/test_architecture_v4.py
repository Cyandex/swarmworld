from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from biofoundry.artifacts import ArtifactSystem
from biofoundry.config import EconomyConfig, GameConfig, PhysicsConfig
from biofoundry.events import EventRecorder, read_records
from biofoundry.materials import DEFAULT_RECIPES, MaterialLab, required_inputs
from biofoundry.memory import AgentMemory, MemoryRecord
from biofoundry.programs import ArtifactProgram, instruction_diff
from biofoundry.simulation import BioFoundrySimulation
from biofoundry.types import ActionType, AgentAction, ArtifactType, Resource, Station
from biofoundry.world import BioWorld


def _batch():
    recipe = DEFAULT_RECIPES[ArtifactType.MATERIAL_SYSTEM]
    inventory = np.zeros(len(Resource), dtype=np.float32)
    for resource, mass in required_inputs(recipe).items():
        inventory[int(resource)] = np.float32(mass)
    return MaterialLab().execute(recipe, inventory, 0, ["agent_000000"])


def _v4_config() -> GameConfig:
    base = GameConfig()
    return replace(
        base,
        science=replace(
            base.science,
            addressed_communication=True,
            program_forking=True,
            skill_library=True,
            retrieval_diagnostics=True,
        ),
        physics=PhysicsConfig(
            closed_artifact_fluxes=True,
            dismantle_recovery=True,
        ),
    )


def test_new_architecture_mechanisms_are_opt_in_by_default() -> None:
    config = GameConfig()
    assert config.world.generator == "fixed"
    assert config.economy.enabled is False
    assert config.physics.closed_artifact_fluxes is False
    assert config.physics.dismantle_recovery is False
    assert config.science.addressed_communication is False
    assert config.science.program_forking is False
    assert config.science.skill_library is False
    assert config.science.retrieval_diagnostics is False
    assert config.trace.deduplicate_prompts is False
    assert config.trace.compression == "none"


def test_procedural_world_is_deterministic_reachable_and_structurally_variable() -> None:
    config = replace(GameConfig().world, generator="procedural")
    first = BioWorld(config, np.random.default_rng(19))
    replay = BioWorld(config, np.random.default_rng(19))
    other = BioWorld(config, np.random.default_rng(20))
    np.testing.assert_array_equal(first.terrain, replay.terrain)
    assert first.generator_manifest == replay.generator_manifest
    assert first.generator_manifest != other.generator_manifest
    for resource in Resource:
        if resource != Resource.NONE:
            assert np.any(first.resource_kind == int(resource))
    for station in Station:
        if station != Station.NONE:
            assert np.count_nonzero(first.stations == int(station)) == 1
            y, x = np.argwhere(first.stations == int(station))[0]
            assert first.walkable[y, x]


def test_artifact_water_and_nutrient_fluxes_are_locally_conservative() -> None:
    world = BioWorld(GameConfig().world, np.random.default_rng(7))
    artifacts = ArtifactSystem(2, PhysicsConfig(closed_artifact_fluxes=True))
    x, y = 10, 10
    world.moisture[y, x] = np.float32(0.8)
    water_program = ArtifactProgram.from_dict(
        {"name": "collector", "instructions": [{"op": "collect_water", "value": 0.02}]}
    )
    water = artifacts.add(
        ArtifactType.MATERIAL_SYSTEM, x, y, "agent_000000", _batch(), 0, water_program
    )
    moisture_before = float(world.moisture[y, x])
    storage_before = float(artifacts.storage[water])
    artifacts.step(world, 1)
    assert moisture_before - float(world.moisture[y, x]) == pytest.approx(
        float(artifacts.storage[water]) - storage_before, abs=1e-7
    )

    signal_program = ArtifactProgram.from_dict(
        {"name": "signal", "instructions": [{"op": "emit_signal", "value": 0.01}]}
    )
    signal = artifacts.add(
        ArtifactType.MATERIAL_SYSTEM, x + 1, y, "agent_000000", _batch(), 0, signal_program
    )
    nutrient_before = float(world.nutrients[y, x + 1])
    reserve_before = float(artifacts.reserve[signal])
    artifacts.step(world, 2)
    assert float(world.nutrients[y, x + 1]) - nutrient_before == pytest.approx(
        reserve_before - float(artifacts.reserve[signal]), abs=1e-7
    )


def test_program_identity_fork_lineage_and_verified_library() -> None:
    simulation = BioFoundrySimulation(_v4_config())
    index = 0
    x, y = int(simulation.population.x[index]), int(simulation.population.y[index])
    parent = ArtifactProgram.from_dict(
        {"name": "parent", "instructions": [{"op": "collect_water", "value": 0.01}]},
        author=simulation.agent_ids[index],
    )
    artifact = simulation.artifacts.add(
        ArtifactType.MATERIAL_SYSTEM, x, y, simulation.agent_ids[index], _batch(), 0, parent
    )
    simulation.program_library.register(parent, tick=0, artifact_id=f"artifact_{artifact:08d}")
    simulation.step({simulation.agent_ids[index]: AgentAction(verb=ActionType.INSPECT)})
    skill = simulation.program_library.skills[simulation.agent_ids[index]][parent.program_id]
    assert skill["verified"] is True
    child_data = {
        "name": "child",
        "parent_program": parent.program_id,
        "instructions": [{"op": "collect_water", "value": 0.02}],
    }
    result = simulation.step(
        {
            simulation.agent_ids[index]: AgentAction(
                verb=ActionType.FORK_PROGRAM,
                target_artifact_id=f"artifact_{artifact:08d}",
                program=child_data,
            )
        }
    )
    child = simulation.artifacts.programs[artifact]
    assert child is not None and child.program_id != parent.program_id
    assert instruction_diff(parent, child)
    assert len(simulation.program_library.lineage_edges) == 1
    assert any(event.kind == "artifact_program_installed" for event in result.events)
    simulation.population.x[1] = simulation.population.x[index]
    simulation.population.y[1] = simulation.population.y[index]
    taught = simulation.step(
        {
            simulation.agent_ids[index]: AgentAction(
                verb=ActionType.TEACH,
                target_agent_id=simulation.agent_ids[1],
                causal_parents=[parent.program_id],
            )
        }
    )
    assert any(event.kind == "knowledge_taught" for event in taught.events)
    assert simulation.program_library.knows(simulation.agent_ids[1], parent.program_id)


def test_addressed_threaded_message_reaches_only_the_selected_neighbor() -> None:
    simulation = BioFoundrySimulation(_v4_config())
    simulation.population.x[:3] = 10
    simulation.population.y[:3] = 10
    sender, target, bystander = simulation.agent_ids[:3]
    result = simulation.step(
        {
            sender: AgentAction(
                verb=ActionType.COMMUNICATE,
                target_agent_id=target,
                message="Can you measure the nearby material?",
            )
        }
    )
    delivered = next(event for event in result.events if event.kind == "message_delivered")
    assert delivered.payload["recipients"] == [target]
    record_id = str(delivered.payload["record_id"])
    assert simulation.memories[1].get(record_id) is not None
    assert simulation.memories[2].get(record_id) is None
    reply = simulation.step(
        {
            target: AgentAction(
                verb=ActionType.COMMUNICATE,
                target_agent_id=sender,
                reply_to=record_id,
                message="Yes; I will inspect it.",
            )
        }
    )
    response = next(event for event in reply.events if event.kind == "message_delivered")
    assert response.payload["reply_to"] == record_id
    assert bystander not in response.payload["recipients"]
    simulation.population.inventory[1, int(Resource.FUNGUS)] = np.float32(0.5)
    fulfilled = simulation.step(
        {
            target: AgentAction(
                verb=ActionType.TRADE,
                target_agent_id=sender,
                reply_to=record_id,
                resource=Resource.FUNGUS,
                amount=0.25,
            )
        }
    )
    assert any(event.kind == "resource_traded" for event in fulfilled.events)
    assert simulation.message_records[record_id]["fulfillments"][0]["verb"] == "TRADE"


def test_optional_metabolism_mortality_respawn_and_cultural_inheritance() -> None:
    economy = EconomyConfig(
        enabled=True,
        mortality_enabled=True,
        respawn_enabled=True,
        cultural_inheritance=True,
        passive_cost=0.6,
        respawn_delay=1,
    )
    simulation = BioFoundrySimulation(replace(_v4_config(), economy=economy))
    index = 0
    agent = simulation.agent_ids[index]
    inherited_program = ArtifactProgram.from_dict(
        {"name": "measured", "instructions": [{"op": "collect_water", "value": 0.01}]},
        author=agent,
    )
    simulation.program_library.register(
        inherited_program, tick=0, artifact_id="artifact_cultural"
    )
    simulation.program_library.observe(
        agent,
        inherited_program.program_id,
        tick=0,
        source="measured_artifact",
        evidence_id="observation_cultural",
        verified=True,
        measurements={"performance": 0.2},
    )
    simulation.population.inventory[index, int(Resource.FUNGUS)] = np.float32(1.0)
    simulation.population.energy[index] = np.float32(0.2)
    metabolic = simulation.step(
        {
            agent: AgentAction(
                verb=ActionType.METABOLIZE, resource=Resource.FUNGUS, amount=0.5
            )
        }
    )
    assert any(event.kind == "resource_metabolized" for event in metabolic.events)
    assert simulation.population.inventory[index, int(Resource.FUNGUS)] < 1.0
    for _ in range(4):
        simulation.step({})
    assert simulation.population.generation[index] >= 1
    assert simulation.population.active[index]
    inherited = simulation.program_library.skills[agent][inherited_program.program_id]
    assert inherited["inherited"] is True


def test_dismantling_is_single_use_and_recovers_bounded_mass() -> None:
    simulation = BioFoundrySimulation(_v4_config())
    index = 0
    x, y = int(simulation.population.x[index]), int(simulation.population.y[index])
    artifact = simulation.artifacts.add(
        ArtifactType.MATERIAL_SYSTEM,
        x,
        y,
        simulation.agent_ids[index],
        _batch(),
        0,
    )
    action = AgentAction(
        verb=ActionType.DISMANTLE, target_artifact_id=f"artifact_{artifact:08d}"
    )
    first = simulation.step({simulation.agent_ids[index]: action})
    payload = next(event.payload for event in first.events if event.kind == "artifact_dismantled")
    assert payload["recovered"]
    second = simulation.step({simulation.agent_ids[index]: action})
    assert any(
        event.kind == "action_rejected" and "already" in str(event.payload.get("reason"))
        for event in second.events
    )


def test_manual_repair_consumes_exactly_the_realized_artifact_reserve() -> None:
    simulation = BioFoundrySimulation(_v4_config())
    index = 0
    x, y = int(simulation.population.x[index]), int(simulation.population.y[index])
    inert_program = ArtifactProgram.from_dict(
        {
            "name": "inert",
            "instructions": [{"op": "const", "dest": "r0", "value": 0.0}],
        }
    )
    artifact = simulation.artifacts.add(
        ArtifactType.MATERIAL_SYSTEM,
        x,
        y,
        simulation.agent_ids[index],
        _batch(),
        0,
        inert_program,
    )
    simulation.artifacts.health[artifact] = np.float32(0.5)
    reserve_before = float(simulation.artifacts.reserve[artifact])
    flux_before = float(simulation.artifacts.fluxes[artifact, 3])
    result = simulation.step(
        {
            simulation.agent_ids[index]: AgentAction(
                verb=ActionType.REPAIR,
                target_artifact_id=f"artifact_{artifact:08d}",
            )
        }
    )
    payload = next(event.payload for event in result.events if event.kind == "artifact_repaired")
    realized = float(payload["health_restored"])
    assert realized > 0.0
    assert reserve_before - float(simulation.artifacts.reserve[artifact]) == pytest.approx(
        realized, abs=1e-6
    )
    assert float(simulation.artifacts.fluxes[artifact, 3]) - flux_before == pytest.approx(
        realized, abs=1e-6
    )


def test_retrieval_diagnostics_and_lossless_compressed_prompt_dedup(tmp_path: Path) -> None:
    memory = AgentMemory(16)
    memory.remember(MemoryRecord("water", 1, "test", "water capture moisture", 0.5))
    memory.remember(MemoryRecord("mineral", 2, "test", "mineral stiffness", 0.5))
    selected = memory.retrieve({"water", "moisture"}, limit=1)
    assert selected[0].record_id == "water"
    memory.mark_retrieval_use({"water"})
    assert memory.last_retrieval["used_as_causal_parent"] == ["water"]

    path = tmp_path / "trace.jsonl.gz"
    repeated = "long action manual " * 100
    with EventRecorder(
        path,
        {"test": True},
        compression="gzip",
        deduplicate_prompts=True,
    ) as recorder:
        for tick in range(2):
            recorder.write_record(
                {
                    "type": "model_trace",
                    "tick": tick,
                    "messages": [{"role": "system", "content": repeated}],
                    "_trace_templates": {"manual": repeated},
                }
            )
    records = list(read_records(path))
    traces = [record for record in records if record["type"] == "model_trace"]
    templates = [record for record in records if record["type"] == "trace_template"]
    assert len(templates) == 1
    assert all(trace["messages"][0]["content"] == repeated for trace in traces)

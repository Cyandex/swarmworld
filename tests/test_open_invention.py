from dataclasses import replace

import numpy as np

from biofoundry.artifacts import ArtifactSystem
from biofoundry.config import GameConfig, ScienceConfig
from biofoundry.materials import DEFAULT_RECIPES, MaterialLab, required_inputs
from biofoundry.programs import ArtifactProgram
from biofoundry.science import ResearchMission
from biofoundry.types import ArtifactType, Resource
from biofoundry.world import BioWorld


def _batch():
    recipe = DEFAULT_RECIPES[ArtifactType.MATERIAL_SYSTEM]
    inventory = np.zeros(len(Resource), dtype=np.float32)
    for resource, mass in required_inputs(recipe).items():
        inventory[int(resource)] = np.float32(mass)
    return MaterialLab().execute(
        recipe,
        inventory,
        tick=0,
        contributors=["agent_000000"],
    )


def _spec(name: str, *, connectivity: float = 0.5) -> dict[str, object]:
    return {
        "name": name,
        "claimed_function": f"Unverified claim made by {name}",
        "architecture": f"Free-form architecture description for {name}",
        "bio_inspiration": ["an agent-selected biological analogy"],
        "predicted_effects": ["a falsifiable field effect"],
        "geometry": {
            "layers": 3,
            "surface_area": 1.4,
            "channel_density": 0.35,
            "anisotropy": 0.45,
            "branching": 0.6,
            "connectivity": connectivity,
            "curvature": 0.4,
            "modularity": 0.5,
        },
    }


def test_artifact_enum_contains_no_prespecified_inventions() -> None:
    assert list(ArtifactType) == [ArtifactType.NONE, ArtifactType.MATERIAL_SYSTEM]


def test_prose_identity_cannot_change_artifact_physics() -> None:
    config = replace(GameConfig().world, width=20, height=20)
    world = BioWorld(config, np.random.default_rng(7))
    artifacts = ArtifactSystem(limit=3)
    program = ArtifactProgram.from_dict(
        {
            "name": "same_program",
            "instructions": [{"op": "collect_water", "value": 0.05}],
        },
        author="agent_000000",
    )
    for name in ("Grandiose miracle membrane", "Plain specimen B"):
        artifacts.add(
            ArtifactType.MATERIAL_SYSTEM,
            10,
            10,
            name,
            _batch(),
            tick=0,
            program=program,
            artifact_spec=_spec(name),
        )
    artifacts.step(world, tick=1)
    np.testing.assert_allclose(artifacts.services[0], artifacts.services[1])
    assert artifacts.performance[0] == artifacts.performance[1]


def test_agent_selected_geometry_changes_measured_service() -> None:
    config = replace(GameConfig().world, width=20, height=20)
    world = BioWorld(config, np.random.default_rng(11))
    artifacts = ArtifactSystem(limit=2)
    for index, connectivity in enumerate((0.05, 0.95)):
        artifacts.add(
            ArtifactType.MATERIAL_SYSTEM,
            10,
            10,
            f"agent_{index:06d}",
            _batch(),
            tick=0,
            artifact_spec=_spec(f"system {index}", connectivity=connectivity),
        )
    artifacts.step(world, tick=1)
    structural_service = 2
    assert artifacts.services[1, structural_service] > artifacts.services[0, structural_service]


def test_reprogramming_preserves_lifetime_peak_and_closes_program_epoch() -> None:
    config = replace(GameConfig().world, width=20, height=20)
    world = BioWorld(config, np.random.default_rng(17))
    artifacts = ArtifactSystem(limit=1)
    first_program = ArtifactProgram.from_dict(
        {
            "name": "first",
            "instructions": [{"op": "collect_water", "value": 0.04}],
        },
        author="agent_000000",
    )
    index = artifacts.add(
        ArtifactType.MATERIAL_SYSTEM,
        10,
        10,
        "agent_000000",
        _batch(),
        tick=0,
        program=first_program,
        artifact_spec=_spec("versioned system"),
    )
    artifacts.step(world, tick=1)
    lifetime_before = float(artifacts.lifetime_peak_performance[index])
    assert lifetime_before > 0.0

    second_program = ArtifactProgram.from_dict(
        {
            "name": "second",
            "instructions": [{"op": "const", "dest": "r0", "value": 0.0}],
        },
        author="agent_000001",
    )
    artifacts.install_program(index, second_program, tick=2)

    assert artifacts.peak_performance[index] == 0.0
    assert artifacts.lifetime_peak_performance[index] == lifetime_before
    history = artifacts.provenance[index]["program_history"]
    assert history[0]["ended_tick"] == 2
    assert history[0]["peak_performance"] == round(lifetime_before, 6)
    assert history[1]["program"]["program_id"] == second_program.program_id


def test_portfolio_metric_rewards_complementary_services_not_artifact_count() -> None:
    artifacts = ArtifactSystem(limit=3)
    artifacts.count = 2
    artifacts.creator[:2] = ["agent_000000", "agent_000001"]
    artifacts.peak_services[0, 0] = np.float32(0.8)
    artifacts.peak_services[1, 1] = np.float32(0.8)
    artifacts.lifetime_peak_services[:] = artifacts.peak_services
    mission = ResearchMission(ScienceConfig())

    portfolio = mission.portfolio_metrics(artifacts)

    assert portfolio["service_breadth"] == 2
    assert portfolio["portfolio_contributors"] == 2
    assert portfolio["mean_behavioral_redundancy"] == 0.0
    assert portfolio["self_organized_portfolio"] is True
    assert portfolio["portfolio_resilience"] == 0.133333

from pathlib import Path

import numpy as np

from biofoundry.config import load_config
from biofoundry.dynamics import ACTION_NAMES, SocietyDynamicsTracker, regional_crowding
from biofoundry.simulation import BioFoundrySimulation
from biofoundry.types import ActionType

ROOT = Path(__file__).resolve().parents[1]


def test_regional_crowding_detects_a_cluster_without_cell_collisions() -> None:
    spread_x = np.tile(np.arange(8, dtype=np.int64) * 10 + 5, 6)
    spread_y = np.repeat(np.arange(6, dtype=np.int64) * 10 + 5, 8)
    packed_x = np.arange(48, dtype=np.int64) % 8 + 31
    packed_y = np.arange(48, dtype=np.int64) // 8 + 21

    spread = regional_crowding(spread_x, spread_y, width=80, height=60)
    packed = regional_crowding(packed_x, packed_y, width=80, height=60)

    assert spread == 0.0
    assert len(set(zip(packed_x.tolist(), packed_y.tolist(), strict=True))) == 48
    assert packed > 0.45


def test_dynamics_point_is_compact_bounded_and_population_normalized() -> None:
    simulation = BioFoundrySimulation(load_config(ROOT / "configs" / "demo.yaml"))
    tracker = SocietyDynamicsTracker()

    simulation.population.x[:] = np.arange(simulation.population.size)
    simulation.population.y[:] = 10
    dispersed = tracker.record(simulation, {"research": {"score": 0.25}})
    assert dispersed["spatial_concentration"] == 0.0
    assert dispersed["research_score"] == 0.25
    assert len(dispersed["action_distribution"]) == len(ACTION_NAMES)
    assert sum(dispersed["action_distribution"]) == 1.0

    simulation.tick += 1
    simulation.population.x[:] = 20
    simulation.population.y[:] = 13
    concentrated = tracker.record(simulation, {"research": {"score": 0.25}})
    assert concentrated["spatial_concentration"] == 1.0
    for name in (
        "mean_energy",
        "spatial_concentration",
        "research_score",
        "communication_rate",
        "specialization",
        "behavioral_diversity",
    ):
        assert 0.0 <= concentrated[name] <= 1.0


def test_specialization_distinguishes_persistent_roles_from_shared_behavior() -> None:
    config = load_config(ROOT / "configs" / "demo.yaml")
    specialized_simulation = BioFoundrySimulation(config)
    shared_simulation = BioFoundrySimulation(config)
    specialized = SocietyDynamicsTracker(role_window=32)
    shared = SocietyDynamicsTracker(role_window=32)
    split = specialized_simulation.population.size // 2

    for tick in range(1, 33):
        specialized_simulation.tick = tick
        specialized_simulation.last_actions[:split] = int(ActionType.MOVE)
        specialized_simulation.last_actions[split:] = int(ActionType.INSPECT)
        specialized.record(specialized_simulation, {"research": {}})

        shared_simulation.tick = tick
        shared_simulation.last_actions[:] = int(
            ActionType.MOVE if tick % 2 else ActionType.INSPECT
        )
        shared.record(shared_simulation, {"research": {}})

    assert specialized.points[-1]["specialization"] > 0.99
    assert shared.points[-1]["specialization"] < 0.01
    assert specialized.points[-1]["behavioral_diversity"] > 0.0
    assert shared.points[-1]["behavioral_diversity"] > 0.0


def test_history_packet_carries_schema_and_all_recorded_ticks() -> None:
    simulation = BioFoundrySimulation(load_config(ROOT / "configs" / "demo.yaml"))
    tracker = SocietyDynamicsTracker(role_window=8)
    for tick in range(5):
        simulation.tick = tick
        tracker.record(simulation, {"research": {}})

    packet = tracker.packet(sample_interval=2)
    assert packet["version"] == 3
    assert packet["sample_interval"] == 2
    assert packet["role_window"] == 8
    assert packet["action_names"][int(ActionType.COMMUNICATE)] == "COMMUNICATE"
    assert [point["tick"] for point in packet["points"]] == list(range(5))

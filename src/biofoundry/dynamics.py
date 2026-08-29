"""Compact, online observables for the full trajectory of a live society."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

from .types import ActionType

if TYPE_CHECKING:
    from .simulation import BioFoundrySimulation


DYNAMICS_VERSION = 3
ACTION_NAMES = [action.name for action in ActionType]
COMMUNICATION_ACTIONS = np.asarray(
    [int(ActionType.COMMUNICATE), int(ActionType.TEACH), int(ActionType.TRADE)],
    dtype=np.int16,
)


def regional_crowding(
    x: np.ndarray,
    y: np.ndarray,
    *,
    width: int,
    height: int,
    columns: int = 8,
    rows: int = 6,
) -> float:
    """Return the probability that two agents occupy the same map-scale region.

    Exact-cell occupancy misses the failure mode where a population fills distinct
    cells around one attractive facility. A coarse occupancy collision probability
    detects that regional aggregation in O(N) time, so it remains usable for much
    larger populations.
    """

    population = int(x.size)
    if population <= 1:
        return 1.0 if population == 1 else 0.0
    bin_x = np.minimum(
        columns - 1,
        np.maximum(0, x.astype(np.int64) * columns // max(1, int(width))),
    )
    bin_y = np.minimum(
        rows - 1,
        np.maximum(0, y.astype(np.int64) * rows // max(1, int(height))),
    )
    counts = np.bincount(bin_y * columns + bin_x, minlength=columns * rows)
    colocated_pairs = float(np.sum(counts * (counts - 1), dtype=np.float64))
    return float(np.clip(colocated_pairs / (population * (population - 1)), 0.0, 1.0))


class SocietyDynamicsTracker:
    """Measure presentation-only collective observables without retaining world frames."""

    def __init__(self, *, role_window: int = 64) -> None:
        if role_window <= 0:
            raise ValueError("role_window must be positive")
        self.role_window = int(role_window)
        self.points: list[dict[str, Any]] = []
        self._action_history: deque[np.ndarray] = deque(maxlen=self.role_window)

    @staticmethod
    def _spatial_concentration(simulation: BioFoundrySimulation) -> float:
        active = np.nonzero(simulation.population.active)[0]
        population = int(active.size)
        if population <= 1:
            return 1.0 if population == 1 else 0.0
        cells = (
            simulation.population.y[active].astype(np.int64) * simulation.world.width
            + simulation.population.x[active].astype(np.int64)
        )
        _, counts = np.unique(cells, return_counts=True)
        shares = counts.astype(np.float64) / population
        hhi = float(np.sum(shares * shares))
        dispersed_baseline = 1.0 / population
        return float(np.clip((hhi - dispersed_baseline) / (1.0 - dispersed_baseline), 0.0, 1.0))

    def _role_metrics(
        self,
        simulation: BioFoundrySimulation,
    ) -> tuple[list[float], float, float, float]:
        actions = np.asarray(simulation.last_actions, dtype=np.int16).copy()
        if not self.points or self.points[-1]["tick"] != int(simulation.tick):
            self._action_history.append(actions)
        active = np.nonzero(simulation.population.active)[0]
        action_count = len(ACTION_NAMES)
        if active.size == 0:
            return [0.0] * action_count, 0.0, 0.0, 0.0

        history = np.stack(tuple(self._action_history), axis=0)[:, active]
        counts = np.bincount(history.ravel(), minlength=action_count).astype(np.float64)
        distribution = counts / max(1.0, float(counts.sum()))

        positive = distribution > 0
        entropy = -float(np.sum(distribution[positive] * np.log(distribution[positive])))
        behavioral_diversity = entropy / np.log(action_count) if action_count > 1 else 0.0

        joint_counts = np.zeros((active.size, action_count), dtype=np.float64)
        for agent_index in range(active.size):
            joint_counts[agent_index] = np.bincount(
                history[:, agent_index], minlength=action_count
            )
        joint = joint_counts / max(1.0, float(joint_counts.sum()))
        agent_probability = joint.sum(axis=1, keepdims=True)
        action_probability = joint.sum(axis=0, keepdims=True)
        expected = agent_probability * action_probability
        valid = (joint > 0) & (expected > 0)
        mutual_information = float(np.sum(joint[valid] * np.log(joint[valid] / expected[valid])))
        specialization = mutual_information / entropy if entropy > 1e-12 else 0.0

        communication_rate = float(distribution[COMMUNICATION_ACTIONS].sum())
        return (
            np.round(distribution, 6).tolist(),
            float(np.clip(behavioral_diversity, 0.0, 1.0)),
            float(np.clip(specialization, 0.0, 1.0)),
            communication_rate,
        )

    def record(
        self,
        simulation: BioFoundrySimulation,
        snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one sample, or replace the latest sample for the same tick."""

        presentation = snapshot or simulation.snapshot(display_limit=0)
        distribution, diversity, specialization, communication_rate = self._role_metrics(
            simulation
        )
        active = simulation.population.active
        energy = simulation.population.energy[active]
        point = {
            "tick": int(simulation.tick),
            "artifact_utility": round(float(simulation.artifacts.summary_score()), 6),
            "best_artifact_performance": round(
                float(
                    np.max(
                        simulation.artifacts.lifetime_peak_performance[
                            : simulation.artifacts.count
                        ],
                        initial=0.0,
                    )
                ),
                6,
            ),
            "best_current_program_peak_performance": round(
                float(
                    np.max(
                        simulation.artifacts.peak_performance[
                            : simulation.artifacts.count
                        ],
                        initial=0.0,
                    )
                ),
                6,
            ),
            "best_final_state_artifact_performance": round(
                float(
                    np.max(
                        simulation.artifacts.performance[
                            : simulation.artifacts.count
                        ],
                        initial=0.0,
                    )
                ),
                6,
            ),
            "mean_energy": round(float(np.mean(energy)) if energy.size else 0.0, 6),
            "spatial_concentration": round(self._spatial_concentration(simulation), 6),
            "regional_crowding": round(
                regional_crowding(
                    simulation.population.x[active],
                    simulation.population.y[active],
                    width=simulation.world.width,
                    height=simulation.world.height,
                ),
                6,
            ),
            "mean_distance_traveled": round(
                float(np.mean(simulation.distance_traveled[active]))
                if np.any(active)
                else 0.0,
                6,
            ),
            "mean_distinct_cells_visited": round(
                float(
                    np.mean(
                        [
                            len(simulation.visited_cells[index])
                            for index in np.nonzero(active)[0].tolist()
                        ]
                    )
                )
                if np.any(active)
                else 0.0,
                6,
            ),
            "moving_agent_fraction": round(
                float(
                    np.mean(
                        simulation.last_actions[active] == int(ActionType.MOVE)
                    )
                )
                if np.any(active)
                else 0.0,
                6,
            ),
            "research_score": round(float(presentation.get("research", {}).get("score", 0.0)), 6),
            "artifact_count": int(simulation.artifacts.count),
            "communication_rate": round(communication_rate, 6),
            "specialization": round(specialization, 6),
            "behavioral_diversity": round(diversity, 6),
            "action_distribution": distribution,
        }
        if self.points and self.points[-1]["tick"] == point["tick"]:
            self.points[-1] = point
        else:
            self.points.append(point)
        return point

    def packet(self, *, sample_interval: int) -> dict[str, Any]:
        return {
            "version": DYNAMICS_VERSION,
            "sample_interval": max(1, int(sample_interval)),
            "role_window": self.role_window,
            "action_names": ACTION_NAMES,
            "points": self.points,
        }

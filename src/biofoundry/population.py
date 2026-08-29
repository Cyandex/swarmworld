"""Structure-of-arrays population state for interactive and large-swarm modes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from .config import EconomyConfig, PopulationConfig
from .types import DIRECTION_DELTAS, Resource
from .world import BioWorld


class Population:
    def __init__(
        self,
        config: PopulationConfig,
        world: BioWorld,
        rng: np.random.Generator,
        economy: EconomyConfig | None = None,
    ) -> None:
        self.config = config
        self.economy = economy or EconomyConfig()
        self.size = config.agents
        walk_y, walk_x = np.nonzero(world.walkable)
        # Use a canonical ordering of walkable cells. Fixed-world population sweeps
        # then use nested prefixes, and initialization consumes the same RNG work for
        # every N.
        spawn_order = rng.permutation(len(walk_x))
        selected = np.resize(spawn_order, self.size)
        default_x = walk_x[selected].astype(np.int32)
        default_y = walk_y[selected].astype(np.int32)
        if config.initial_positions is None:
            self.x: NDArray[np.int32] = default_x
            self.y: NDArray[np.int32] = default_y
        else:
            positions = np.asarray(config.initial_positions, dtype=np.int32)
            self.x = positions[:, 0].copy()
            self.y = positions[:, 1].copy()
            if not np.all(world.walkable[self.y, self.x]):
                raise ValueError("population.initial_positions must all be walkable")
        initial_energy = self.economy.initial_energy if self.economy.enabled else 1.0
        self.energy: NDArray[np.float32] = np.full(
            self.size, initial_energy, dtype=np.float32
        )
        self.inventory: NDArray[np.float32] = np.zeros(
            (self.size, len(Resource)), dtype=np.float32
        )
        self.active: NDArray[np.bool_] = np.ones(self.size, dtype=np.bool_)
        self.generation: NDArray[np.int32] = np.zeros(self.size, dtype=np.int32)
        self.death_tick: NDArray[np.int64] = np.full(self.size, -1, dtype=np.int64)
        self.last_macroturn: NDArray[np.int64] = np.full(self.size, -1, dtype=np.int64)
        self.agent_ids = [f"agent_{index:06d}" for index in range(self.size)]
        self.id_to_index = {name: index for index, name in enumerate(self.agent_ids)}

    def apply_movement(
        self,
        directions: NDArray[np.int16],
        world: BioWorld,
    ) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.bool_]]:
        old_x = self.x.copy()
        old_y = self.y.copy()
        dx = np.zeros(self.size, dtype=np.int32)
        dy = np.zeros(self.size, dtype=np.int32)
        for direction, (step_x, step_y) in DIRECTION_DELTAS.items():
            mask = directions == int(direction)
            dx[mask] = step_x
            dy[mask] = step_y
        proposed_x = np.clip(self.x + dx, 0, world.width - 1)
        proposed_y = np.clip(self.y + dy, 0, world.height - 1)
        valid = world.walkable[proposed_y, proposed_x] & self.active
        self.x[valid] = proposed_x[valid]
        self.y[valid] = proposed_y[valid]
        moved = (self.x != old_x) | (self.y != old_y)
        if not self.economy.enabled:
            self.energy[moved] = np.maximum(
                0.0, self.energy[moved] - np.float32(0.001)
            )
        return old_x, old_y, moved

    def free_capacity(self, index: int) -> float:
        return max(
            0.0,
            self.config.inventory_capacity - float(self.inventory[index].sum()),
        )

    def add_resource(self, index: int, resource: Resource, amount: float) -> float:
        accepted = min(max(0.0, amount), self.free_capacity(index))
        if resource != Resource.NONE and accepted > 0:
            self.inventory[index, int(resource)] += np.float32(accepted)
        return accepted

    def build_spatial_index(self, width: int) -> dict[int, list[int]]:
        buckets: dict[int, list[int]] = defaultdict(list)
        for index in np.nonzero(self.active)[0].tolist():
            buckets[int(self.y[index]) * width + int(self.x[index])].append(index)
        return buckets

    def neighbors(
        self,
        index: int,
        radius: int,
        width: int,
        height: int,
        buckets: dict[int, list[int]],
    ) -> list[int]:
        if radius <= 0:
            return []
        cx, cy = int(self.x[index]), int(self.y[index])
        found: list[int] = []
        radius_sq = radius * radius
        for y in range(max(0, cy - radius), min(height, cy + radius + 1)):
            for x in range(max(0, cx - radius), min(width, cx + radius + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 > radius_sq:
                    continue
                for candidate in buckets.get(y * width + x, []):
                    if candidate != index:
                        found.append(candidate)
        return found

    def indices_for_ids(self, agent_ids: Iterable[str]) -> list[int]:
        return [self.id_to_index[name] for name in agent_ids if name in self.id_to_index]

    def snapshot(self, limit: int | None = None) -> dict[str, object]:
        count = self.size if limit is None else min(self.size, limit)
        return {
            "count": self.size,
            "display_count": count,
            "ids": self.agent_ids[:count],
            "x": self.x[:count].tolist(),
            "y": self.y[:count].tolist(),
            "energy": np.round(self.energy[:count], 4).tolist(),
            "active": self.active[:count].tolist(),
            "generation": self.generation[:count].tolist(),
            "visor": (np.arange(count, dtype=np.int16) % 12).tolist(),
            "inventory_total": np.round(self.inventory[:count].sum(axis=1), 4).tolist(),
        }

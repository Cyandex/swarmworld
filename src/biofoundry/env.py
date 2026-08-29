"""PettingZoo ParallelEnv adapter over the authoritative simulation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .config import GameConfig
from .simulation import BioFoundrySimulation
from .types import ActionType, AgentAction, ArtifactType, Direction, Resource


class BioFoundryParallelEnv(ParallelEnv):
    """Simultaneous multi-agent interface for baselines and controlled experiments."""

    metadata = {
        "name": "biofoundry_world_v0",
        "render_modes": ["state"],
        "is_parallelizable": True,
    }

    def __init__(self, config: GameConfig, render_mode: str | None = None):
        self.config = config
        self.render_mode = render_mode
        self.simulation = BioFoundrySimulation(config)
        self.possible_agents = list(self.simulation.agent_ids)
        self.agents = list(self.possible_agents)
        self._agent_name_mapping = {
            name: index for index, name in enumerate(self.possible_agents)
        }
        self._observation_spaces = {
            agent: spaces.Dict(
                {
                    "position": spaces.MultiDiscrete(
                        np.asarray([config.world.width, config.world.height], dtype=np.int64)
                    ),
                    "energy": spaces.Box(
                        0.0,
                        config.economy.maximum_energy if config.economy.enabled else 1.0,
                        shape=(1,),
                        dtype=np.float32,
                    ),
                    "inventory": spaces.Box(
                        0.0,
                        config.population.inventory_capacity,
                        shape=(len(Resource),),
                        dtype=np.float32,
                    ),
                    "local_patch": spaces.Box(
                        0.0, 1.0, shape=(5, 5, 5), dtype=np.float32
                    ),
                    "tick": spaces.Box(
                        0,
                        config.simulation.max_ticks,
                        shape=(1,),
                        dtype=np.int64,
                    ),
                }
            )
            for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: spaces.Dict(
                {
                    "verb": spaces.Discrete(len(ActionType)),
                    "direction": spaces.Discrete(len(Direction)),
                    "resource": spaces.Discrete(len(Resource)),
                    "artifact": spaces.Discrete(len(ArtifactType)),
                    "amount": spaces.Box(
                        0.0,
                        config.population.inventory_capacity,
                        shape=(),
                        dtype=np.float32,
                    ),
                    "target": spaces.MultiDiscrete(
                        np.asarray([config.world.width, config.world.height], dtype=np.int64)
                    ),
                    "payload": spaces.Text(min_length=0, max_length=4096),
                }
            )
            for agent in self.possible_agents
        }

    def observation_space(self, agent: str) -> spaces.Space[Any]:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space[Any]:
        return self._action_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        del options
        self.simulation.reset(seed)
        self.agents = list(self.possible_agents)
        observations = {
            agent: self.simulation.observation(self._agent_name_mapping[agent])
            for agent in self.agents
        }
        infos = {
            agent: {
                "text_observation": self.simulation.semantic_observation(
                    self._agent_name_mapping[agent]
                )
            }
            for agent in self.agents
        }
        return observations, infos

    @staticmethod
    def _decode_action(value: AgentAction | Mapping[str, Any] | None) -> AgentAction:
        if value is None or isinstance(value, AgentAction):
            return AgentAction.from_value(value)
        merged = dict(value)
        target = merged.get("target")
        if target is not None:
            merged["target_x"] = int(target[0])
            merged["target_y"] = int(target[1])
        payload = str(merged.pop("payload", "")).strip()
        if payload:
            try:
                semantic = json.loads(payload)
            except json.JSONDecodeError:
                semantic = {"message": payload}
            if isinstance(semantic, dict):
                merged.update(semantic)
        return AgentAction.from_value(merged)

    def step(
        self,
        actions: Mapping[str, AgentAction | Mapping[str, Any] | None],
    ) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        decoded = {agent: self._decode_action(value) for agent, value in actions.items()}
        result = self.simulation.step(decoded)
        current = list(self.agents)
        observations = {
            agent: self.simulation.observation(self._agent_name_mapping[agent])
            for agent in current
        }
        rewards = {
            agent: float(result.rewards[self._agent_name_mapping[agent]]) for agent in current
        }
        terminations = {
            agent: (
                not self.config.economy.respawn_enabled
                and not bool(
                    self.simulation.population.active[self._agent_name_mapping[agent]]
                )
            )
            for agent in current
        }
        time_limit = self.simulation.tick >= self.config.simulation.max_ticks
        truncations = {agent: time_limit for agent in current}
        infos = {
            agent: {
                "text_observation": self.simulation.semantic_observation(
                    self._agent_name_mapping[agent]
                ),
                "artifact_score": result.artifact_score,
                "event_count": len(result.events),
            }
            for agent in current
        }
        if time_limit:
            self.agents = []
        elif self.config.economy.mortality_enabled and not self.config.economy.respawn_enabled:
            self.agents = [
                agent for agent in current if not terminations[agent]
            ]
        return observations, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        pop = self.simulation.population
        return np.concatenate(
            [
                pop.x.astype(np.float32) / self.config.world.width,
                pop.y.astype(np.float32) / self.config.world.height,
                pop.energy,
                pop.inventory.ravel(),
            ]
        )

    def render(self) -> dict[str, Any] | None:
        if self.render_mode == "state":
            return self.simulation.snapshot()
        return None

    def close(self) -> None:
        return None


def parallel_env(config: GameConfig, render_mode: str | None = None) -> BioFoundryParallelEnv:
    return BioFoundryParallelEnv(config=config, render_mode=render_mode)

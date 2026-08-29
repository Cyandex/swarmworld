"""Authoritative deterministic BioFoundry simulation."""

from __future__ import annotations

import json
import re
from collections import Counter, deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .artifacts import SERVICE_NAMES, ArtifactSystem
from .capabilities import available_action_types
from .config import GameConfig
from .events import digest_snapshot
from .knowledge import EmpiricalKnowledge
from .materials import (
    DEFAULT_RECIPES,
    FEEDSTOCK_COMPOSITION,
    MaterialLab,
    RecipeValidationError,
    required_inputs,
)
from .memory import AgentMemory, CulturalArchive, MemoryRecord
from .population import Population
from .program_library import ProgramLibrary
from .programs import ArtifactProgram, ProgramValidationError
from .scenarios import ScenarioMaterialLab, ScenarioWorld, load_scenario
from .science import ResearchMission
from .types import (
    PROTOCOL_VERSION,
    ActionType,
    AgentAction,
    ArtifactType,
    Resource,
    Station,
    Terrain,
    WorldEvent,
)
from .world import BioWorld


@dataclass(slots=True)
class StepResult:
    tick: int
    events: list[WorldEvent]
    rewards: NDArray[np.float32]
    artifact_score: float


class BioFoundrySimulation:
    """State machine independent of PettingZoo, LLM providers, and rendering."""

    def __init__(self, config: GameConfig):
        config.validate()
        self.config = config
        self.scenario = load_scenario(config.world.scenario_package)
        self.reset(config.simulation.seed)

    def reset(self, seed: int | None = None) -> None:
        self.seed = self.config.simulation.seed if seed is None else int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.world = (
            ScenarioWorld(self.config.world, self.rng, self.scenario)
            if self.scenario is not None
            else BioWorld(self.config.world, self.rng)
        )
        self.population = Population(
            self.config.population, self.world, self.rng, self.config.economy
        )
        self.tick = 0
        self.material_lab = (
            ScenarioMaterialLab(self.scenario)
            if self.scenario is not None
            else MaterialLab()
        )
        self.artifacts = ArtifactSystem(
            self.config.simulation.artifact_limit, self.config.physics
        )
        self.program_library = ProgramLibrary(self.population.agent_ids)
        self.message_records: dict[str, dict[str, Any]] = {}
        self.memories = [
            AgentMemory(self.config.population.memory_capacity) for _ in range(self.population.size)
        ]
        self.empirical_knowledge = [
            EmpiricalKnowledge(self.world.width) for _ in range(self.population.size)
        ]
        for index in range(self.population.size):
            self._observe_visible_cells(index)
        self.archive = CulturalArchive()
        self.deposited_insights: list[dict[str, Any]] = []
        self.research = (
            ResearchMission(self.config.science, scenario=self.scenario)
            if self.config.science.enabled
            else None
        )
        feedback_capacity = self.config.science.feedback_history
        self.action_feedback: list[deque[dict[str, Any]]] = [
            deque(maxlen=feedback_capacity) for _ in range(self.population.size)
        ]
        self.needs_replan = np.zeros(self.population.size, dtype=np.bool_)
        self._research_milestones: dict[str, bool] = {}
        self._research_completed = False
        self.last_rewards = np.zeros(self.population.size, dtype=np.float32)
        self.last_actions = np.zeros(self.population.size, dtype=np.int16)
        self.last_events: list[WorldEvent] = []
        self._record_counter = 0
        self.distance_traveled = np.zeros(self.population.size, dtype=np.int64)
        self.visited_cells: list[set[int]] = [
            {
                int(self.population.y[index]) * self.world.width
                + int(self.population.x[index])
            }
            for index in range(self.population.size)
        ]
        self.action_attempts_used = 0
        self.action_attempts_blocked = 0
        self.action_budget_exhausted_tick: int | None = None

    def _action_energy_cost(self, action: AgentAction) -> float:
        economy = self.config.economy
        if action.verb in {ActionType.WAIT, ActionType.METABOLIZE}:
            return 0.0
        if action.verb == ActionType.MOVE:
            return economy.move_cost
        if action.verb == ActionType.BUILD:
            return economy.build_cost
        if action.verb == ActionType.OPERATE:
            return economy.operate_cost
        if action.verb in {ActionType.COMMUNICATE, ActionType.TEACH}:
            return economy.communication_cost
        return economy.default_action_cost

    def _observe_visible_programs(self, index: int) -> None:
        if not self.config.science.skill_library:
            return
        x, y = int(self.population.x[index]), int(self.population.y[index])
        agent_id = self.population.agent_ids[index]
        for artifact in range(self.artifacts.count):
            if self.artifacts.retired[artifact]:
                continue
            distance = abs(int(self.artifacts.x[artifact]) - x) + abs(
                int(self.artifacts.y[artifact]) - y
            )
            program = self.artifacts.programs[artifact]
            if distance <= 4 and program is not None:
                self.program_library.observe(
                    agent_id,
                    program.program_id,
                    tick=self.tick,
                    source="local_artifact_observation",
                    evidence_id=f"artifact_{artifact:08d}",
                )

    @property
    def agent_ids(self) -> list[str]:
        return self.population.agent_ids

    def terrain_name(self, value: int | Terrain) -> str:
        if self.scenario is None:
            return Terrain(int(value)).name
        return self.scenario.identifier("terrains", int(value))

    def resource_name(self, value: int | Resource) -> str:
        if self.scenario is None:
            return Resource(int(value)).name
        return self.scenario.identifier("resources", int(value))

    def station_name(self, value: int | Station) -> str:
        if self.scenario is None:
            return Station(int(value)).name
        return self.scenario.identifier("facilities", int(value))

    def _required_inputs(self, recipe: dict[str, Any]) -> dict[Resource, float]:
        if self.scenario is None:
            return required_inputs(recipe)
        return self.material_lab.required_inputs(recipe)

    def _record_id(self, prefix: str) -> str:
        value = f"{prefix}_{self._record_counter:010d}"
        self._record_counter += 1
        return value

    def _observe_visible_cells(self, index: int) -> dict[str, list[str]]:
        """Persist raw passive evidence and report first-of-kind discoveries."""

        x = int(self.population.x[index])
        y = int(self.population.y[index])
        known_resources = self.empirical_knowledge[index].directly_observed_resources()
        known_stations = self.empirical_knowledge[index].directly_observed_stations()
        known_terrains = self.empirical_knowledge[index].directly_observed_terrains()
        discovered_resources: set[Resource] = set()
        discovered_stations: set[Station] = set()
        discovered_terrains: set[Terrain] = set()
        for dx, dy in ((0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)):
            observed_x, observed_y = x + dx, y + dy
            if not (0 <= observed_x < self.world.width and 0 <= observed_y < self.world.height):
                continue
            resource = Resource(int(self.world.resource_kind[observed_y, observed_x]))
            station = Station(int(self.world.stations[observed_y, observed_x]))
            terrain = Terrain(int(self.world.terrain[observed_y, observed_x]))
            self.empirical_knowledge[index].observe(
                x=observed_x,
                y=observed_y,
                tick=self.tick,
                terrain=terrain,
                resource=resource,
                resource_mass=float(self.world.resource_mass[observed_y, observed_x]),
                station=station,
            )
            if resource != Resource.NONE and resource not in known_resources:
                discovered_resources.add(resource)
            if station != Station.NONE and station not in known_stations:
                discovered_stations.add(station)
            if terrain not in known_terrains:
                discovered_terrains.add(terrain)
        return {
            "resources": [
                self.resource_name(item)
                for item in sorted(discovered_resources, key=int)
            ],
            "stations": [self.station_name(item) for item in sorted(discovered_stations, key=int)],
            "terrains": [self.terrain_name(item) for item in sorted(discovered_terrains, key=int)],
        }

    def grounded_resources(
        self, index: int, *, include_shared_depot: bool = False
    ) -> set[Resource]:
        """Matter supported by this agent's evidence or conserved possession."""

        depot = (
            self.research.depot
            if include_shared_depot and self.research is not None
            else None
        )
        return self.empirical_knowledge[index].grounded_resources(
            self.population.inventory[index], depot
        )

    def _require_grounded_recipe(
        self,
        index: int,
        recipe: dict[str, Any],
        *,
        cited_resources: set[Resource] | None = None,
    ) -> None:
        """Keep executable designs coupled to empirical matter without inferring beliefs."""

        required = set(self._required_inputs(recipe))
        grounded = self.grounded_resources(index) | set(cited_resources or ())
        ungrounded = sorted(required - grounded, key=int)
        if ungrounded:
            names = ", ".join(self.resource_name(resource) for resource in ungrounded)
            raise RecipeValidationError(
                "recipe inputs lack empirical grounding for this agent: "
                f"{names}; directly observe, obtain, or cite grounded evidence before "
                "submitting an executable recipe"
            )

    def _publication_grounded_resources(self, index: int, content: str) -> set[Resource]:
        """Attach only explicitly named, personally grounded matter to a publication."""

        normalized = content.casefold().replace("_", " ")
        personal = self.grounded_resources(index, include_shared_depot=False)
        return {
            resource
            for resource in personal
            if self.resource_name(resource).casefold().replace("_", " ") in normalized
        }

    def scheduled_macro_agents(self) -> list[str]:
        budget = self.config.simulation.action_attempt_budget
        if budget is not None and self.action_attempts_used >= budget:
            return []
        interval = self.config.population.macro_interval
        phase_offset = self.config.population.macro_phase_offset
        scheduled = (
            self.tick + np.arange(self.population.size) + phase_offset
        ) % interval == 0
        if (
            self.research is not None
            and self.config.science.immediate_replanning
            and self.config.science.decision_schedule == "event-driven"
        ):
            scheduled |= self.needs_replan
        indices = np.nonzero(scheduled & self.population.active)[0]
        return [self.population.agent_ids[index] for index in indices.tolist()]

    def acknowledge_macroturn(self, agent_ids: list[str]) -> None:
        for agent_id in agent_ids:
            index = self.population.id_to_index[agent_id]
            self.needs_replan[index] = False
            self.population.last_macroturn[index] = self.tick

    def step(
        self,
        actions: Mapping[str, AgentAction | Mapping[str, Any] | None],
    ) -> StepResult:
        action_list = [AgentAction() for _ in range(self.population.size)]
        for agent_id, value in actions.items():
            index = self.population.id_to_index.get(agent_id)
            if index is None or not self.population.active[index]:
                continue
            try:
                if self.scenario is not None:
                    if isinstance(value, AgentAction):
                        value = value.as_dict()
                    if isinstance(value, Mapping):
                        value = self.scenario.normalize_action_mapping(dict(value))
                action_list[index] = AgentAction.from_value(value)
            except (ValueError, TypeError, KeyError, IndexError):
                action_list[index] = AgentAction()

        events: list[WorldEvent] = []
        attempted = [
            index
            for index, action in enumerate(action_list)
            if self.population.active[index] and action.verb != ActionType.WAIT
        ]
        budget = self.config.simulation.action_attempt_budget
        admitted = attempted
        if budget is not None:
            remaining = max(0, budget - self.action_attempts_used)
            rotation = self.tick % self.population.size
            fair_order = sorted(
                attempted,
                key=lambda index: ((index - rotation) % self.population.size, index),
            )
            admitted = fair_order[:remaining]
            blocked = fair_order[remaining:]
            for index in blocked:
                original = action_list[index]
                action_list[index] = AgentAction()
                self.action_attempts_blocked += 1
                events.append(
                    WorldEvent(
                        tick=self.tick,
                        kind="action_budget_blocked",
                        payload={
                            "agent": self.population.agent_ids[index],
                            "verb": int(original.verb),
                            "budget": budget,
                        },
                    )
                )
        self.action_attempts_used += len(admitted)
        if (
            budget is not None
            and self.action_attempts_used >= budget
            and self.action_budget_exhausted_tick is None
        ):
            self.action_budget_exhausted_tick = self.tick
        rewards = np.zeros(self.population.size, dtype=np.float32)
        energy_rejections: dict[int, str] = {}
        if self.config.economy.enabled:
            for index, action in enumerate(action_list):
                if not self.population.active[index]:
                    continue
                cost = self._action_energy_cost(action)
                if float(self.population.energy[index]) + 1e-9 < cost:
                    energy_rejections[index] = (
                        f"insufficient energy: action costs {cost:.6f}, "
                        f"available {float(self.population.energy[index]):.6f}"
                    )
                    continue
                self.population.energy[index] -= np.float32(cost)
        self.last_actions = np.asarray([int(action.verb) for action in action_list], dtype=np.int16)
        directions = np.zeros(self.population.size, dtype=np.int16)
        for index, action in enumerate(action_list):
            if action.verb == ActionType.MOVE and index not in energy_rejections:
                directions[index] = int(action.direction)
        old_x, old_y, moved = self.population.apply_movement(directions, self.world)
        moved_indices = np.nonzero(moved)[0]
        if moved_indices.size:
            self.distance_traveled[moved_indices] += 1
            for index in moved_indices.tolist():
                self.visited_cells[index].add(
                    int(self.population.y[index]) * self.world.width
                    + int(self.population.x[index])
                )
            events.append(
                WorldEvent(
                    tick=self.tick,
                    kind="agents_moved",
                    payload={
                        "indices": moved_indices.tolist(),
                        "from_x": old_x[moved_indices].tolist(),
                        "from_y": old_y[moved_indices].tolist(),
                        "to_x": self.population.x[moved_indices].tolist(),
                        "to_y": self.population.y[moved_indices].tolist(),
                    },
                )
            )

        for index in np.nonzero(self.population.active)[0].tolist():
            discoveries = self._observe_visible_cells(index)
            self._observe_visible_programs(index)
            if not any(discoveries.values()):
                continue
            if self.research is not None and self.config.science.immediate_replanning:
                self.needs_replan[index] = True
            events.append(
                WorldEvent(
                    tick=self.tick,
                    kind="salient_discovery",
                    payload={
                        "agent": self.population.agent_ids[index],
                        **discoveries,
                    },
                )
            )

        communicating = any(
            action.verb in (ActionType.COMMUNICATE, ActionType.TEACH, ActionType.TRADE)
            for action in action_list
        )
        buckets = self.population.build_spatial_index(self.world.width) if communicating else {}
        executable_actions = set(available_action_types(self.config))

        for index, action in enumerate(action_list):
            if not self.population.active[index]:
                continue
            if index in energy_rejections:
                reason = energy_rejections[index]
                events.append(
                    WorldEvent(
                        tick=self.tick,
                        kind="action_rejected",
                        payload={
                            "agent": self.population.agent_ids[index],
                            "verb": int(action.verb),
                            "reason": reason,
                        },
                    )
                )
                if self.research is not None and self.config.science.action_feedback:
                    self._record_action_outcome(index, action, False, reason, events)
                if self.research is not None and self.config.science.immediate_replanning:
                    self.needs_replan[index] = True
                continue
            if action.verb == ActionType.WAIT:
                continue
            agent_id = self.population.agent_ids[index]
            x = int(self.population.x[index])
            y = int(self.population.y[index])
            event_start = len(events)
            success = True
            failure_reason = ""
            try:
                if action.verb not in executable_actions:
                    raise ValueError(
                        f"{action.verb.name} is disabled by this experimental treatment"
                    )
                if action.reply_to and action.verb not in {
                    ActionType.COMMUNICATE,
                    ActionType.TEACH,
                    ActionType.TRADE,
                }:
                    if not self.config.science.request_tracking:
                        raise ValueError(
                            "reply-linked arbitrary actions are disabled in this ablation"
                        )
                    self._validate_reply_to(index, action.reply_to)
                if action.verb == ActionType.MOVE:
                    if not bool(moved[index]):
                        raise ValueError("movement was blocked or direction was STAY")
                elif action.verb == ActionType.INSPECT:
                    self._inspect(index, events)
                elif action.verb == ActionType.HARVEST:
                    if action.resource != Resource.NONE:
                        raise ValueError(
                            "HARVEST samples the current location and requires resource=NONE"
                        )
                    free = self.population.free_capacity(index)
                    resource, amount = self.world.harvest(x, y, min(0.30, free))
                    accepted = self.population.add_resource(index, resource, amount)
                    self.empirical_knowledge[index].record_extraction(resource, accepted)
                    if accepted > 0:
                        rewards[index] += np.float32(accepted * 0.02)
                        events.append(
                            WorldEvent(
                                tick=self.tick,
                                kind="resource_harvested",
                                payload={
                                    "agent": agent_id,
                                    "resource": int(resource),
                                    "amount": round(accepted, 6),
                                    "x": x,
                                    "y": y,
                                },
                            )
                        )
                    else:
                        raise ValueError("the current location produced no collectable material")
                elif action.verb == ActionType.DEPOSIT:
                    self._deposit_resource(index, action, events)
                elif action.verb == ActionType.OPERATE:
                    self._operate(index, action, events)
                elif action.verb == ActionType.TEST:
                    self._test(index, action, events)
                elif action.verb == ActionType.PROPOSE_RECIPE:
                    self._propose_recipe(index, action, events, combined=False)
                elif action.verb == ActionType.BUILD:
                    self._build(index, action, events, rewards)
                elif action.verb == ActionType.COMMUNICATE:
                    if self.research is not None and not self.config.science.communication:
                        raise ValueError("communication is disabled in this ablation")
                    if not action.message.strip():
                        raise ValueError("COMMUNICATE requires a non-empty message")
                    candidates = self.population.neighbors(
                        index,
                        self.config.population.communication_radius,
                        self.world.width,
                        self.world.height,
                        buckets,
                    )
                    recipients = self._addressed_recipients(index, action, candidates)
                    if not recipients:
                        raise ValueError("COMMUNICATE has no nearby recipients")
                    record_id = self._record_id("message")
                    if action.reply_to:
                        self._validate_reply_to(index, action.reply_to)
                    record = MemoryRecord(
                        record_id=record_id,
                        tick=self.tick,
                        kind="message",
                        content=action.message.strip(),
                        salience=0.45,
                        causal_parents=tuple(
                            dict.fromkeys(
                                [
                                    *action.causal_parents,
                                    *([action.reply_to] if action.reply_to else []),
                                ]
                            )
                        ),
                    )
                    self.memories[index].remember(record)
                    for recipient in recipients:
                        self.memories[recipient].remember(record)
                        if (
                            self.research is not None
                            and self.config.science.immediate_replanning
                            and self.config.science.message_interrupts
                        ):
                            self.needs_replan[recipient] = True
                    self.message_records[record_id] = {
                        "record_id": record_id,
                        "tick": self.tick,
                        "sender": agent_id,
                        "content": action.message.strip(),
                        "recipients": [
                            self.population.agent_ids[item] for item in recipients
                        ],
                        "reply_to": action.reply_to,
                        "addressed": bool(action.target_agent_id),
                        "responses": [],
                        "fulfillments": [],
                    }
                    if action.reply_to:
                        self.message_records[action.reply_to]["responses"].append(
                            {
                                "tick": self.tick,
                                "agent": agent_id,
                                "message_id": record_id,
                            }
                        )
                    events.append(
                        WorldEvent(
                            tick=self.tick,
                            kind="message_delivered",
                            payload={
                                "record_id": record_id,
                                "sender": agent_id,
                                "recipients": [
                                    self.population.agent_ids[item] for item in recipients
                                ],
                                "message": action.message.strip(),
                                "reply_to": action.reply_to,
                                "addressed": bool(action.target_agent_id),
                                "x": x,
                                "y": y,
                            },
                        )
                    )
                elif action.verb in (ActionType.PUBLISH, ActionType.DEPOSIT_INSIGHT):
                    if (
                        action.verb == ActionType.PUBLISH
                        and self.research is not None
                        and not self.config.science.communication
                    ):
                        raise ValueError("public archive is disabled in this ablation")
                    self._deposit_insight(index, action, events)
                elif action.verb in (ActionType.WRITE_PROGRAM, ActionType.FORK_PROGRAM):
                    self._write_program(index, action, events)
                elif action.verb == ActionType.REPAIR:
                    artifact = self._artifact_at_or_near(
                        x,
                        y,
                        action.target_x,
                        action.target_y,
                        action.target_artifact_id,
                    )
                    if artifact is None:
                        raise ValueError("no artifact at or next to the repair target")
                    if self.artifacts.retired[artifact]:
                        raise ValueError("cannot repair a dismantled artifact")
                    needed = min(
                        0.04, 1.0 - float(self.artifacts.health[artifact])
                    )
                    healed = needed
                    if self.config.physics.closed_artifact_fluxes:
                        healed = min(needed, float(self.artifacts.reserve[artifact]))
                        if healed <= 1e-9:
                            raise ValueError("artifact has no reserve available for repair")
                        self.artifacts.reserve[artifact] -= np.float32(healed)
                        self.artifacts.fluxes[artifact, 3] += healed
                    if healed <= 1e-9:
                        raise ValueError("artifact health is already full")
                    self.artifacts.health[artifact] += np.float32(healed)
                    rewards[index] += np.float32(0.02)
                    events.append(
                        WorldEvent(
                            tick=self.tick,
                            kind="artifact_repaired",
                            payload={
                                "agent": agent_id,
                                "artifact": f"artifact_{artifact:08d}",
                                "health_restored": round(healed, 6),
                                "reserve_remaining": round(
                                    float(self.artifacts.reserve[artifact]), 6
                                ),
                            },
                        )
                    )
                elif action.verb == ActionType.DISMANTLE:
                    artifact = self._artifact_at_or_near(
                        x,
                        y,
                        action.target_x,
                        action.target_y,
                        action.target_artifact_id,
                    )
                    if artifact is None:
                        raise ValueError("no artifact at or next to the dismantle target")
                    recovered = self._dismantle(index, artifact)
                    events.append(
                        WorldEvent(
                            tick=self.tick,
                            kind="artifact_dismantled",
                            payload={
                                "agent": agent_id,
                                "artifact": f"artifact_{artifact:08d}",
                                "recovered": recovered,
                            },
                        )
                    )
                elif action.verb == ActionType.TRADE:
                    self._trade(index, action, buckets, events)
                elif action.verb == ActionType.CLAIM_TASK:
                    self._claim_task(index, action, events)
                elif action.verb == ActionType.TEACH:
                    if self.research is not None and not self.config.science.communication:
                        raise ValueError("teaching is disabled in this ablation")
                    self._teach(index, action, buckets, events)
                elif action.verb == ActionType.COMBINE_DESIGN:
                    self._propose_recipe(index, action, events, combined=True)
                elif action.verb == ActionType.METABOLIZE:
                    self._metabolize(index, action, events)
                else:
                    self._record_semantic_action(index, action, events)
            except (RecipeValidationError, ProgramValidationError, RuntimeError, ValueError) as exc:
                success = False
                failure_reason = str(exc)[:240]
                events.append(
                    WorldEvent(
                        tick=self.tick,
                        kind="action_rejected",
                        payload={
                            "agent": agent_id,
                            "verb": int(action.verb),
                            "reason": failure_reason,
                        },
                    )
                )
            if success and action.verb != ActionType.COMMUNICATE:
                self._record_request_fulfillment(
                    index, action, events, event_start=event_start
                )
            if self.research is not None and self.config.science.action_feedback:
                if action.verb == ActionType.MOVE and success:
                    detail = f"moved from ({int(old_x[index])},{int(old_y[index])}) to ({x},{y})"
                elif success:
                    detail = ", ".join(event.kind for event in events[event_start:])
                    if not detail:
                        success = False
                        failure_reason = "action produced no observable state change"
                        detail = failure_reason
                else:
                    detail = failure_reason
                self._record_action_outcome(index, action, success, detail, events)
            replanning_verbs = (
                {ActionType.TEST, ActionType.BUILD}
                if self.config.science.selective_replanning
                else {
                    ActionType.OPERATE,
                    ActionType.TEST,
                    ActionType.PUBLISH,
                    ActionType.BUILD,
                }
            )
            if (
                success
                and self.research is not None
                and self.config.science.immediate_replanning
                and action.verb in replanning_verbs
            ):
                self.needs_replan[index] = True
                events.append(
                    WorldEvent(
                        tick=self.tick,
                        kind="replan_requested",
                        payload={
                            "agent": agent_id,
                            "reason": (
                                "new experimental state from OPERATE"
                                if action.verb == ActionType.OPERATE
                                else f"new evidence from {action.verb.name}"
                            ),
                        },
                    )
                )

        disturbances = self.world.step(self.tick)
        for disturbance in disturbances:
            events.append(
                WorldEvent(
                    tick=self.tick,
                    kind="environmental_disturbance",
                    payload=dict(disturbance),
                )
            )
        if (
            disturbances
            and self.research is not None
            and self.config.science.immediate_replanning
        ):
            affected = self.world.last_disturbance_mask[
                self.population.y, self.population.x
            ]
            self.needs_replan |= affected & self.population.active
        artifact_events = self.artifacts.step(self.world, self.tick)
        events.extend(artifact_events)
        if self.config.economy.enabled:
            self._apply_economy_turnover(events)
        if self.research is not None:
            self._update_research_milestones(events)
        self.tick += 1
        self.last_rewards = rewards
        self.last_events = events
        return StepResult(
            tick=self.tick,
            events=events,
            rewards=rewards,
            artifact_score=self.artifacts.summary_score(),
        )

    def _addressed_recipients(
        self, index: int, action: AgentAction, candidates: list[int]
    ) -> list[int]:
        if not action.target_agent_id:
            return candidates
        if not self.config.science.addressed_communication:
            raise ValueError("addressed communication is disabled in this ablation")
        target = self.population.id_to_index.get(action.target_agent_id)
        if target is None or target not in candidates:
            raise ValueError("target_agent_id is not a locally reachable agent")
        if target == index:
            raise ValueError("an agent cannot address itself")
        return [target]

    def _validate_reply_to(self, index: int, reply_to: str) -> dict[str, Any]:
        prior = self.message_records.get(reply_to)
        if prior is None:
            raise ValueError("reply_to does not identify a delivered message")
        agent_id = self.population.agent_ids[index]
        participants = {str(prior["sender"]), *map(str, prior["recipients"])}
        if agent_id not in participants:
            raise ValueError("reply_to message was not visible to this agent")
        return prior

    def _pending_requests(self, index: int, limit: int = 6) -> list[dict[str, Any]]:
        """Expose unresolved addressed work as state, without interpreting its prose."""

        agent_id = self.population.agent_ids[index]
        pending: list[dict[str, Any]] = []
        for message in reversed(list(self.message_records.values())):
            if agent_id not in message.get("recipients", []):
                continue
            fulfilled_by_agent = any(
                item.get("agent") == agent_id
                for item in message.get("fulfillments", [])
            )
            if fulfilled_by_agent:
                continue
            pending.append(
                {
                    "message_id": message["record_id"],
                    "sender": message["sender"],
                    "tick": int(message["tick"]),
                    "age": self.tick - int(message["tick"]),
                    "content": str(message.get("content", ""))[:500],
                    "responded": any(
                        item.get("agent") == agent_id
                        for item in message.get("responses", [])
                    ),
                }
            )
            if len(pending) >= limit:
                break
        return pending

    def _record_request_fulfillment(
        self,
        index: int,
        action: AgentAction,
        events: list[WorldEvent],
        *,
        event_start: int,
    ) -> None:
        """Attach a successful arbitrary action to an agent-selected request edge."""

        if not action.reply_to:
            return
        prior = self._validate_reply_to(index, action.reply_to)
        result_ids = sorted(
            {
                str(value)
                for event in events[event_start:]
                for key, value in event.payload.items()
                if key.endswith("_id") and isinstance(value, str) and value
            }
        )
        fulfillment_record_id = self._record_id("fulfillment")
        fulfillment = {
            "tick": self.tick,
            "agent": self.population.agent_ids[index],
            "verb": action.verb.name,
            "causal_parents": list(action.causal_parents),
            "result_ids": result_ids,
            "fulfillment_record_id": fulfillment_record_id,
        }
        prior["fulfillments"].append(fulfillment)
        receipt = MemoryRecord(
            record_id=fulfillment_record_id,
            tick=self.tick,
            kind="request_fulfillment",
            content=json.dumps(
                {"message_id": action.reply_to, **fulfillment}, sort_keys=True
            ),
            salience=0.75,
            causal_parents=tuple(
                dict.fromkeys([action.reply_to, *action.causal_parents, *result_ids])
            ),
        )
        sender_index = self.population.id_to_index.get(str(prior["sender"]))
        if sender_index is not None:
            self.memories[sender_index].remember(receipt, notebook=True)
            for result_id in result_ids:
                result_record = self.memories[index].get(result_id)
                if result_record is not None:
                    self.memories[sender_index].remember(result_record, notebook=True)
        events.append(
            WorldEvent(
                tick=self.tick,
                kind="request_fulfilled",
                payload={"message_id": action.reply_to, **fulfillment},
            )
        )

    def _metabolize(
        self, index: int, action: AgentAction, events: list[WorldEvent]
    ) -> None:
        if not self.config.economy.enabled:
            raise ValueError("metabolism is disabled in this experimental condition")
        if action.resource == Resource.NONE:
            raise ValueError("METABOLIZE requires a personally held resource")
        composition = FEEDSTOCK_COMPOSITION.get(action.resource)
        if composition is None:
            raise ValueError("the selected resource has no metabolic composition")
        organic_fraction = float(composition[[0, 1, 2, 4]].sum())
        if organic_fraction < 0.20:
            raise ValueError("the selected resource has insufficient metabolizable matter")
        available = float(self.population.inventory[index, int(action.resource)])
        requested = action.amount if action.amount > 0 else 0.25
        energy_per_mass = organic_fraction * self.config.economy.metabolize_efficiency
        deficit = max(
            0.0,
            self.config.economy.maximum_energy - float(self.population.energy[index]),
        )
        consumed = min(available, requested, deficit / max(1e-12, energy_per_mass))
        if consumed <= 1e-9:
            raise ValueError("no metabolic mass can be consumed or energy is already full")
        gained = consumed * energy_per_mass
        self.population.inventory[index, int(action.resource)] -= np.float32(consumed)
        self.population.energy[index] = np.float32(
            min(
                self.config.economy.maximum_energy,
                float(self.population.energy[index]) + gained,
            )
        )
        self.world.flux_ledger["metabolic_mass_consumed"] = (
            self.world.flux_ledger.get("metabolic_mass_consumed", 0.0) + consumed
        )
        events.append(
            WorldEvent(
                tick=self.tick,
                kind="resource_metabolized",
                payload={
                    "agent": self.population.agent_ids[index],
                    "resource": int(action.resource),
                    "mass": round(consumed, 6),
                    "energy_gained": round(gained, 6),
                    "energy": round(float(self.population.energy[index]), 6),
                },
            )
        )

    def _dismantle(self, index: int, artifact: int) -> dict[str, float]:
        if self.artifacts.retired[artifact]:
            raise ValueError("artifact has already been dismantled")
        recovered: dict[str, float] = {}
        recovered_mass = 0.0
        batch = self.artifacts.provenance[artifact].get("batch", {})
        batch_mass = float(batch.get("mass", 0.0))
        if self.config.physics.dismantle_recovery:
            recipe = batch.get("recipe", {})
            required = self._required_inputs(recipe)
            input_mass = sum(required.values())
            retained_ratio = float(batch.get("mass", 0.0)) / max(1e-12, input_mass)
            factor = (
                retained_ratio
                * float(self.artifacts.health[artifact])
                * self.config.physics.dismantle_efficiency
            )
            for resource, mass in sorted(required.items(), key=lambda item: int(item[0])):
                accepted = self.population.add_resource(index, resource, mass * factor)
                if accepted > 0:
                    recovered[self.resource_name(resource)] = round(accepted, 6)
                    recovered_mass += accepted
        self.world.flux_ledger["dismantled_mass_recovered"] += recovered_mass
        self.world.flux_ledger["dismantled_mass_discarded"] += max(
            0.0, batch_mass - recovered_mass
        )
        self.artifacts.retired[artifact] = True
        self.artifacts.health[artifact] = np.float32(0.0)
        self.artifacts.performance[artifact] = np.float32(0.0)
        return recovered

    def _apply_economy_turnover(self, events: list[WorldEvent]) -> None:
        economy = self.config.economy
        active = self.population.active
        self.population.energy[active] = np.maximum(
            0.0,
            self.population.energy[active] - np.float32(economy.passive_cost),
        )
        if economy.mortality_enabled:
            dying = np.nonzero(active & (self.population.energy <= 1e-9))[0]
            for index in dying.tolist():
                self.population.active[index] = False
                self.population.death_tick[index] = self.tick
                events.append(
                    WorldEvent(
                        tick=self.tick,
                        kind="agent_died",
                        payload={
                            "agent": self.population.agent_ids[index],
                            "generation": int(self.population.generation[index]),
                        },
                    )
                )
        if not economy.respawn_enabled:
            return
        due = np.nonzero(
            (~self.population.active)
            & (self.population.death_tick >= 0)
            & ((self.tick - self.population.death_tick) >= economy.respawn_delay)
        )[0]
        for index in due.tolist():
            agent_id = self.population.agent_ids[index]
            if economy.cultural_inheritance and self.config.science.skill_library:
                inherited = self.program_library.inherit(
                    agent_id, limit=economy.inherited_skill_limit, tick=self.tick
                )
            else:
                self.program_library.clear_agent(agent_id)
                inherited = []
            self.memories[index] = AgentMemory(self.config.population.memory_capacity)
            self.empirical_knowledge[index] = EmpiricalKnowledge(self.world.width)
            self.action_feedback[index].clear()
            self.population.energy[index] = np.float32(economy.initial_energy)
            self.population.active[index] = True
            self.population.generation[index] += 1
            self.population.death_tick[index] = -1
            self._observe_visible_cells(index)
            self.needs_replan[index] = True
            events.append(
                WorldEvent(
                    tick=self.tick,
                    kind="agent_respawned",
                    payload={
                        "agent": agent_id,
                        "generation": int(self.population.generation[index]),
                        "inherited_program_ids": inherited,
                    },
                )
            )

    def _record_action_outcome(
        self,
        index: int,
        action: AgentAction,
        success: bool,
        detail: str,
        events: list[WorldEvent],
    ) -> None:
        agent_id = self.population.agent_ids[index]
        self.memories[index].mark_causal_outcome(
            set(map(str, action.causal_parents)), success=success
        )
        outcome = {
            "tick": self.tick,
            "agent": agent_id,
            "verb": int(action.verb),
            "action": action.verb.name,
            "success": success,
            "detail": detail[:500],
            "causal_parents": list(map(str, action.causal_parents)),
            "reply_to": action.reply_to,
        }
        intent = action.message.strip()
        if intent:
            outcome["intent"] = intent[:500]
        self.action_feedback[index].append(outcome)
        if not success and self.config.science.immediate_replanning:
            self.needs_replan[index] = True
        intent_suffix = f"; intended: {intent[:300]}" if intent else ""
        record = MemoryRecord(
            record_id=self._record_id("outcome"),
            tick=self.tick,
            kind="action_result",
            content=(
                f"{action.verb.name} {'succeeded' if success else 'failed'}: "
                f"{detail[:400]}{intent_suffix}"
            ),
            salience=0.75 if not success else 0.55,
        )
        self.memories[index].remember(record)
        events.append(WorldEvent(tick=self.tick, kind="action_result", payload=outcome))

    def _inspect(self, index: int, events: list[WorldEvent]) -> None:
        x, y = int(self.population.x[index]), int(self.population.y[index])
        self._observe_visible_cells(index)
        agent_id = self.population.agent_ids[index]
        record_id = self._record_id("observation")
        observation = {
            "record_id": record_id,
            "agent": agent_id,
            "x": x,
            "y": y,
            "terrain": self.terrain_name(int(self.world.terrain[y, x])),
            "station": self.station_name(int(self.world.stations[y, x])),
            "resource": self.resource_name(int(self.world.resource_kind[y, x])),
            "resource_mass": round(float(self.world.resource_mass[y, x]), 6),
            "moisture": round(float(self.world.moisture[y, x]), 6),
            "nutrients": round(float(self.world.nutrients[y, x]), 6),
            "temperature": round(float(self.world.temperature[y, x]), 6),
            "solar": round(float(self.world.solar[y, x]), 6),
            "contamination": round(float(self.world.contamination[y, x]), 6),
            "artifact_measurements": [],
        }
        for artifact in range(self.artifacts.count):
            distance = abs(int(self.artifacts.x[artifact]) - x) + abs(
                int(self.artifacts.y[artifact]) - y
            )
            program = self.artifacts.programs[artifact]
            if distance > 1 or program is None or self.artifacts.retired[artifact]:
                continue
            measurement = {
                "artifact_id": f"artifact_{artifact:08d}",
                "program_id": program.program_id,
                "performance": round(float(self.artifacts.performance[artifact]), 6),
                "services": {
                    service: round(
                        float(self.artifacts.services[artifact, service_index]), 6
                    )
                    for service_index, service in enumerate(SERVICE_NAMES)
                },
            }
            observation["artifact_measurements"].append(measurement)
            if self.config.science.skill_library:
                self.program_library.observe(
                    agent_id,
                    program.program_id,
                    tick=self.tick,
                    source="measured_artifact",
                    evidence_id=record_id,
                    verified=True,
                    measurements=measurement,
                )
        record = MemoryRecord(
            record_id=record_id,
            tick=self.tick,
            kind="observation",
            content=json.dumps(observation, sort_keys=True),
            salience=0.62,
        )
        self.memories[index].remember(record, notebook=True)
        if self.research is not None:
            self.research.record_inspection(observation)
        events.append(WorldEvent(tick=self.tick, kind="sample_inspected", payload=observation))

    def _deposit_resource(
        self,
        index: int,
        action: AgentAction,
        events: list[WorldEvent],
    ) -> None:
        if self.research is None:
            self._record_semantic_action(index, action, events)
            return
        if not self.config.science.shared_depot:
            raise ValueError("shared depot is disabled in this ablation")
        if action.resource == Resource.NONE:
            raise ValueError("DEPOSIT requires a nonzero resource")
        x, y = int(self.population.x[index]), int(self.population.y[index])
        if (
            Terrain(int(self.world.terrain[y, x])) != Terrain.FOUNDRY
            and Station(int(self.world.stations[y, x])) == Station.NONE
        ):
            raise ValueError("DEPOSIT requires a foundry or station tile")
        available = float(self.population.inventory[index, int(action.resource)])
        requested = action.amount if action.amount > 0.0 else available
        amount = min(available, requested)
        if amount <= 1e-7:
            raise ValueError(
                f"agent carries no {self.resource_name(action.resource)} to deposit"
            )
        self.population.inventory[index, int(action.resource)] -= np.float32(amount)
        agent_id = self.population.agent_ids[index]
        self.research.deposit(agent_id, action.resource, amount)
        events.append(
            WorldEvent(
                tick=self.tick,
                kind="resource_deposited",
                payload={
                    "agent": agent_id,
                    "resource": int(action.resource),
                    "amount": round(amount, 6),
                    "depot_total": round(float(self.research.depot[int(action.resource)]), 6),
                    "x": x,
                    "y": y,
                },
            )
        )

    def _propose_recipe(
        self,
        index: int,
        action: AgentAction,
        events: list[WorldEvent],
        *,
        combined: bool,
    ) -> None:
        if self.research is None:
            self._record_semantic_action(index, action, events)
            return
        if action.recipe is None:
            raise RecipeValidationError("recipe proposal requires a recipe")
        self.material_lab.validate(action.recipe)
        if combined:
            parents = sorted(set(action.causal_parents))
            if len(parents) < 2:
                raise RecipeValidationError(
                    "COMBINE_DESIGN requires at least two causal parent records"
                )
            missing = [parent for parent in parents if parent not in self.archive.by_id]
            if missing:
                raise RecipeValidationError(
                    f"COMBINE_DESIGN references unpublished records: {missing}"
                )
            missing_provenance = [
                parent for parent in parents if parent not in self.research.publications
            ]
            if missing_provenance:
                raise RecipeValidationError(
                    "COMBINE_DESIGN references records without structured publication "
                    f"provenance: {missing_provenance}"
                )
            parent_authors = sorted(
                {
                    str(self.research.publications[parent]["author"])
                    for parent in parents
                }
            )
            if len(parent_authors) < self.config.science.minimum_contributors:
                raise RecipeValidationError(
                    "COMBINE_DESIGN requires published evidence from at least "
                    f"{self.config.science.minimum_contributors} distinct agents"
                )
            grounded_parent_authors = {
                str(self.research.publications[parent]["author"])
                for parent in parents
                if self.research.publications[parent]["grounded_resources"]
            }
            if len(grounded_parent_authors) < self.config.science.minimum_contributors:
                raise RecipeValidationError(
                    "COMBINE_DESIGN requires empirically grounded publications from at "
                    f"least {self.config.science.minimum_contributors} distinct agents"
                )
        else:
            parent_authors = []
        cited_resources = (
            {
                Resource[str(resource)]
                for parent in sorted(set(action.causal_parents))
                for resource in self.research.publications[parent]["grounded_resources"]
            }
            if combined
            else set()
        )
        own_grounded = self.grounded_resources(index)
        inherited_resources = sorted(
            (set(self._required_inputs(action.recipe)) - own_grounded) & cited_resources,
            key=int,
        )
        self._require_grounded_recipe(
            index,
            action.recipe,
            cited_resources=cited_resources,
        )
        agent_id = self.population.agent_ids[index]
        proposal = self.research.propose(
            agent_id,
            action.recipe,
            self.tick,
            action.causal_parents,
            combined=combined,
            parent_authors=parent_authors,
            inherited_resources=[self.resource_name(resource) for resource in inherited_resources],
        )
        record = MemoryRecord(
            record_id=self._record_id("design"),
            tick=self.tick,
            kind="combined_design" if combined else "recipe_proposal",
            content=(
                f"{proposal['recipe_id']} validated; combined={combined}; "
                f"recipe={json.dumps(action.recipe, sort_keys=True)}"
            ),
            salience=0.78,
            causal_parents=tuple(action.causal_parents),
        )
        self.memories[index].remember(record, notebook=True)
        events.append(
            WorldEvent(
                tick=self.tick,
                kind="design_combined" if combined else "recipe_proposed",
                payload={**proposal, "record_id": record.record_id},
            )
        )

    def _execute_recipe_from_pool(
        self,
        index: int,
        recipe: dict[str, Any],
        causal_parents: list[str],
    ):
        self.material_lab.validate(recipe)
        inventory = self.population.inventory[index]
        if self.research is None:
            return self.material_lab.execute(
                recipe,
                inventory,
                tick=self.tick,
                contributors=[self.population.agent_ids[index]],
                causal_parents=causal_parents,
            )
        available = (
            self.research.available(inventory)
            if self.config.science.shared_depot
            else inventory.copy()
        )
        required = self._required_inputs(recipe)
        if any(
            float(available[int(resource)]) + 1e-7 < mass for resource, mass in required.items()
        ):
            raise RecipeValidationError("combined personal and depot inventory is insufficient")
        agent_id = self.population.agent_ids[index]
        contributors = [agent_id]
        synthetic = available.copy()
        batch = self.material_lab.execute(
            recipe,
            synthetic,
            tick=self.tick,
            contributors=contributors,
            causal_parents=causal_parents,
        )
        for resource, mass in required.items():
            personal = min(float(inventory[int(resource)]), mass)
            inventory[int(resource)] -= np.float32(personal)
            depot_amount = mass - personal
            if depot_amount > 1e-7 and self.config.science.shared_depot:
                contributors.extend(self.research.consume_depot(resource, depot_amount))
            elif depot_amount > 1e-7:
                raise RecipeValidationError("personal inventory is insufficient")
        batch.contributors = sorted(set(contributors))
        return batch

    def _operate(
        self,
        index: int,
        action: AgentAction,
        events: list[WorldEvent],
    ) -> None:
        if self.research is None:
            self._record_semantic_action(index, action, events)
            return
        if action.recipe is None:
            raise RecipeValidationError("OPERATE requires a complete recipe")
        x, y = int(self.population.x[index]), int(self.population.y[index])
        if (
            Terrain(int(self.world.terrain[y, x])) != Terrain.FOUNDRY
            and Station(int(self.world.stations[y, x])) == Station.NONE
        ):
            raise RecipeValidationError("OPERATE requires a foundry or station tile")
        if self.scenario is not None:
            supported, missing = self.scenario.workspace_supports(
                int(self.world.terrain[y, x]),
                int(self.world.stations[y, x]),
                action.recipe,
            )
            if not supported:
                raise RecipeValidationError(
                    "current workspace lacks operation capabilities: "
                    + ", ".join(missing)
                )
        self.material_lab.validate(action.recipe)
        scaled = self.research.scaled_recipe(action.recipe)
        batch = self._execute_recipe_from_pool(index, scaled, action.causal_parents)
        staged = self.research.stage_batch(
            self.population.agent_ids[index], action.recipe, batch
        )
        if self.scenario is not None:
            staged["workspace"] = {
                "x": x,
                "y": y,
                "terrain": self.terrain_name(int(self.world.terrain[y, x])),
                "facility": self.station_name(int(self.world.stations[y, x])),
            }
        events.append(WorldEvent(tick=self.tick, kind="microbatch_fabricated", payload=staged))

    def _test(
        self,
        index: int,
        action: AgentAction,
        events: list[WorldEvent],
    ) -> None:
        if self.research is None:
            self._record_semantic_action(index, action, events)
            return
        agent_id = self.population.agent_ids[index]
        batch = self.research.latest_untested_batch(agent_id)
        if batch is None:
            raise ValueError("TEST requires an untested microbatch produced by OPERATE")
        result = self.research.test_batch(batch, self.tick)
        observable_result = {
            key: value
            for key, value in result.items()
            if key != "passes_material_target"
        }
        record = MemoryRecord(
            record_id=self._record_id("test"),
            tick=self.tick,
            kind="experiment_result",
            content=json.dumps(observable_result, sort_keys=True),
            salience=0.90,
        )
        self.memories[index].remember(record, notebook=True)
        events.append(
            WorldEvent(
                tick=self.tick,
                kind="material_tested",
                payload={**result, "record_id": record.record_id},
            )
        )

    def _claim_task(
        self,
        index: int,
        action: AgentAction,
        events: list[WorldEvent],
    ) -> None:
        if self.research is None:
            self._record_semantic_action(index, action, events)
            return
        if not self.config.science.communication:
            raise ValueError("the public task board is disabled in this ablation")
        task = action.message.strip()
        if not task:
            raise ValueError("CLAIM_TASK requires a task description in message")
        agent_id = self.population.agent_ids[index]
        self.research.claim(agent_id, task)
        events.append(
            WorldEvent(
                tick=self.tick,
                kind="task_claimed",
                payload={"agent": agent_id, "task": task[:160]},
            )
        )

    def _teach(
        self,
        index: int,
        action: AgentAction,
        buckets: dict[int, list[int]],
        events: list[WorldEvent],
    ) -> None:
        if action.reply_to:
            self._validate_reply_to(index, action.reply_to)
        if not buckets:
            buckets = self.population.build_spatial_index(self.world.width)
        candidates = self.population.neighbors(
            index,
            self.config.population.communication_radius,
            self.world.width,
            self.world.height,
            buckets,
        )
        recipients = self._addressed_recipients(index, action, candidates)
        if not recipients:
            raise ValueError("TEACH has no nearby recipients")
        records: list[MemoryRecord] = []
        program_ids: list[str] = []
        for parent in action.causal_parents:
            record = self.memories[index].get(parent) or self.archive.by_id.get(parent)
            if record is not None:
                records.append(record)
            if self.config.science.skill_library and self.program_library.knows(
                self.population.agent_ids[index], parent
            ):
                program_ids.append(parent)
        if not records and action.message.strip():
            records.append(
                MemoryRecord(
                    record_id=self._record_id("teaching"),
                    tick=self.tick,
                    kind="teaching",
                    content=action.message.strip(),
                    salience=0.70,
                )
            )
        if not records and not program_ids:
            raise ValueError("TEACH requires retained record/program IDs or a message")
        transfer_evidence = (
            action.reply_to
            or (records[0].record_id if records else self._record_id("teaching_program"))
        )
        for recipient in recipients:
            for record in records:
                self.memories[recipient].remember(record, notebook=True)
            for program_id in program_ids:
                self.program_library.transfer(
                    self.population.agent_ids[index],
                    self.population.agent_ids[recipient],
                    program_id,
                    tick=self.tick,
                    evidence_id=transfer_evidence,
                )
            if (
                self.research is not None
                and self.config.science.immediate_replanning
                and self.config.science.message_interrupts
            ):
                self.needs_replan[recipient] = True
        events.append(
            WorldEvent(
                tick=self.tick,
                kind="knowledge_taught",
                payload={
                    "teacher": self.population.agent_ids[index],
                    "recipients": [self.population.agent_ids[item] for item in recipients],
                    "records": [record.record_id for record in records],
                    "program_ids": sorted(set(program_ids)),
                    "reply_to": action.reply_to,
                },
            )
        )
    def _update_research_milestones(self, events: list[WorldEvent]) -> None:
        if self.research is None:
            return
        card = self.research.scorecard(self.artifacts, len(self.archive.records))
        milestones = card["milestones"]
        for name, reached in milestones.items():
            if reached and not self._research_milestones.get(name, False):
                events.append(
                    WorldEvent(
                        tick=self.tick,
                        kind="research_milestone",
                        payload={"milestone": name, "score": card["score"]},
                    )
                )
        if card["completed"] and not self._research_completed:
            events.append(
                WorldEvent(
                    tick=self.tick,
                    kind="research_mission_completed",
                    payload=card,
                )
            )
        self._research_milestones = dict(milestones)
        self._research_completed = bool(card["completed"])

    def _build(
        self,
        index: int,
        action: AgentAction,
        events: list[WorldEvent],
        rewards: NDArray[np.float32],
    ) -> None:
        if action.artifact == ArtifactType.NONE:
            raise RecipeValidationError("BUILD requires a nonzero artifact type")
        if self.research is not None and action.recipe is None:
            raise RecipeValidationError(
                "BUILD requires an explicit tested recipe in collective-science mode"
            )
        if self.research is not None and action.artifact_spec is None:
            raise RecipeValidationError(
                "BUILD requires an agent-authored artifact_spec in collective-science mode"
            )
        recipe = action.recipe or DEFAULT_RECIPES[action.artifact]
        agent_id = self.population.agent_ids[index]
        batch = self._execute_recipe_from_pool(index, recipe, action.causal_parents)
        program = (
            ArtifactProgram.from_dict(action.program, author=agent_id)
            if action.program is not None
            else None
        )
        artifact = self.artifacts.add(
            action.artifact,
            int(self.population.x[index]),
            int(self.population.y[index]),
            creator=agent_id,
            batch=batch,
            tick=self.tick,
            program=program,
            artifact_spec=action.artifact_spec,
        )
        installed_program = self.artifacts.programs[artifact]
        if installed_program is not None:
            self.program_library.register(
                installed_program,
                tick=self.tick,
                artifact_id=f"artifact_{artifact:08d}",
                index_skill=self.config.science.skill_library,
            )
        spec = self.artifacts.specs[artifact]
        rewards[index] += np.float32(0.5)
        events.append(
            WorldEvent(
                tick=self.tick,
                kind="artifact_built",
                payload={
                    "artifact_id": f"artifact_{artifact:08d}",
                    "artifact_type": int(action.artifact),
                    "artifact_name": spec["name"],
                    "artifact_spec": spec,
                    "agent": agent_id,
                    "x": int(self.population.x[index]),
                    "y": int(self.population.y[index]),
                    "batch": batch.as_dict(),
                    "program": installed_program.as_dict() if installed_program else None,
                },
            )
        )

    def _deposit_insight(
        self,
        index: int,
        action: AgentAction,
        events: list[WorldEvent],
    ) -> None:
        agent_id = self.population.agent_ids[index]
        insight = dict(action.insight or {})
        content = str(insight.get("content", action.message)).strip()
        if not content:
            raise ValueError("insight content cannot be empty")
        record = MemoryRecord(
            record_id=self._record_id("insight"),
            tick=self.tick,
            kind=str(insight.get("kind", "insight"))[:40],
            content=f"{agent_id} | {content[:1000]}",
            salience=float(np.clip(insight.get("salience", 0.7), 0.0, 1.0)),
            causal_parents=tuple(action.causal_parents),
        )
        self.memories[index].remember(record, notebook=True)
        x, y = int(self.population.x[index]), int(self.population.y[index])
        spatial = {
            **record.as_dict(),
            "author": agent_id,
            "x": x,
            "y": y,
            "published": action.verb == ActionType.PUBLISH,
        }
        self.deposited_insights.append(spatial)
        if (
            self.research is not None
            and self.config.science.immediate_replanning
            and self.config.science.message_interrupts
        ):
            distances = np.abs(self.population.x - x) + np.abs(self.population.y - y)
            nearby = np.nonzero(
                (distances <= self.config.population.communication_radius)
                & self.population.active
            )[0]
            for recipient in nearby.tolist():
                if recipient != index:
                    self.needs_replan[recipient] = True
        if action.verb == ActionType.PUBLISH:
            self.archive.publish(record)
            if self.research is not None:
                self.research.register_publication(
                    record.record_id,
                    author=agent_id,
                    tick=self.tick,
                    grounded_resources=self._publication_grounded_resources(index, content),
                    causal_parents=action.causal_parents,
                )
        events.append(WorldEvent(tick=self.tick, kind="insight_deposited", payload=spatial))

    def _write_program(
        self,
        index: int,
        action: AgentAction,
        events: list[WorldEvent],
    ) -> None:
        if action.program is None:
            raise ProgramValidationError("WRITE_PROGRAM requires a program")
        x, y = int(self.population.x[index]), int(self.population.y[index])
        artifact = self._artifact_at_or_near(
            x,
            y,
            action.target_x,
            action.target_y,
            action.target_artifact_id,
        )
        if artifact is None:
            raise ProgramValidationError("no artifact at or next to the target")
        if self.artifacts.retired[artifact]:
            raise ProgramValidationError("cannot install a program on a dismantled artifact")
        agent_id = self.population.agent_ids[index]
        program = ArtifactProgram.from_dict(action.program, author=agent_id)
        parent: ArtifactProgram | None = None
        if action.verb == ActionType.FORK_PROGRAM:
            if not self.config.science.program_forking:
                raise ProgramValidationError("program forking is disabled in this ablation")
            parent_id = program.parent_program or ""
            if not parent_id:
                raise ProgramValidationError("FORK_PROGRAM requires parent_program=program_id")
            installed_parent = self.artifacts.programs[artifact]
            parent_known = (
                self.program_library.knows(agent_id, parent_id)
                if self.config.science.skill_library
                else installed_parent is not None
                and installed_parent.program_id == parent_id
            )
            if not parent_known:
                raise ProgramValidationError(
                    "fork parent is not personally observed, authored, taught, or inherited"
                )
            parent = self.program_library.resolve(parent_id)
            if parent is None:
                raise ProgramValidationError("fork parent program_id is unknown")
            if parent.program_id == program.program_id:
                raise ProgramValidationError("fork must change at least one instruction")
            if self.program_library.would_create_cycle(
                parent.program_id, program.program_id
            ):
                raise ProgramValidationError("fork would create a cyclic program lineage")
        elif program.parent_program:
            raise ProgramValidationError(
                "WRITE_PROGRAM cannot claim ancestry; use FORK_PROGRAM"
            )
        self.artifacts.install_program(artifact, program, self.tick)
        lineage = self.program_library.register(
            program,
            tick=self.tick,
            artifact_id=f"artifact_{artifact:08d}",
            parent=parent,
            index_skill=self.config.science.skill_library,
        )
        events.append(
            WorldEvent(
                tick=self.tick,
                kind="artifact_program_installed",
                payload={
                    "artifact_id": f"artifact_{artifact:08d}",
                    "agent": agent_id,
                    "program": program.as_dict(),
                    "lineage": lineage,
                },
            )
        )

    def _artifact_at_or_near(
        self,
        x: int,
        y: int,
        target_x: int,
        target_y: int,
        target_artifact_id: str = "",
    ) -> int | None:
        if self.artifacts.count == 0:
            return None
        if target_artifact_id:
            prefix = "artifact_"
            if not target_artifact_id.startswith(prefix):
                return None
            try:
                selected = int(target_artifact_id[len(prefix) :])
            except ValueError:
                return None
            if not 0 <= selected < self.artifacts.count:
                return None
            distance = abs(int(self.artifacts.x[selected]) - x) + abs(
                int(self.artifacts.y[selected]) - y
            )
            return selected if distance <= 1 else None
        tx = x if target_x < 0 else target_x
        ty = y if target_y < 0 else target_y
        distance = np.abs(self.artifacts.x[: self.artifacts.count] - tx) + np.abs(
            self.artifacts.y[: self.artifacts.count] - ty
        )
        nearest = int(np.argmin(distance))
        return nearest if int(distance[nearest]) <= 1 else None

    def _trade(
        self,
        index: int,
        action: AgentAction,
        buckets: dict[int, list[int]],
        events: list[WorldEvent],
    ) -> None:
        if action.reply_to:
            self._validate_reply_to(index, action.reply_to)
        if not buckets:
            buckets = self.population.build_spatial_index(self.world.width)
        neighbors = self.population.neighbors(
            index, 1, self.world.width, self.world.height, buckets
        )
        if not neighbors:
            raise ValueError("TRADE requires another agent within one tile")
        if action.resource == Resource.NONE:
            raise ValueError("TRADE requires a nonzero resource")
        recipient = self._addressed_recipients(index, action, neighbors)[0]
        requested = action.amount if action.amount > 0.0 else 0.25
        amount = min(requested, float(self.population.inventory[index, int(action.resource)]))
        amount = min(amount, self.population.free_capacity(recipient))
        if amount <= 0:
            raise ValueError("TRADE has no transferable mass or recipient capacity")
        self.population.inventory[index, int(action.resource)] -= np.float32(amount)
        self.population.inventory[recipient, int(action.resource)] += np.float32(amount)
        if (
            self.research is not None
            and self.config.science.immediate_replanning
            and self.config.science.message_interrupts
        ):
            self.needs_replan[recipient] = True
        events.append(
            WorldEvent(
                tick=self.tick,
                kind="resource_traded",
                payload={
                    "sender": self.population.agent_ids[index],
                    "recipient": self.population.agent_ids[recipient],
                    "resource": int(action.resource),
                    "amount": amount,
                    "reply_to": action.reply_to,
                },
            )
        )
    def _record_semantic_action(
        self,
        index: int,
        action: AgentAction,
        events: list[WorldEvent],
    ) -> None:
        payload = {
            "agent": self.population.agent_ids[index],
            "verb": int(action.verb),
            "x": int(self.population.x[index]),
            "y": int(self.population.y[index]),
            "message": action.message,
            "recipe": action.recipe,
            "insight": action.insight,
            "artifact_spec": action.artifact_spec,
            "causal_parents": action.causal_parents,
        }
        events.append(WorldEvent(tick=self.tick, kind="semantic_action", payload=payload))

    def observation(self, index: int, radius: int = 2) -> dict[str, Any]:
        x, y = int(self.population.x[index]), int(self.population.y[index])
        size = radius * 2 + 1
        patch = np.zeros((size, size, 5), dtype=np.float32)
        for py, world_y in enumerate(range(y - radius, y + radius + 1)):
            for px, world_x in enumerate(range(x - radius, x + radius + 1)):
                if not (0 <= world_x < self.world.width and 0 <= world_y < self.world.height):
                    continue
                patch[py, px, 0] = self.world.terrain[world_y, world_x] / 8.0
                patch[py, px, 1] = self.world.resource_kind[world_y, world_x] / 8.0
                capacity = self.world.resource_capacity[world_y, world_x]
                patch[py, px, 2] = (
                    self.world.resource_mass[world_y, world_x] / capacity if capacity > 0 else 0.0
                )
                patch[py, px, 3] = self.world.moisture[world_y, world_x]
                patch[py, px, 4] = self.world.nutrients[world_y, world_x]
        return {
            "position": np.asarray([x, y], dtype=np.int32),
            "energy": np.asarray([self.population.energy[index]], dtype=np.float32),
            "inventory": self.population.inventory[index].copy(),
            "local_patch": patch,
            "tick": np.asarray([self.tick], dtype=np.int64),
        }

    def semantic_observation(
        self,
        index: int,
        *,
        compact: bool = False,
        focus_text: str = "",
        artifact_detail_limit: int = 6,
        archive_limit: int = 4,
    ) -> str:
        x, y = int(self.population.x[index]), int(self.population.y[index])
        inventory = {
            self.resource_name(resource): round(float(amount), 3)
            for resource, amount in enumerate(self.population.inventory[index])
            if resource and amount > 1e-5
        }
        empirical_knowledge = self.empirical_knowledge[index].view(
            x=x,
            y=y,
            inventory=self.population.inventory[index],
            # Shared mass is visible separately in the mission packet, but possession
            # by the collective is not personal empirical knowledge.
            depot=None,
            terrain_name=self.terrain_name,
            resource_name=self.resource_name,
            station_name=self.station_name,
        )
        nearby_indices = np.nonzero(
            (np.abs(self.population.x - x) <= 3)
            & (np.abs(self.population.y - y) <= 3)
            & self.population.active
        )[0]
        nearby_agents = []
        for other in nearby_indices.tolist():
            if other == index:
                continue
            nearby_agents.append(
                {
                    "id": self.population.agent_ids[other],
                    "position": [
                        int(self.population.x[other]),
                        int(self.population.y[other]),
                    ],
                    "distance": int(
                        abs(int(self.population.x[other]) - x)
                        + abs(int(self.population.y[other]) - y)
                    ),
                    "inventory": {
                        self.resource_name(resource): round(float(amount), 3)
                        for resource, amount in enumerate(self.population.inventory[other])
                        if resource and amount > 1e-5
                    },
                    "last_action": ActionType(int(self.last_actions[other])).name,
                }
            )
        nearby_artifacts = []
        for artifact in range(self.artifacts.count):
            distance = abs(int(self.artifacts.x[artifact]) - x) + abs(
                int(self.artifacts.y[artifact]) - y
            )
            if distance <= 4:
                spec = self.artifacts.specs[artifact]
                nearby_artifacts.append(
                    {
                        "id": f"artifact_{artifact:08d}",
                        "name": spec.get("name", "Untitled material system"),
                        "architecture": spec.get("architecture", ""),
                        "claimed_function": spec.get("claimed_function", ""),
                        "bio_inspiration": spec.get("bio_inspiration", []),
                        "geometry": spec.get("geometry", {}),
                        "position": [
                            int(self.artifacts.x[artifact]),
                            int(self.artifacts.y[artifact]),
                        ],
                        "distance": distance,
                        "health": round(float(self.artifacts.health[artifact]), 4),
                        "performance": round(float(self.artifacts.performance[artifact]), 4),
                        "peak_performance_since_program_install": round(
                            float(self.artifacts.peak_performance[artifact]), 4
                        ),
                        "lifetime_peak_performance": round(
                            float(self.artifacts.lifetime_peak_performance[artifact]), 4
                        ),
                        "lifetime_peak_tick": int(
                            self.artifacts.lifetime_peak_tick[artifact]
                        ),
                        "lifetime_peak_program_id": (
                            self.artifacts.lifetime_peak_program_id[artifact] or None
                        ),
                        "services": {
                            service: round(
                                float(self.artifacts.services[artifact, service_index]),
                                4,
                            )
                            for service_index, service in enumerate(SERVICE_NAMES)
                        },
                        "peak_services_since_program_install": {
                            service: round(
                                float(self.artifacts.peak_services[artifact, service_index]),
                                4,
                            )
                            for service_index, service in enumerate(SERVICE_NAMES)
                        },
                        "lifetime_peak_services": {
                            service: round(
                                float(
                                    self.artifacts.lifetime_peak_services[
                                        artifact, service_index
                                    ]
                                ),
                                4,
                            )
                            for service_index, service in enumerate(SERVICE_NAMES)
                        },
                        "program": (
                            self.artifacts.programs[artifact].name
                            if self.artifacts.programs[artifact] is not None
                            else None
                        ),
                        "program_id": (
                            self.artifacts.programs[artifact].program_id
                            if self.artifacts.programs[artifact] is not None
                            else None
                        ),
                    }
                )
        artifact_catalog: list[dict[str, Any]] = []
        if compact:
            focus_artifacts = set(re.findall(r"artifact_[0-9]{8}", focus_text))
            nearby_artifacts.sort(
                key=lambda item: (
                    str(item["id"]) not in focus_artifacts,
                    -int(
                        self.artifacts.created_tick[
                            int(str(item["id"]).removeprefix("artifact_"))
                        ]
                    ),
                    int(item["distance"]),
                    str(item["id"]),
                )
            )
            artifact_catalog = [
                {
                    "id": item["id"],
                    "position": item["position"],
                    "distance": item["distance"],
                    "performance": item["performance"],
                    "program_id": item["program_id"],
                }
                for item in nearby_artifacts[:64]
            ]
        nearby_insights = []
        for insight in reversed(self.deposited_insights[-128:]):
            insight_x = int(insight.get("x", -10_000))
            insight_y = int(insight.get("y", -10_000))
            distance = abs(insight_x - x) + abs(insight_y - y)
            if distance > self.config.population.communication_radius:
                continue
            nearby_insights.append(
                {
                    "record_id": insight.get("record_id"),
                    "author": insight.get("author"),
                    "kind": insight.get("kind"),
                    "content": insight.get("content"),
                    "causal_parents": insight.get("causal_parents", []),
                    "position": [insight_x, insight_y],
                    "distance": distance,
                    "published": bool(insight.get("published", False)),
                }
            )
            if len(nearby_insights) >= (4 if compact else 8):
                break
        resource_landmarks: dict[str, dict[str, Any]] = {}
        for resource in Resource:
            if resource == Resource.NONE:
                continue
            ys, xs = np.nonzero(
                (self.world.resource_kind == int(resource)) & (self.world.resource_mass > 0.05)
            )
            if len(xs) == 0:
                continue
            distances = np.abs(xs - x) + np.abs(ys - y)
            nearest = int(np.argmin(distances))
            if (
                self.research is not None
                and not self.config.science.global_landmarks
                and int(distances[nearest]) > 4
            ):
                continue
            resource_landmarks[self.resource_name(resource)] = {
                "position": [int(xs[nearest]), int(ys[nearest])],
                "distance": int(distances[nearest]),
                "mass": round(float(self.world.resource_mass[ys[nearest], xs[nearest]]), 3),
            }
        station_landmarks: dict[str, dict[str, Any]] = {}
        for station in Station:
            if station == Station.NONE:
                continue
            ys, xs = np.nonzero(self.world.stations == int(station))
            if len(xs) == 0:
                continue
            distances = np.abs(xs - x) + np.abs(ys - y)
            nearest = int(np.argmin(distances))
            if (
                self.research is not None
                and not self.config.science.global_landmarks
                and int(distances[nearest]) > 4
            ):
                continue
            station_landmarks[self.station_name(station)] = {
                "position": [int(xs[nearest]), int(ys[nearest])],
                "distance": int(distances[nearest]),
            }
        public_infrastructure: dict[str, dict[str, Any]] = {}
        if self.research is not None and self.config.science.public_infrastructure_map:
            foundry_ys, foundry_xs = np.nonzero(
                self.world.terrain == int(Terrain.FOUNDRY)
            )
            if len(foundry_xs):
                distances = np.abs(foundry_xs - x) + np.abs(foundry_ys - y)
                nearest = int(np.argmin(distances))
                public_infrastructure["FOUNDRY_WORKSPACE"] = {
                    "position": [
                        int(foundry_xs[nearest]),
                        int(foundry_ys[nearest]),
                    ],
                    "distance": int(distances[nearest]),
                    "supports": ["DEPOSIT", "OPERATE"],
                }
            for station in Station:
                if station == Station.NONE:
                    continue
                station_ys, station_xs = np.nonzero(
                    self.world.stations == int(station)
                )
                if len(station_xs) == 0:
                    continue
                distances = np.abs(station_xs - x) + np.abs(station_ys - y)
                nearest = int(np.argmin(distances))
                public_infrastructure[self.station_name(station)] = {
                    "position": [
                        int(station_xs[nearest]),
                        int(station_ys[nearest]),
                    ],
                    "distance": int(distances[nearest]),
                    "supports": ["DEPOSIT", "OPERATE"],
                }
        adjacent: dict[str, Any] = {}
        walkable_directions: list[str] = []
        for name, dx, dy in (
            ("NORTH", 0, -1),
            ("EAST", 1, 0),
            ("SOUTH", 0, 1),
            ("WEST", -1, 0),
        ):
            ax, ay = x + dx, y + dy
            if 0 <= ax < self.world.width and 0 <= ay < self.world.height:
                walkable = bool(self.world.walkable[ay, ax])
                adjacent[name] = {
                    "walkable": walkable,
                    "terrain": self.terrain_name(int(self.world.terrain[ay, ax])),
                    "resource": self.resource_name(int(self.world.resource_kind[ay, ax])),
                }
                if walkable:
                    walkable_directions.append(name)
            else:
                adjacent[name] = {"walkable": False, "terrain": "BOUNDARY"}
        terrain_here = Terrain(int(self.world.terrain[y, x]))
        station_here = Station(int(self.world.stations[y, x]))
        resource_here = Resource(int(self.world.resource_kind[y, x]))
        workspace_here = terrain_here == Terrain.FOUNDRY or station_here != Station.NONE
        has_adjacent_agent = any(item["distance"] <= 1 for item in nearby_agents)
        has_nearby_artifact = any(item["distance"] <= 1 for item in nearby_artifacts)
        pending_test = bool(
            self.research is not None
            and self.research.latest_untested_batch(self.population.agent_ids[index])
            is not None
        )
        recent_feedback = list(self.action_feedback[index])
        recent_action_counts = Counter(
            str(item.get("action", "UNKNOWN")) for item in recent_feedback
        )
        same_action_streak = 0
        if recent_feedback:
            latest_action = recent_feedback[-1].get("action")
            for item in reversed(recent_feedback):
                if item.get("action") != latest_action:
                    break
                same_action_streak += 1
        packet: dict[str, Any] = {
            "tick": self.tick,
            "self": {
                "id": self.population.agent_ids[index],
                "position": [x, y],
                "inventory": inventory,
                "inventory_capacity": self.config.population.inventory_capacity,
                "free_capacity": round(self.population.free_capacity(index), 3),
                "energy": round(float(self.population.energy[index]), 4),
                "generation": int(self.population.generation[index]),
            },
            "local": {
                "terrain": self.terrain_name(terrain_here),
                "station": self.station_name(station_here),
                "resource": self.resource_name(resource_here),
                "resource_mass": round(float(self.world.resource_mass[y, x]), 3),
                "moisture": round(float(self.world.moisture[y, x]), 3),
                "nutrients": round(float(self.world.nutrients[y, x]), 3),
                "temperature": round(float(self.world.temperature[y, x]), 3),
                "solar": round(float(self.world.solar[y, x]), 3),
                "contamination": round(float(self.world.contamination[y, x]), 3),
                "adjacent": adjacent,
            },
            # These are current physical affordances, not a recommended plan. Exposing
            # them avoids asking an LLM to reverse-engineer engine legality from labels
            # such as FOUNDRY while keeping recipes, goals, and scientific choices open.
            "local_affordances": {
                "walkable_directions": walkable_directions,
                "collectable_material_here": (
                    self.resource_name(resource_here)
                    if resource_here != Resource.NONE
                    and float(self.world.resource_mass[y, x]) > 1e-7
                    and self.population.free_capacity(index) > 1e-7
                    else None
                ),
                "fabrication_workspace_here": workspace_here,
                "shared_depot_access_here": bool(
                    workspace_here
                    and self.research is not None
                    and self.config.science.shared_depot
                ),
                "untested_microbatch_ready": pending_test,
                "adjacent_agent_exchange_possible": has_adjacent_agent,
                "nearby_artifact_manipulation_possible": has_nearby_artifact,
                "metabolism_enabled": self.config.economy.enabled,
            },
            "nearby_agents": nearby_agents[:12],
            "nearby_artifacts": nearby_artifacts[
                : (artifact_detail_limit if compact else 12)
            ],
            "nearby_insights": nearby_insights,
            "nearest_resources": resource_landmarks,
            "nearest_stations": station_landmarks,
            "public_infrastructure": public_infrastructure,
            "empirical_knowledge": empirical_knowledge,
            "verified_skill_library": (
                self.program_library.view(
                    self.population.agent_ids[index],
                    limit=6 if compact else 12,
                    focus_program_ids=set(
                        re.findall(r"program_[0-9a-f]{24}", focus_text)
                    ),
                    compact=compact,
                )
                if self.config.science.skill_library
                else []
            ),
            "recent_action_results": (
                recent_feedback
                if self.research is None or self.config.science.action_feedback
                else []
            ),
            "recent_behavior_summary": (
                {
                    "window": len(recent_feedback),
                    "action_counts": dict(sorted(recent_action_counts.items())),
                    "successful_actions": sum(
                        bool(item.get("success")) for item in recent_feedback
                    ),
                    "failed_actions": sum(
                        not bool(item.get("success")) for item in recent_feedback
                    ),
                    "latest_action_streak": same_action_streak,
                }
                if self.research is None or self.config.science.action_feedback
                else {}
            ),
            "public_archive": (
                (
                    self.archive.retrieve(focus_text, archive_limit)
                    if compact
                    else self.archive.recent(8)
                )
                if self.research is None or self.config.science.communication
                else []
            ),
            "locally_available_counts": {
                "nearby_artifacts": len(nearby_artifacts),
                "visible_archive_entries": (
                    len(self.archive.records)
                    if self.research is None or self.config.science.communication
                    else 0
                ),
            },
        }
        if self.scenario is not None:
            packet["scenario"] = self.scenario.descriptor()
            packet["local"]["fields"] = self.world.field_values(x, y)
            packet["local"]["compatibility_field_aliases"] = {
                "moisture": "water_availability",
                "nutrients": "ground_stability",
                "contamination": "toxic_gas",
            }
        if compact:
            packet["nearby_artifact_catalog"] = artifact_catalog
        if self.config.science.request_tracking:
            packet["pending_requests"] = self._pending_requests(index)
        if self.research is not None:
            packet["research_mission"] = self.research.agent_view(
                self.artifacts,
                len(self.archive.records),
                evidence_limit=8 if compact else 16,
            )
            if self.scenario is not None:
                scenario_mission = self.scenario.descriptor()["mission"]
                packet["research_mission"].update(
                    {
                        "mission": self.scenario.scenario_id,
                        "title": scenario_mission["title"],
                        "objective": scenario_mission["objective"],
                        "scenario_milestones": scenario_mission["milestones"],
                        "metric_backend_note": (
                            "Shared research counters retain stable wire keys; interpret "
                            "resources, properties, and services through the scenario catalog."
                        ),
                    }
                )
            private_experiments = self.research.private_experiment_view(
                self.population.agent_ids[index], limit=4 if compact else 8
            )
            if (
                private_experiments["pending_microbatches"]
                or private_experiments["recent_test_results"]
            ):
                packet["private_experiments"] = private_experiments
        return json.dumps(packet, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _presentation_evidence_summary(
        record_id: str,
        kind: str,
        payload: Mapping[str, Any],
    ) -> str:
        """Describe cited private evidence without adding scientific interpretation."""

        if kind == "observation":
            location = f"({payload.get('x', '?')}, {payload.get('y', '?')})"
            terrain = str(payload.get("terrain", "unknown terrain")).replace("_", " ")
            resource = str(payload.get("resource", "NONE")).replace("_", " ")
            station = str(payload.get("station", "NONE")).replace("_", " ")
            measured = payload.get("artifact_measurements", [])
            suffix = (
                f" · measured {len(measured)} artifact{'s' if len(measured) != 1 else ''}"
                if isinstance(measured, list) and measured
                else ""
            )
            return (
                f"Observed {terrain} at {location} · resource {resource} · "
                f"station {station}{suffix}"
            )
        if kind == "experiment_result":
            recipe = payload.get("source_recipe", {})
            inputs = recipe.get("inputs", []) if isinstance(recipe, Mapping) else []
            resources = [
                str(item.get("resource", "material")).replace("_", " ")
                for item in inputs
                if isinstance(item, Mapping)
            ]
            material = " + ".join(resources) or "material coupon"
            utility = payload.get("material_utility")
            suffix = f" · utility {float(utility):.3f}" if isinstance(utility, (int, float)) else ""
            return f"Tested {material}{suffix}"
        if kind == "recipe":
            recipe = payload.get("recipe", payload)
            output = recipe.get("output_form") if isinstance(recipe, Mapping) else None
            return f"Proposed recipe · {output}" if output else "Proposed material recipe"
        return kind.replace("_", " ").strip().capitalize() or record_id

    def _causal_evidence_snapshot(self, limit: int = 256) -> list[dict[str, Any]]:
        """Return compact metadata only for private records cited by public outcomes.

        This observer instrument does not create an agent communication channel. It
        makes already-recorded causal IDs legible without broadcasting the rest of an
        agent's private memory.
        """

        public_ids = {
            record.record_id for record in self.archive.records
        } | {
            str(record.get("record_id", "")) for record in self.deposited_insights
        } | {
            f"artifact_{index:08d}" for index in range(self.artifacts.count)
        } | set(self.program_library.programs)
        public_ids.discard("")

        referenced: set[str] = set()
        for record in self.archive.records:
            referenced.update(map(str, record.causal_parents))
        for record in self.deposited_insights:
            referenced.update(map(str, record.get("causal_parents", [])))
        for provenance in self.artifacts.provenance.values():
            referenced.update(map(str, provenance.get("causal_parents", [])))

        records: dict[str, tuple[MemoryRecord, str]] = {}
        for owner, memory in zip(self.population.agent_ids, self.memories, strict=True):
            for record in (*memory.working, *memory.episodic, *memory.notebook):
                records.setdefault(record.record_id, (record, owner))

        recipes: dict[str, dict[str, Any]] = {}
        if self.research is not None:
            for proposal in self.research.proposals:
                recipes.setdefault(str(proposal["recipe_id"]), proposal)
            for batch in self.research.batches:
                recipes.setdefault(
                    str(batch.recipe_id),
                    {
                        "recipe_id": batch.recipe_id,
                        "agent": batch.agent,
                        "tick": int(batch.batch.tick),
                        "recipe": batch.source_recipe,
                        "causal_parents": list(batch.batch.causal_parents),
                    },
                )

        selected: dict[str, dict[str, Any]] = {}
        pending = sorted(referenced)
        visited: set[str] = set()
        while pending and len(selected) < max(1, limit):
            record_id = pending.pop(0)
            if not record_id or record_id in visited or record_id in public_ids:
                continue
            visited.add(record_id)
            record_entry = records.get(record_id)
            if record_entry is not None:
                record, owner = record_entry
                try:
                    parsed = json.loads(record.content)
                except (TypeError, json.JSONDecodeError):
                    parsed = {}
                payload = parsed if isinstance(parsed, Mapping) else {}
                author = str(payload.get("agent", owner))
                related_artifacts = sorted(
                    {
                        str(measurement.get("artifact_id"))
                        for measurement in payload.get("artifact_measurements", [])
                        if isinstance(measurement, Mapping)
                        and measurement.get("artifact_id")
                    }
                )
                selected[record_id] = {
                    "record_id": record_id,
                    "kind": record.kind,
                    "author": author,
                    "tick": record.tick,
                    "summary": self._presentation_evidence_summary(
                        record_id, record.kind, payload
                    ),
                    "content": record.content[:1200],
                    "causal_parents": list(record.causal_parents),
                    "related_artifacts": related_artifacts,
                }
                pending.extend(
                    parent
                    for parent in map(str, record.causal_parents)
                    if parent not in visited
                )
                continue
            proposal = recipes.get(record_id)
            if proposal is not None:
                parents = list(map(str, proposal.get("causal_parents", [])))
                selected[record_id] = {
                    "record_id": record_id,
                    "kind": "recipe",
                    "author": str(proposal.get("agent", "")),
                    "tick": int(proposal.get("tick", 0)),
                    "summary": self._presentation_evidence_summary(
                        record_id, "recipe", proposal
                    ),
                    "content": json.dumps(
                        proposal.get("recipe", {}), sort_keys=True
                    )[:1200],
                    "causal_parents": parents,
                    "related_artifacts": [],
                }
                pending.extend(parent for parent in parents if parent not in visited)

        return sorted(
            selected.values(),
            key=lambda item: (int(item.get("tick", 0)), str(item["record_id"])),
        )

    def snapshot(self, display_limit: int | None = 2048) -> dict[str, Any]:
        agents = self.population.snapshot(display_limit)
        display_count = int(agents["display_count"])
        agents["last_action"] = self.last_actions[:display_count].tolist()
        agents["distance_traveled"] = self.distance_traveled[:display_count].tolist()
        agents["distinct_cells_visited"] = [
            len(self.visited_cells[index]) for index in range(display_count)
        ]
        snapshot = {
            "protocol": PROTOCOL_VERSION,
            "tick": self.tick,
            "seed": self.seed,
            "world": self.world.snapshot(),
            "agents": agents,
            "artifacts": self.artifacts.snapshot(display_limit),
            "archive": self.archive.recent(32),
            "insights": self.deposited_insights[-64:],
            "causal_evidence": self._causal_evidence_snapshot(),
            "metrics": {
                "artifact_score": round(self.artifacts.summary_score(), 6),
                "archive_entries": len(self.archive.records),
                "deposited_insights": len(self.deposited_insights),
                "programs": len(self.program_library.programs),
                "program_lineage_edges": len(self.program_library.lineage_edges),
                "messages": len(self.message_records),
                "active_agents": int(np.count_nonzero(self.population.active)),
                "action_attempts_used": self.action_attempts_used,
                "action_attempts_blocked": self.action_attempts_blocked,
                "action_attempt_budget": self.config.simulation.action_attempt_budget,
            },
            "program_lineage": self.program_library.lineage_edges[-128:],
            "program_catalog": list(self.program_library.programs.values())[-256:],
        }
        if self.scenario is not None:
            snapshot["scenario"] = self.scenario.descriptor()
        if self.research is not None:
            research = self.research.scorecard(self.artifacts, len(self.archive.records))
            snapshot["research"] = research
            snapshot["metrics"]["research_score"] = research["score"]
            snapshot["metrics"]["research_completed"] = research["completed"]
        return snapshot

    def state_digest(self) -> str:
        return digest_snapshot(self.authoritative_state())

    def authoritative_state(self) -> dict[str, Any]:
        """Complete deterministic state used for replay verification, not rendering."""
        artifact_count = self.artifacts.count
        state = {
            "seed": self.seed,
            "tick": self.tick,
            "record_counter": self._record_counter,
            "distance_traveled": self.distance_traveled.tolist(),
            "visited_cells": [sorted(cells) for cells in self.visited_cells],
            "action_attempts_used": self.action_attempts_used,
            "action_attempts_blocked": self.action_attempts_blocked,
            "action_budget_exhausted_tick": self.action_budget_exhausted_tick,
            "rng_state": self.rng.bit_generator.state,
            "world": {
                "terrain": self.world.terrain.tolist(),
                "resource_kind": self.world.resource_kind.tolist(),
                "resource_mass": self.world.resource_mass.tolist(),
                "resource_capacity": self.world.resource_capacity.tolist(),
                "stations": self.world.stations.tolist(),
                "temperature": self.world.temperature.tolist(),
                "moisture": self.world.moisture.tolist(),
                "nutrients": self.world.nutrients.tolist(),
                "contamination": self.world.contamination.tolist(),
                "solar": self.world.solar.tolist(),
                "disturbance_centers": self.world._disturbance_centers,
                "last_disturbance_mask": self.world.last_disturbance_mask.tolist(),
                "generator_manifest": self.world.generator_manifest,
                "flux_ledger": self.world.flux_ledger,
            },
            "population": {
                "x": self.population.x.tolist(),
                "y": self.population.y.tolist(),
                "energy": self.population.energy.tolist(),
                "inventory": self.population.inventory.tolist(),
                "active": self.population.active.tolist(),
                "generation": self.population.generation.tolist(),
                "death_tick": self.population.death_tick.tolist(),
                "last_macroturn": self.population.last_macroturn.tolist(),
            },
            "artifacts": {
                "count": artifact_count,
                "kind": self.artifacts.kind[:artifact_count].tolist(),
                "x": self.artifacts.x[:artifact_count].tolist(),
                "y": self.artifacts.y[:artifact_count].tolist(),
                "health": self.artifacts.health[:artifact_count].tolist(),
                "maturity": self.artifacts.maturity[:artifact_count].tolist(),
                "performance": self.artifacts.performance[:artifact_count].tolist(),
                "peak_performance": self.artifacts.peak_performance[:artifact_count].tolist(),
                "lifetime_peak_performance": self.artifacts.lifetime_peak_performance[
                    :artifact_count
                ].tolist(),
                "lifetime_peak_tick": self.artifacts.lifetime_peak_tick[
                    :artifact_count
                ].tolist(),
                "lifetime_peak_program_id": self.artifacts.lifetime_peak_program_id[
                    :artifact_count
                ],
                "storage": self.artifacts.storage[:artifact_count].tolist(),
                "reserve": self.artifacts.reserve[:artifact_count].tolist(),
                "open_fraction": self.artifacts.open_fraction[:artifact_count].tolist(),
                "retired": self.artifacts.retired[:artifact_count].tolist(),
                "fluxes": self.artifacts.fluxes[:artifact_count].tolist(),
                "properties": self.artifacts.properties[:artifact_count].tolist(),
                "services": self.artifacts.services[:artifact_count].tolist(),
                "peak_services": self.artifacts.peak_services[:artifact_count].tolist(),
                "lifetime_peak_services": self.artifacts.lifetime_peak_services[
                    :artifact_count
                ].tolist(),
                "specs": self.artifacts.specs[:artifact_count],
                "created_tick": self.artifacts.created_tick[:artifact_count].tolist(),
                "creator": self.artifacts.creator[:artifact_count],
                "batch_id": self.artifacts.batch_id[:artifact_count],
                "provenance": self.artifacts.provenance,
                "programs": [
                    program.as_dict() if program is not None else None
                    for program in self.artifacts.programs[:artifact_count]
                ],
            },
            "memories": [
                {
                    "working": [record.as_dict() for record in memory.working],
                    "episodic": [record.as_dict() for record in memory.episodic],
                    "notebook": [record.as_dict() for record in memory.notebook],
                }
                for memory in self.memories
            ],
            "empirical_knowledge": [
                knowledge.authoritative_state() for knowledge in self.empirical_knowledge
            ],
            "archive": [record.as_dict() for record in self.archive.records],
            "deposited_insights": self.deposited_insights,
            "program_library": self.program_library.authoritative_state(),
            "message_records": self.message_records,
            "last_rewards": self.last_rewards.tolist(),
            "last_actions": self.last_actions.tolist(),
            "last_events": [event.as_dict() for event in self.last_events],
        }
        if self.research is not None:
            state["research"] = self.research.authoritative_state()
            state["action_feedback"] = [list(records) for records in self.action_feedback]
            state["needs_replan"] = self.needs_replan.tolist()
            state["research_milestones"] = self._research_milestones
            state["research_completed"] = self._research_completed
        if self.scenario is not None:
            state["scenario"] = self.scenario.descriptor()
            state["world"]["fields"] = {
                name: field.tolist() for name, field in self.world.fields.items()
            }
            state["world"]["resource_regrowth"] = self.world._resource_regrowth.tolist()
        return state

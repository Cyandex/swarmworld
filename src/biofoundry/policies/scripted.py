"""Deterministic policies used for smoke tests, rendering, and causal fixtures."""

from __future__ import annotations

from collections import deque

import numpy as np

from ..materials import DEFAULT_RECIPES, MaterialLab, required_inputs
from ..simulation import BioFoundrySimulation
from ..types import (
    ActionType,
    AgentAction,
    ArtifactType,
    Direction,
    Resource,
    Station,
    Terrain,
)


class ScalableForagerPolicy:
    """O(N) policy suitable for simulator scaling benchmarks."""

    def actions(self, simulation: BioFoundrySimulation) -> dict[str, AgentAction]:
        pop = simulation.population
        masses = simulation.world.resource_mass[pop.y, pop.x]
        result: dict[str, AgentAction] = {}
        for index, agent_id in enumerate(pop.agent_ids):
            if masses[index] > 0.05 and pop.free_capacity(index) > 0.05:
                result[agent_id] = AgentAction(
                    verb=ActionType.HARVEST,
                )
                continue
            direction = Direction(1 + ((index * 17 + simulation.tick * 7) % 4))
            result[agent_id] = AgentAction(verb=ActionType.MOVE, direction=direction)
        return result


class ScenarioIndustryPolicy:
    """Catalog-aware baseline for declarative resource-processing scenarios.

    It contains no Ashen Realms names. The active package supplies the recipe,
    resource slots, facilities, and world geometry.
    """

    @staticmethod
    def _step_toward(
        simulation: BioFoundrySimulation,
        start_x: int,
        start_y: int,
        targets: np.ndarray,
    ) -> Direction:
        if targets[start_y, start_x]:
            return Direction.STAY
        height, width = targets.shape
        queue: deque[tuple[int, int]] = deque([(start_x, start_y)])
        first_step: dict[tuple[int, int], Direction] = {(start_x, start_y): Direction.STAY}
        for_x_y = (
            (0, -1, Direction.NORTH),
            (1, 0, Direction.EAST),
            (0, 1, Direction.SOUTH),
            (-1, 0, Direction.WEST),
        )
        while queue:
            x, y = queue.popleft()
            for dx, dy, direction in for_x_y:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if (nx, ny) in first_step or not simulation.world.walkable[ny, nx]:
                    continue
                first_step[(nx, ny)] = (
                    direction if (x, y) == (start_x, start_y) else first_step[(x, y)]
                )
                if targets[ny, nx]:
                    return first_step[(nx, ny)]
                queue.append((nx, ny))
        return Direction.STAY

    def actions(self, simulation: BioFoundrySimulation) -> dict[str, AgentAction]:
        if simulation.scenario is None or simulation.research is None:
            return ScalableForagerPolicy().actions(simulation)
        recipe = simulation.scenario.default_recipe
        required = simulation.material_lab.required_inputs(recipe)
        result: dict[str, AgentAction] = {}
        for index, agent_id in enumerate(simulation.population.agent_ids):
            x = int(simulation.population.x[index])
            y = int(simulation.population.y[index])
            pending = simulation.research.latest_untested_batch(agent_id)
            if pending is not None:
                result[agent_id] = AgentAction(verb=ActionType.TEST)
                continue
            already_built = agent_id in simulation.artifacts.creator[: simulation.artifacts.count]
            if already_built:
                result[agent_id] = AgentAction(verb=ActionType.INSPECT)
                continue
            grounded = simulation.grounded_resources(index)
            missing = [
                resource
                for resource, mass in required.items()
                if resource not in grounded
                or float(simulation.population.inventory[index, int(resource)]) + 1e-7 < mass
            ]
            if missing:
                resource = missing[(index + simulation.tick // 24) % len(missing)]
                if (
                    int(simulation.world.resource_kind[y, x]) == int(resource)
                    and float(simulation.world.resource_mass[y, x]) > 0.05
                ):
                    result[agent_id] = AgentAction(verb=ActionType.HARVEST)
                    continue
                targets = (
                    (simulation.world.resource_kind == int(resource))
                    & (simulation.world.resource_mass > 0.05)
                )
                direction = self._step_toward(simulation, x, y, targets)
                result[agent_id] = AgentAction(verb=ActionType.MOVE, direction=direction)
                continue
            workspace = (simulation.world.stations != int(Station.NONE)) | (
                simulation.world.terrain == int(Terrain.FOUNDRY)
            )
            if workspace[y, x]:
                tested = any(
                    str(record.get("agent", "")) == agent_id
                    for record in simulation.research.tests
                )
                if tested:
                    result[agent_id] = AgentAction(
                        verb=ActionType.BUILD,
                        artifact=ArtifactType.MATERIAL_SYSTEM,
                        recipe=simulation.scenario.externalize_recipe(recipe),
                        artifact_spec={
                            "name": "Volcanic load-and-heat barrier",
                            "claimed_function": (
                                "Maintain structural support while moderating volcanic heat."
                            ),
                            "architecture": (
                                "Forged ribs joined to a modular mineral-facing shell."
                            ),
                            "bio_inspiration": ["layered volcanic strata"],
                            "predicted_effects": [
                                "redirect heat across the facing",
                                "carry load through connected forged ribs",
                            ],
                            "geometry": {
                                "layers": 4,
                                "surface_area": 1.4,
                                "channel_density": 0.18,
                                "anisotropy": 0.62,
                                "branching": 0.22,
                                "connectivity": 0.78,
                                "curvature": 0.16,
                                "modularity": 0.70,
                            },
                        },
                        program={
                            "name": "passive_thermal_aperture",
                            "parent_program": None,
                            "instructions": [
                                {"op": "sense", "dest": "r0", "sensor": "temperature"},
                                {"op": "set_open", "value": "r0"},
                            ],
                        },
                    )
                else:
                    result[agent_id] = AgentAction(
                        verb=ActionType.OPERATE,
                        recipe=simulation.scenario.externalize_recipe(recipe),
                    )
            else:
                direction = self._step_toward(simulation, x, y, workspace)
                result[agent_id] = AgentAction(verb=ActionType.MOVE, direction=direction)
        return result


class ScriptedBioFoundryPolicy:
    """A transparent society that renders varied continuous-architecture fixtures."""

    def __init__(self) -> None:
        self._cells: dict[Resource, np.ndarray] | None = None
        self._build_counts: dict[str, int] = {}
        self._program_written: set[str] = set()

    def _resource_cells(self, simulation: BioFoundrySimulation) -> dict[Resource, np.ndarray]:
        if self._cells is None:
            self._cells = {}
            for resource in Resource:
                if resource == Resource.NONE:
                    continue
                ys, xs = np.nonzero(simulation.world.resource_kind == int(resource))
                self._cells[resource] = np.stack([xs, ys], axis=1) if len(xs) else np.empty((0, 2))
        return self._cells

    def actions(self, simulation: BioFoundrySimulation) -> dict[str, AgentAction]:
        if simulation.population.size > 512:
            return ScalableForagerPolicy().actions(simulation)
        cells = self._resource_cells(simulation)
        result: dict[str, AgentAction] = {}
        for index, agent_id in enumerate(simulation.population.agent_ids):
            artifact = ArtifactType.MATERIAL_SYSTEM
            recipe = DEFAULT_RECIPES[artifact]
            inventory = simulation.population.inventory[index]
            built = self._build_counts.get(agent_id, 0)
            if built >= 1 and agent_id not in self._program_written:
                self._program_written.add(agent_id)
                result[agent_id] = AgentAction(
                    verb=ActionType.WRITE_PROGRAM,
                    target_x=int(simulation.population.x[index]),
                    target_y=int(simulation.population.y[index]),
                    program=self._authored_program(index),
                )
                continue
            if built >= 2:
                if (simulation.tick + index * 5) % 113 == 0:
                    result[agent_id] = AgentAction(
                        verb=ActionType.PUBLISH,
                        insight={
                            "kind": "validated_method",
                            "content": (
                                f"{artifact.name.lower()} cycle {built} remained functional; "
                                "compare its program and local field response"
                            ),
                            "salience": 0.72,
                        },
                    )
                elif (simulation.tick + index * 7) % 37 == 0:
                    result[agent_id] = AgentAction(
                        verb=ActionType.COMMUNICATE,
                        message=f"Sharing observations from {artifact.name.lower()} testing.",
                    )
                else:
                    direction = Direction(1 + ((simulation.tick // 8 + index) % 4))
                    result[agent_id] = AgentAction(verb=ActionType.MOVE, direction=direction)
                continue
            if MaterialLab.can_execute(recipe, inventory):
                target_x, target_y = self._build_target(simulation, index, built)
                x = int(simulation.population.x[index])
                y = int(simulation.population.y[index])
                if x == target_x and y == target_y:
                    result[agent_id] = AgentAction(
                        verb=ActionType.BUILD,
                        artifact=artifact,
                        recipe=recipe,
                        artifact_spec=self._scripted_spec(index),
                    )
                    self._build_counts[agent_id] = built + 1
                else:
                    result[agent_id] = AgentAction(
                        verb=ActionType.MOVE,
                        direction=self._move_toward(x, y, target_x, target_y),
                    )
                continue

            missing = [
                resource
                for resource, mass in required_inputs(recipe).items()
                if float(inventory[int(resource)]) + 1e-6 < mass
            ]
            target_resource = missing[0]
            x = int(simulation.population.x[index])
            y = int(simulation.population.y[index])
            current = Resource(int(simulation.world.resource_kind[y, x]))
            if (
                current == target_resource
                and simulation.world.resource_mass[y, x] > 0.05
                and simulation.population.free_capacity(index) > 0.05
            ):
                result[agent_id] = AgentAction(verb=ActionType.HARVEST)
                continue

            options = cells.get(target_resource, np.empty((0, 2)))
            if len(options) == 0:
                result[agent_id] = AgentAction()
                continue
            distances = np.abs(options[:, 0] - x) + np.abs(options[:, 1] - y)
            target_x, target_y = map(int, options[int(np.argmin(distances))])
            result[agent_id] = AgentAction(
                verb=ActionType.MOVE,
                direction=self._move_toward(x, y, target_x, target_y),
            )
        return result

    @staticmethod
    def _move_toward(x: int, y: int, target_x: int, target_y: int) -> Direction:
        if abs(target_x - x) >= abs(target_y - y):
            return Direction.EAST if target_x > x else Direction.WEST
        return Direction.SOUTH if target_y > y else Direction.NORTH

    @staticmethod
    def _build_target(simulation: BioFoundrySimulation, index: int, built: int) -> tuple[int, int]:
        ys, xs = np.nonzero(simulation.world.terrain == int(Terrain.TEST_FIELD))
        if len(xs) == 0:
            return int(simulation.world.width * 0.6), int(simulation.world.height * 0.7)
        choice = (index * 11 + built * 17) % len(xs)
        return int(xs[choice]), int(ys[choice])

    @staticmethod
    def _authored_program(index: int) -> dict[str, object]:
        behavior = index % 4
        if behavior == 0:
            instructions = [
                {"op": "gt", "dest": "r0", "a": "moisture", "b": 0.45},
                {"op": "mul", "dest": "r1", "a": "r0", "b": 0.05},
                {"op": "collect_water", "value": "r1"},
            ]
        elif behavior == 1:
            instructions = [
                {"op": "mul", "dest": "r0", "a": "moisture", "b": "healing"},
                {"op": "mul", "dest": "r1", "a": "r0", "b": 0.009},
                {"op": "grow", "value": "r1"},
            ]
        elif behavior == 2:
            instructions = [
                {"op": "sub", "dest": "r0", "a": 1.0, "b": "health"},
                {"op": "mul", "dest": "r1", "a": "r0", "b": "healing"},
                {"op": "mul", "dest": "r2", "a": "r1", "b": 0.012},
                {"op": "heal", "value": "r2"},
            ]
        else:
            instructions = [
                {"op": "sub", "dest": "r0", "a": 1.0, "b": "solar"},
                {"op": "mul", "dest": "r1", "a": "r0", "b": "responsiveness"},
                {"op": "set_open", "value": "r1"},
            ]
        return {
            "name": f"agent_variant_{index:03d}",
            "parent_program": None,
            "instructions": instructions,
        }

    @staticmethod
    def _scripted_spec(index: int) -> dict[str, object]:
        return {
            "name": f"Scripted material system {index:03d}",
            "claimed_function": "A deterministic renderer and mechanics fixture",
            "architecture": f"Procedural fixture architecture variant {index % 4}",
            "bio_inspiration": ["generic biological structure"],
            "predicted_effects": ["exercise one bounded field actuator"],
            "geometry": {
                "layers": 2,
                "surface_area": 1.2,
                "channel_density": 0.35,
                "anisotropy": 0.4,
                "branching": 0.2 + 0.2 * (index % 4),
                "connectivity": 0.65,
                "curvature": 0.3,
                "modularity": 0.4,
            },
        }


class OracleResearchSocietyPolicy:
    """Transparent upper-bound policy that exercises the complete research lifecycle."""

    def __init__(self) -> None:
        self.inspected: set[str] = set()
        self.published: set[str] = set()
        self.deposited: set[str] = set()
        self.extra_inspected: set[str] = set()

    def actions(self, simulation: BioFoundrySimulation) -> dict[str, AgentAction]:
        if simulation.research is None or simulation.population.size < 3:
            return ScriptedBioFoundryPolicy().actions(simulation)
        result = {agent_id: AgentAction() for agent_id in simulation.agent_ids}
        recipe = DEFAULT_RECIPES[ArtifactType.MATERIAL_SYSTEM]
        resource_workers = ((0, Resource.KELP), (1, Resource.SHELL))
        for index, resource in resource_workers:
            result[simulation.agent_ids[index]] = self._resource_worker_action(
                simulation, index, resource, recipe
            )
        result[simulation.agent_ids[2]] = self._builder_action(simulation, 2, recipe)
        for index in range(3, simulation.population.size):
            agent_id = simulation.agent_ids[index]
            if agent_id not in self.extra_inspected:
                self.extra_inspected.add(agent_id)
                result[agent_id] = AgentAction(verb=ActionType.INSPECT)
        return result

    def _resource_worker_action(
        self,
        simulation: BioFoundrySimulation,
        index: int,
        resource: Resource,
        recipe: dict[str, object],
    ) -> AgentAction:
        agent_id = simulation.agent_ids[index]
        if agent_id not in self.inspected:
            self.inspected.add(agent_id)
            return AgentAction(verb=ActionType.INSPECT)
        required = required_inputs(recipe)
        target_mass = required[resource] * (1.0 + simulation.config.science.test_scale)
        inventory = float(simulation.population.inventory[index, int(resource)])
        if inventory + 1e-7 < target_mass:
            x, y = int(simulation.population.x[index]), int(simulation.population.y[index])
            if (
                Resource(int(simulation.world.resource_kind[y, x])) == resource
                and float(simulation.world.resource_mass[y, x]) > 0.05
            ):
                return AgentAction(verb=ActionType.HARVEST)
            target = self._nearest_resource(simulation, x, y, resource)
            return AgentAction(
                verb=ActionType.MOVE,
                direction=self._path_direction(simulation, x, y, *target),
            )
        if agent_id not in self.published:
            self.published.add(agent_id)
            x, y = int(simulation.population.x[index]), int(simulation.population.y[index])
            return AgentAction(
                verb=ActionType.PUBLISH,
                insight={
                    "kind": "independent_material_observation",
                    "content": (
                        f"After direct collection at ({x},{y}), I possess measured "
                        f"{resource.name} and propose testing its contribution to a "
                        "resilient material system."
                    ),
                    "salience": 0.82,
                },
            )
        if agent_id not in self.deposited:
            x, y = int(simulation.population.x[index]), int(simulation.population.y[index])
            target_x, target_y = self._tester_location(simulation)
            if not self._on_foundry(simulation, x, y):
                return AgentAction(
                    verb=ActionType.MOVE,
                    direction=self._path_direction(simulation, x, y, target_x, target_y),
                )
            self.deposited.add(agent_id)
            return AgentAction(
                verb=ActionType.DEPOSIT,
                resource=resource,
                amount=target_mass,
            )
        return AgentAction()

    def _builder_action(
        self,
        simulation: BioFoundrySimulation,
        index: int,
        recipe: dict[str, object],
    ) -> AgentAction:
        research = simulation.research
        assert research is not None
        agent_id = simulation.agent_ids[index]
        if research.scorecard(simulation.artifacts, len(simulation.archive.records))["completed"]:
            return AgentAction()
        if agent_id not in research.claims:
            return AgentAction(
                verb=ActionType.CLAIM_TASK,
                message=(
                    "Integrate independent evidence, test a material hypothesis, "
                    "invent a system, build it, and program its field behavior"
                ),
            )
        own_proposals = [item for item in research.proposals if item["agent"] == agent_id]
        parent_ids = [record.record_id for record in simulation.archive.records[:2]]
        if not own_proposals:
            if set(required_inputs(recipe)).issubset(simulation.grounded_resources(index)):
                return AgentAction(verb=ActionType.PROPOSE_RECIPE, recipe=recipe)
            if len(parent_ids) >= 2:
                return AgentAction(
                    verb=ActionType.COMBINE_DESIGN,
                    recipe=recipe,
                    causal_parents=parent_ids,
                )
            return AgentAction()
        if len(parent_ids) < 2:
            return AgentAction()
        if not any(item["combined"] for item in own_proposals):
            return AgentAction(
                verb=ActionType.COMBINE_DESIGN,
                recipe=recipe,
                causal_parents=parent_ids,
            )
        own_batches = [batch for batch in research.batches if batch.agent == agent_id]
        own_tests = [test for test in research.tests if test["agent"] == agent_id]
        required = required_inputs(recipe)
        required_scale = 0.0
        if not own_batches:
            required_scale = 1.0 + simulation.config.science.test_scale
        elif own_tests and simulation.artifacts.count == 0:
            required_scale = 1.0
        if required_scale > 0.0:
            if any(
                float(research.depot[int(resource)]) + 1e-7 < mass * required_scale
                for resource, mass in required.items()
            ):
                return AgentAction()
        x, y = int(simulation.population.x[index]), int(simulation.population.y[index])
        target_x, target_y = self._tester_location(simulation)
        if not self._on_foundry(simulation, x, y):
            return AgentAction(
                verb=ActionType.MOVE,
                direction=self._path_direction(simulation, x, y, target_x, target_y),
            )
        if not own_batches:
            return AgentAction(
                verb=ActionType.OPERATE,
                recipe=recipe,
                causal_parents=parent_ids,
            )
        if not own_tests:
            return AgentAction(verb=ActionType.TEST)
        if simulation.artifacts.count == 0:
            return AgentAction(
                verb=ActionType.BUILD,
                artifact=ArtifactType.MATERIAL_SYSTEM,
                recipe=recipe,
                causal_parents=parent_ids,
                artifact_spec={
                    "name": "Lamellar moisture-buffering collector",
                    "claimed_function": (
                        "Capture atmospheric water under wet conditions while "
                        "remaining a persistent habitat material"
                    ),
                    "architecture": (
                        "A multilayer asymmetric lamellar collector with branching "
                        "microchannels feeding a central storage edge"
                    ),
                    "bio_inspiration": [
                        "fog-basking beetle surface patterning",
                        "kelp lamellae",
                    ],
                    "predicted_effects": [
                        "measurable water storage when local moisture is high",
                        "load-bearing persistence under cyclic damage",
                    ],
                    "geometry": {
                        "layers": 3,
                        "surface_area": 1.7,
                        "channel_density": 0.55,
                        "anisotropy": 0.45,
                        "branching": 0.72,
                        "connectivity": 0.68,
                        "curvature": 0.38,
                        "modularity": 0.55,
                    },
                },
            )
        history = simulation.artifacts.provenance[0]["program_history"]
        if len(history) < 2:
            return AgentAction(
                verb=ActionType.FORK_PROGRAM,
                target_x=x,
                target_y=y,
                program={
                    "name": "evidence_gated_moisture_capture",
                    "parent_program": history[-1]["program"]["program_id"],
                    "instructions": [
                        {"op": "gt", "dest": "r0", "a": "moisture", "b": 0.45},
                        {"op": "mul", "dest": "r1", "a": "r0", "b": 0.05},
                        {"op": "collect_water", "value": "r1"},
                    ],
                },
            )
        if not any(
            record.kind == "validated_result" and record.content.startswith(f"{agent_id} |")
            for record in simulation.archive.records
        ):
            tested = own_tests[-1]
            return AgentAction(
                verb=ActionType.PUBLISH,
                insight={
                    "kind": "validated_result",
                    "content": (
                        f"Controlled microbatch {tested['batch_id']} measured material utility "
                        f"{tested['material_utility']}; the installed program now runs in field."
                    ),
                    "salience": 0.9,
                },
                causal_parents=parent_ids,
            )
        return AgentAction(verb=ActionType.INSPECT)

    @staticmethod
    def _on_foundry(simulation: BioFoundrySimulation, x: int, y: int) -> bool:
        return (
            Terrain(int(simulation.world.terrain[y, x])) == Terrain.FOUNDRY
            or Station(int(simulation.world.stations[y, x])) != Station.NONE
        )

    @staticmethod
    def _tester_location(simulation: BioFoundrySimulation) -> tuple[int, int]:
        ys, xs = np.nonzero(simulation.world.stations == int(Station.TESTER))
        if len(xs):
            return int(xs[0]), int(ys[0])
        ys, xs = np.nonzero(simulation.world.terrain == int(Terrain.FOUNDRY))
        return int(xs[0]), int(ys[0])

    @staticmethod
    def _nearest_resource(
        simulation: BioFoundrySimulation,
        x: int,
        y: int,
        resource: Resource,
    ) -> tuple[int, int]:
        ys, xs = np.nonzero(
            (simulation.world.resource_kind == int(resource))
            & (simulation.world.resource_mass > 0.05)
        )
        distances = np.abs(xs - x) + np.abs(ys - y)
        choice = int(np.argmin(distances))
        return int(xs[choice]), int(ys[choice])

    @staticmethod
    def _path_direction(
        simulation: BioFoundrySimulation,
        start_x: int,
        start_y: int,
        target_x: int,
        target_y: int,
    ) -> Direction:
        if (start_x, start_y) == (target_x, target_y):
            return Direction.STAY
        queue = deque([(start_x, start_y)])
        parent: dict[tuple[int, int], tuple[tuple[int, int], Direction]] = {}
        seen = {(start_x, start_y)}
        directions = (
            (Direction.NORTH, 0, -1),
            (Direction.EAST, 1, 0),
            (Direction.SOUTH, 0, 1),
            (Direction.WEST, -1, 0),
        )
        while queue:
            x, y = queue.popleft()
            for direction, dx, dy in directions:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < simulation.world.width and 0 <= ny < simulation.world.height):
                    continue
                if not simulation.world.walkable[ny, nx] or (nx, ny) in seen:
                    continue
                parent[(nx, ny)] = ((x, y), direction)
                if (nx, ny) == (target_x, target_y):
                    cursor = (nx, ny)
                    while True:
                        previous, first = parent[cursor]
                        if previous == (start_x, start_y):
                            return first
                        cursor = previous
                seen.add((nx, ny))
                queue.append((nx, ny))
        return Direction.STAY

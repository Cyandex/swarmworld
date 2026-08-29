"""Autonomous artifact dynamics that continue without LLM intervention."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import PhysicsConfig
from .materials import PROPERTIES, MaterialBatch
from .programs import ArtifactProgram, ProgramVM, default_program
from .types import ArtifactType, WorldEvent
from .world import BioWorld

SERVICE_NAMES = (
    "water_capture",
    "remediation",
    "structural_support",
    "adaptive_regulation",
    "self_maintenance",
    "ecological_support",
)

def normalize_artifact_spec(
    value: dict[str, Any] | None, batch: MaterialBatch
) -> dict[str, Any]:
    raw = dict(value or {})
    geometry_raw = dict(raw.get("geometry") or {})
    requested_area = float(np.clip(geometry_raw.get("surface_area", 1.0), 0.25, 4.0))
    mass_limited_area = max(0.25, batch.mass * (0.55 + batch.porosity))
    geometry = {
        "layers": int(np.clip(geometry_raw.get("layers", 1), 1, 16)),
        "surface_area": round(min(requested_area, mass_limited_area), 6),
        "requested_surface_area": round(requested_area, 6),
        "channel_density": round(
            float(np.clip(geometry_raw.get("channel_density", 0.25), 0.0, 1.0)), 6
        ),
        "anisotropy": round(
            float(np.clip(geometry_raw.get("anisotropy", 0.25), 0.0, 1.0)), 6
        ),
        "branching": round(
            float(np.clip(geometry_raw.get("branching", 0.25), 0.0, 1.0)), 6
        ),
        "connectivity": round(
            float(np.clip(geometry_raw.get("connectivity", 0.5), 0.0, 1.0)), 6
        ),
        "curvature": round(
            float(np.clip(geometry_raw.get("curvature", 0.25), 0.0, 1.0)), 6
        ),
        "modularity": round(
            float(np.clip(geometry_raw.get("modularity", 0.25), 0.0, 1.0)), 6
        ),
    }
    return {
        "name": str(raw.get("name", "Untitled material system"))[:120],
        "claimed_function": str(raw.get("claimed_function", "unspecified"))[:500],
        "architecture": str(raw.get("architecture", "unspecified"))[:1000],
        "bio_inspiration": [str(item)[:160] for item in raw.get("bio_inspiration", [])[:8]],
        "predicted_effects": [
            str(item)[:200] for item in raw.get("predicted_effects", [])[:8]
        ],
        "geometry": geometry,
    }


class ArtifactSystem:
    def __init__(self, limit: int, physics: PhysicsConfig | None = None) -> None:
        self.limit = limit
        self.physics = physics or PhysicsConfig()
        self.count = 0
        self.kind: NDArray[np.int16] = np.zeros(limit, dtype=np.int16)
        self.x: NDArray[np.int32] = np.zeros(limit, dtype=np.int32)
        self.y: NDArray[np.int32] = np.zeros(limit, dtype=np.int32)
        self.health: NDArray[np.float32] = np.zeros(limit, dtype=np.float32)
        self.maturity: NDArray[np.float32] = np.zeros(limit, dtype=np.float32)
        self.performance: NDArray[np.float32] = np.zeros(limit, dtype=np.float32)
        # Current-program peaks reset on reprogramming; lifetime peaks never do.
        self.peak_performance: NDArray[np.float32] = np.zeros(limit, dtype=np.float32)
        self.lifetime_peak_performance: NDArray[np.float32] = np.zeros(
            limit, dtype=np.float32
        )
        self.storage: NDArray[np.float32] = np.zeros(limit, dtype=np.float32)
        self.reserve: NDArray[np.float32] = np.zeros(limit, dtype=np.float32)
        self.open_fraction: NDArray[np.float32] = np.zeros(limit, dtype=np.float32)
        self.retired: NDArray[np.bool_] = np.zeros(limit, dtype=np.bool_)
        self.fluxes: NDArray[np.float64] = np.zeros((limit, 4), dtype=np.float64)
        self.properties: NDArray[np.float32] = np.zeros((limit, len(PROPERTIES)), dtype=np.float32)
        self.services: NDArray[np.float32] = np.zeros(
            (limit, len(SERVICE_NAMES)), dtype=np.float32
        )
        self.peak_services: NDArray[np.float32] = np.zeros(
            (limit, len(SERVICE_NAMES)), dtype=np.float32
        )
        self.lifetime_peak_services: NDArray[np.float32] = np.zeros(
            (limit, len(SERVICE_NAMES)), dtype=np.float32
        )
        self.lifetime_peak_tick: NDArray[np.int64] = np.full(limit, -1, dtype=np.int64)
        self.lifetime_peak_program_id: list[str] = [""] * limit
        self.created_tick: NDArray[np.int64] = np.zeros(limit, dtype=np.int64)
        self.creator: list[str] = [""] * limit
        self.batch_id: list[str] = [""] * limit
        self.specs: list[dict[str, Any]] = [{} for _ in range(limit)]
        self.provenance: dict[int, dict[str, Any]] = {}
        self.programs: list[ArtifactProgram | None] = [None] * limit
        self.vm = ProgramVM()

    def add(
        self,
        kind: ArtifactType,
        x: int,
        y: int,
        creator: str,
        batch: MaterialBatch,
        tick: int,
        program: ArtifactProgram | None = None,
        artifact_spec: dict[str, Any] | None = None,
    ) -> int:
        if self.count >= self.limit:
            raise RuntimeError("artifact limit reached")
        index = self.count
        self.count += 1
        self.kind[index] = int(kind)
        self.x[index] = x
        self.y[index] = y
        self.health[index] = np.float32(0.68 + 0.30 * batch.properties["quality"])
        self.maturity[index] = np.float32(1.0)
        self.performance[index] = np.float32(0.0)
        self.storage[index] = np.float32(0.0)
        self.reserve[index] = np.float32(batch.properties["healing"])
        self.open_fraction[index] = np.float32(0.5)
        self.retired[index] = False
        self.properties[index] = np.asarray(
            [batch.properties[name] for name in PROPERTIES], dtype=np.float32
        )
        self.created_tick[index] = tick
        self.creator[index] = creator
        self.batch_id[index] = batch.batch_id
        self.specs[index] = normalize_artifact_spec(artifact_spec, batch)
        self.provenance[index] = {
            "artifact_id": f"artifact_{index:08d}",
            "creator": creator,
            "contributors": batch.contributors,
            "causal_parents": batch.causal_parents,
            "batch": batch.as_dict(),
            "artifact_spec": self.specs[index],
            "program_history": [],
        }
        self.install_program(index, program or default_program(kind), tick)
        return index

    def install_program(self, index: int, program: ArtifactProgram, tick: int) -> None:
        if not 0 <= index < self.count:
            raise IndexError(f"unknown artifact index {index}")
        program.validate()
        history = self.provenance[index]["program_history"]
        if history:
            history[-1]["ended_tick"] = int(tick)
            history[-1]["peak_performance"] = round(
                float(self.peak_performance[index]), 6
            )
            history[-1]["peak_services"] = {
                name: round(float(self.peak_services[index, service]), 6)
                for service, name in enumerate(SERVICE_NAMES)
            }
        self.programs[index] = program
        self.peak_performance[index] = np.float32(0.0)
        self.peak_services[index].fill(np.float32(0.0))
        history.append(
            {"tick": tick, "program": program.as_dict()}
        )

    def step(self, world: BioWorld, tick: int) -> list[WorldEvent]:
        if self.count == 0:
            return []
        sl = slice(0, self.count)
        indices = np.arange(self.count)
        x = self.x[sl]
        y = self.y[sl]
        moisture = world.moisture[y, x]
        solar = world.solar[y, x]
        contamination = world.contamination[y, x]
        nutrients = world.nutrients[y, x]
        old_maturity = self.maturity[sl].copy()

        stiffness = self.properties[sl, PROPERTIES.index("stiffness")]
        permeability = self.properties[sl, PROPERTIES.index("permeability")]
        toughness = self.properties[sl, PROPERTIES.index("toughness")]
        adhesion = self.properties[sl, PROPERTIES.index("adhesion")]
        healing = self.properties[sl, PROPERTIES.index("healing")]
        responsiveness = self.properties[sl, PROPERTIES.index("responsiveness")]
        degradation = self.properties[sl, PROPERTIES.index("degradation")]

        program_collect = np.zeros(self.count, dtype=np.float32)
        program_grow = np.zeros(self.count, dtype=np.float32)
        program_heal = np.zeros(self.count, dtype=np.float32)
        program_set_open = np.full(self.count, np.nan, dtype=np.float32)
        program_remediate = np.zeros(self.count, dtype=np.float32)
        program_signal = np.zeros(self.count, dtype=np.float32)
        for index in range(self.count):
            program = self.programs[index]
            if program is None:
                continue
            sensors = {
                "moisture": float(moisture[index]),
                "nutrients": float(nutrients[index]),
                "temperature": float(world.temperature[y[index], x[index]]),
                "solar": float(solar[index]),
                "contamination": float(contamination[index]),
                "health": float(self.health[index]),
                "maturity": float(self.maturity[index]),
                "storage": float(self.storage[index]),
                "reserve": float(self.reserve[index]),
                "open_fraction": float(self.open_fraction[index]),
                "permeability": float(permeability[index]),
                "toughness": float(toughness[index]),
                "healing": float(healing[index]),
                "responsiveness": float(responsiveness[index]),
            }
            outputs = self.vm.execute(program, sensors)
            program_collect[index] = np.float32(outputs.get("collect_water", 0.0))
            program_grow[index] = np.float32(outputs.get("grow", 0.0))
            program_heal[index] = np.float32(outputs.get("heal", 0.0))
            if "set_open" in outputs:
                program_set_open[index] = np.float32(outputs["set_open"])
            program_remediate[index] = np.float32(
                outputs.get("reduce_contamination", 0.0)
            )
            program_signal[index] = np.float32(outputs.get("emit_signal", 0.0))

        area = np.asarray(
            [self.specs[index]["geometry"]["surface_area"] for index in range(self.count)],
            dtype=np.float32,
        )
        channels = np.asarray(
            [self.specs[index]["geometry"]["channel_density"] for index in range(self.count)],
            dtype=np.float32,
        )
        anisotropy = np.asarray(
            [self.specs[index]["geometry"]["anisotropy"] for index in range(self.count)],
            dtype=np.float32,
        )
        branching = np.asarray(
            [self.specs[index]["geometry"]["branching"] for index in range(self.count)],
            dtype=np.float32,
        )
        connectivity = np.asarray(
            [
                self.specs[index]["geometry"]["connectivity"]
                for index in range(self.count)
            ],
            dtype=np.float32,
        )
        curvature = np.asarray(
            [self.specs[index]["geometry"]["curvature"] for index in range(self.count)],
            dtype=np.float32,
        )
        modularity = np.asarray(
            [self.specs[index]["geometry"]["modularity"] for index in range(self.count)],
            dtype=np.float32,
        )
        layers = np.asarray(
            [self.specs[index]["geometry"]["layers"] for index in range(self.count)],
            dtype=np.float32,
        )
        surface = np.clip(
            np.sqrt(area)
            * (0.65 + 0.35 * branching)
            * (0.85 + 0.15 * np.log2(layers + 1.0)),
            0.25,
            2.5,
        )
        structure_factor = np.clip(
            (0.45 + 0.85 * connectivity)
            * (0.8 + 0.2 * anisotropy)
            * (1.0 - 0.2 * channels),
            0.35,
            1.5,
        )
        adaptive_factor = np.clip(
            0.55 + 0.25 * curvature + 0.20 * modularity, 0.55, 1.0
        )
        health = self.health[sl]

        requested_capture = (
            program_collect * moisture * (0.2 + 0.8 * permeability) * surface * health
        )
        requested_growth = program_grow * moisture * (0.2 + 0.8 * healing) * health

        periodic_load = np.float32(0.0025 * (0.5 + 0.5 * np.sin(tick * 0.071)))
        damage = periodic_load * (1.15 - toughness) / np.maximum(0.35, structure_factor)
        self.health[sl] -= damage.astype(np.float32)
        requested_heal = program_heal * (0.2 + 0.8 * healing) * moisture
        requested_signal = program_signal * (0.2 + 0.8 * responsiveness) * health
        captured = np.zeros(self.count, dtype=np.float32)
        growth = np.zeros(self.count, dtype=np.float32)
        bounded_program_heal = np.zeros(self.count, dtype=np.float32)
        remediated = np.zeros(self.count, dtype=np.float32)
        ecological = np.zeros(self.count, dtype=np.float32)
        for index in range(self.count):
            if self.retired[index]:
                continue
            cell_x, cell_y = int(x[index]), int(y[index])
            if self.physics.closed_artifact_fluxes:
                water = min(
                    float(requested_capture[index]),
                    float(world.moisture[cell_y, cell_x]),
                )
                world.moisture[cell_y, cell_x] -= np.float32(water)
                captured[index] = np.float32(water)
                demands = np.asarray(
                    [requested_growth[index], requested_heal[index], requested_signal[index]],
                    dtype=np.float64,
                )
                demand_total = float(demands.sum())
                scale = min(1.0, float(self.reserve[index]) / max(1e-12, demand_total))
                allocated = demands * scale
                growth[index] = np.float32(allocated[0])
                bounded_program_heal[index] = np.float32(allocated[1])
                nutrient_room = max(
                    0.0,
                    self.configured_nutrient_capacity(world)
                    - float(world.nutrients[cell_y, cell_x]),
                )
                ecological[index] = np.float32(min(float(allocated[2]), nutrient_room))
                consumed = float(growth[index] + bounded_program_heal[index] + ecological[index])
                self.reserve[index] -= np.float32(consumed)
            else:
                captured[index] = np.float32(requested_capture[index])
                growth[index] = np.float32(requested_growth[index])
                bounded_program_heal[index] = np.float32(
                    min(float(self.reserve[index]), float(requested_heal[index]))
                )
                self.reserve[index] -= bounded_program_heal[index]
                ecological[index] = np.float32(requested_signal[index])

            requested_remediation = float(
                program_remediate[index]
                * (0.2 + 0.8 * adhesion[index])
                * surface[index]
                * health[index]
            )
            remediated[index] = np.float32(
                min(requested_remediation, float(world.contamination[cell_y, cell_x]))
            )
            world.contamination[cell_y, cell_x] -= remediated[index]
            world.nutrients[cell_y, cell_x] += ecological[index]
            self.fluxes[index, 0] += float(captured[index])
            self.fluxes[index, 1] += float(ecological[index])
            self.fluxes[index, 2] += float(remediated[index])
            self.fluxes[index, 3] += float(
                growth[index] + bounded_program_heal[index] + ecological[index]
            )
            world.flux_ledger["artifact_water_removed"] += float(captured[index])
            world.flux_ledger["artifact_nutrients_added"] += float(ecological[index])
            world.flux_ledger["artifact_contamination_removed"] += float(remediated[index])

        self.storage[sl] += captured
        self.maturity[sl] += growth
        self.health[sl] += bounded_program_heal

        target_open = np.clip(1.0 - solar + 0.25 * moisture, 0.0, 1.0)
        has_open_command = ~np.isnan(program_set_open)
        self.open_fraction[sl][has_open_command] += (
            program_set_open[has_open_command] - self.open_fraction[sl][has_open_command]
        ) * (0.05 + 0.35 * responsiveness[has_open_command])
        tracking = 1.0 - np.abs(self.open_fraction[sl] - target_open)

        service_values = np.column_stack(
            [
                np.clip(captured / 0.04, 0.0, 1.0),
                np.clip(remediated / 0.04, 0.0, 1.0),
                np.clip(
                    health
                    * self.maturity[sl]
                    * structure_factor
                    * (0.45 * stiffness + 0.55 * toughness)
                    * (0.65 + 0.35 * anisotropy)
                    * (1.0 - 0.3 * channels),
                    0.0,
                    1.0,
                ),
                np.clip(
                    has_open_command
                    * tracking
                    * responsiveness
                    * adaptive_factor,
                    0.0,
                    1.0,
                ),
                np.clip(bounded_program_heal / 0.03, 0.0, 1.0),
                np.clip((ecological + growth) / 0.04, 0.0, 1.0),
            ]
        ).astype(np.float32)
        self.services[sl] = service_values
        ordered = np.sort(service_values, axis=1)
        self.performance[sl] = np.clip(
            0.72 * ordered[:, -1] + 0.28 * ordered[:, -2], 0.0, 1.0
        )
        np.maximum(
            self.peak_services[sl], self.services[sl], out=self.peak_services[sl]
        )
        np.maximum(
            self.peak_performance[sl],
            self.performance[sl],
            out=self.peak_performance[sl],
        )
        np.maximum(
            self.lifetime_peak_services[sl],
            self.services[sl],
            out=self.lifetime_peak_services[sl],
        )
        lifetime_improved = self.performance[sl] > self.lifetime_peak_performance[sl]
        if np.any(lifetime_improved):
            improved_indices = indices[lifetime_improved]
            self.lifetime_peak_performance[improved_indices] = self.performance[
                improved_indices
            ]
            self.lifetime_peak_tick[improved_indices] = np.int64(tick)
            for index in improved_indices.tolist():
                program = self.programs[index]
                self.lifetime_peak_program_id[index] = (
                    program.program_id if program is not None else ""
                )

        weathering = degradation * (0.2 + contamination) * np.float32(0.00035)
        self.health[sl] -= weathering.astype(np.float32)
        np.clip(self.health[sl], 0.0, 1.0, out=self.health[sl])
        np.clip(self.maturity[sl], 0.0, 1.0, out=self.maturity[sl])
        np.clip(self.reserve[sl], 0.0, 1.0, out=self.reserve[sl])
        np.clip(self.open_fraction[sl], 0.0, 1.0, out=self.open_fraction[sl])

        events: list[WorldEvent] = []
        for threshold in (0.25, 0.5, 0.75, 1.0):
            crossed = indices[(old_maturity < threshold) & (self.maturity[sl] >= threshold)]
            for index in crossed.tolist():
                events.append(
                    WorldEvent(
                        tick=tick,
                        kind="artifact_milestone",
                        payload={
                            "artifact_id": f"artifact_{index:08d}",
                            "maturity": threshold,
                        },
                    )
                )
        return events

    @staticmethod
    def configured_nutrient_capacity(world: BioWorld) -> float:
        return float(world.config.nutrient_capacity)

    def summary_score(self) -> float:
        if self.count == 0:
            return 0.0
        return float(np.sum(self.performance[: self.count]))

    def snapshot(self, limit: int | None = None) -> dict[str, object]:
        count = self.count if limit is None else min(self.count, limit)
        return {
            "count": self.count,
            "display_count": count,
            "ids": [f"artifact_{index:08d}" for index in range(count)],
            "kind": self.kind[:count].tolist(),
            "x": self.x[:count].tolist(),
            "y": self.y[:count].tolist(),
            "health": np.round(self.health[:count], 4).tolist(),
            "maturity": np.round(self.maturity[:count], 4).tolist(),
            "performance": np.round(self.performance[:count], 4).tolist(),
            "peak_performance": np.round(self.peak_performance[:count], 4).tolist(),
            "lifetime_peak_performance": np.round(
                self.lifetime_peak_performance[:count], 4
            ).tolist(),
            "lifetime_peak_tick": self.lifetime_peak_tick[:count].tolist(),
            "lifetime_peak_program_id": self.lifetime_peak_program_id[:count],
            "storage": np.round(self.storage[:count], 4).tolist(),
            "open_fraction": np.round(self.open_fraction[:count], 4).tolist(),
            "retired": self.retired[:count].tolist(),
            "cumulative_fluxes": {
                "water_captured": np.round(self.fluxes[:count, 0], 6).tolist(),
                "nutrients_released": np.round(self.fluxes[:count, 1], 6).tolist(),
                "contamination_removed": np.round(self.fluxes[:count, 2], 6).tolist(),
                "reserve_consumed": np.round(self.fluxes[:count, 3], 6).tolist(),
            },
            "program": [
                self.programs[index].name if self.programs[index] is not None else ""
                for index in range(count)
            ],
            "program_id": [
                self.programs[index].program_id
                if self.programs[index] is not None
                else ""
                for index in range(count)
            ],
            "name": [self.specs[index].get("name", "") for index in range(count)],
            "claimed_function": [
                self.specs[index].get("claimed_function", "") for index in range(count)
            ],
            "architecture": [
                self.specs[index].get("architecture", "") for index in range(count)
            ],
            "bio_inspiration": [
                self.specs[index].get("bio_inspiration", []) for index in range(count)
            ],
            "geometry": [self.specs[index].get("geometry", {}) for index in range(count)],
            "creator": self.creator[:count],
            "created_tick": self.created_tick[:count].tolist(),
            "causal_parents": [
                list(self.provenance[index].get("causal_parents", []))
                for index in range(count)
            ],
            "services": {
                name: np.round(self.services[:count, service], 4).tolist()
                for service, name in enumerate(SERVICE_NAMES)
            },
            "peak_services": {
                name: np.round(self.peak_services[:count, service], 4).tolist()
                for service, name in enumerate(SERVICE_NAMES)
            },
            "lifetime_peak_services": {
                name: np.round(
                    self.lifetime_peak_services[:count, service], 4
                ).tolist()
                for service, name in enumerate(SERVICE_NAMES)
            },
        }

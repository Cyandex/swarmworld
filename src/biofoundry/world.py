"""Deterministic grid world and vectorized environmental fields."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import WorldConfig
from .types import Resource, Station, Terrain

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int16]


class BioWorld:
    def __init__(self, config: WorldConfig, rng: np.random.Generator):
        self.config = config
        self.width = config.width
        self.height = config.height
        self.terrain: IntArray = np.full(
            (self.height, self.width), int(Terrain.MEADOW), dtype=np.int16
        )
        self.resource_kind: IntArray = np.zeros((self.height, self.width), dtype=np.int16)
        self.resource_mass: FloatArray = np.zeros((self.height, self.width), dtype=np.float32)
        self.resource_capacity: FloatArray = np.zeros((self.height, self.width), dtype=np.float32)
        self.stations: IntArray = np.zeros((self.height, self.width), dtype=np.int16)
        self.temperature: FloatArray = np.zeros((self.height, self.width), dtype=np.float32)
        self.moisture: FloatArray = np.zeros((self.height, self.width), dtype=np.float32)
        self.nutrients: FloatArray = np.zeros((self.height, self.width), dtype=np.float32)
        self.contamination: FloatArray = np.zeros((self.height, self.width), dtype=np.float32)
        self.solar: FloatArray = np.zeros((self.height, self.width), dtype=np.float32)
        self.last_disturbance_mask: NDArray[np.bool_] = np.zeros(
            (self.height, self.width), dtype=np.bool_
        )
        self._disturbance_centers: list[tuple[int, int]] = []
        self._disturbance_sequence: tuple[str, ...] = (
            "contamination_front",
            "drought",
            "storm",
        )
        self.generator_manifest: dict[str, object] = {}
        self.flux_ledger: dict[str, float] = {
            "spring_water_added": 0.0,
            "resource_regrowth_added": 0.0,
            "artifact_water_removed": 0.0,
            "artifact_nutrients_added": 0.0,
            "artifact_contamination_removed": 0.0,
            "disturbance_water_added": 0.0,
            "disturbance_water_removed": 0.0,
            "disturbance_contamination_added": 0.0,
            "dismantled_mass_recovered": 0.0,
            "dismantled_mass_discarded": 0.0,
        }
        self._generate(rng)

    def _generate(self, rng: np.random.Generator) -> None:
        yy, xx = np.mgrid[0 : self.height, 0 : self.width]
        nx = xx / max(1, self.width - 1)
        ny = yy / max(1, self.height - 1)
        if self.config.generator == "procedural":
            for _generation_attempt in range(self.config.procedural_attempts):
                coast_base = float(rng.uniform(0.07, 0.18))
                coast_amplitude = float(rng.uniform(0.015, 0.065))
                coast_frequency = int(rng.integers(2, 7))
                coast_phase = float(rng.uniform(0.0, 2.0 * np.pi))
                centers = rng.uniform([0.22, 0.14], [0.88, 0.86], size=(4, 2))
                radii = rng.uniform([0.11, 0.13], [0.24, 0.29], size=(4, 2))
                foundry_center = rng.uniform([0.38, 0.32], [0.76, 0.68])
                candidate_masks = [
                    ((nx - center[0]) / radius[0]) ** 2
                    + ((ny - center[1]) / radius[1]) ** 2
                    < 1.0
                    for center, radius in zip(centers, radii, strict=True)
                ]
                # Deterministic generate-and-retry: every biome must support several
                # cells even at the minimum world size.
                if all(int(np.count_nonzero(mask)) >= 4 for mask in candidate_masks):
                    break
            else:
                raise RuntimeError("procedural generator could not satisfy biome invariants")
        else:
            _generation_attempt = 0
            coast_base = 0.12
            coast_amplitude = 0.035
            coast_frequency = 5
            coast_phase = 0.0
            centers = np.asarray(
                [[0.81, 0.22], [0.25, 0.78], [0.33, 0.42], [0.82, 0.73]],
                dtype=np.float64,
            )
            radii = np.asarray(
                [[0.22, 0.28], [0.22, 0.25], [0.20, 0.22], [0.14, 0.17]],
                dtype=np.float64,
            )
            foundry_center = np.asarray([0.56, 0.52], dtype=np.float64)

        coast = coast_base + coast_amplitude * np.sin(
            ny * np.pi * coast_frequency + coast_phase
        )
        self.terrain[nx < coast * 0.58] = int(Terrain.DEEP_WATER)
        self.terrain[(nx >= coast * 0.58) & (nx < coast)] = int(Terrain.TIDAL)

        masks = [
            ((nx - center[0]) / radius[0]) ** 2
            + ((ny - center[1]) / radius[1]) ** 2
            < 1.0
            for center, radius in zip(centers, radii, strict=True)
        ]
        fungal, chitin, cellulose, mineral = masks
        self.terrain[fungal] = int(Terrain.FUNGAL_GROVE)
        self.terrain[chitin] = int(Terrain.CHITIN_GARDEN)
        self.terrain[cellulose] = int(Terrain.CELLULOSE_FIELD)
        self.terrain[mineral] = int(Terrain.MINERAL_SPRING)

        cx, cy = (
            int(self.width * float(foundry_center[0])),
            int(self.height * float(foundry_center[1])),
        )
        foundry_radius_x = 2 if self.config.workspace_layout == "distributed" else 6
        foundry_radius_y = 2 if self.config.workspace_layout == "distributed" else 5
        fx0, fx1 = (
            max(1, cx - foundry_radius_x),
            min(self.width - 1, cx + foundry_radius_x + 1),
        )
        fy0, fy1 = (
            max(1, cy - foundry_radius_y),
            min(self.height - 1, cy + foundry_radius_y + 1),
        )
        self.terrain[fy0:fy1, fx0:fx1] = int(Terrain.FOUNDRY)
        self.terrain[
            max(1, cy + 7) : min(self.height - 1, cy + 13),
            max(1, cx - 4) : min(self.width - 1, cx + 10),
        ] = int(Terrain.TEST_FIELD)

        station_cycle = [
            Station.WASHER,
            Station.FERMENTER,
            Station.ALIGNER,
            Station.PRESS,
            Station.TESTER,
            Station.ARCHIVE,
        ]
        station_positions: list[dict[str, object]] = []
        if self.config.workspace_layout == "distributed":
            normalized_targets = [
                tuple(map(float, center)) for center in centers
            ] + [
                (
                    min(0.95, max(0.05, (cx + 3) / max(1, self.width - 1))),
                    min(0.95, max(0.05, (cy + 10) / max(1, self.height - 1))),
                ),
                (float(foundry_center[0]), float(foundry_center[1])),
            ]
            available = self.walkable.copy()
            for station, target in zip(
                station_cycle, normalized_targets, strict=True
            ):
                candidates_y, candidates_x = np.nonzero(available)
                distance = (
                    candidates_x / max(1, self.width - 1) - target[0]
                ) ** 2 + (
                    candidates_y / max(1, self.height - 1) - target[1]
                ) ** 2
                selected = int(np.argmin(distance))
                sx, sy = int(candidates_x[selected]), int(candidates_y[selected])
                self.stations[sy, sx] = int(station)
                available[sy, sx] = False
                station_positions.append(
                    {"station": station.name, "position": [sx, sy]}
                )
        else:
            for offset, station in enumerate(station_cycle):
                sx = fx0 + 2 + (offset % 3) * 4
                sy = fy0 + 2 + (offset // 3) * 5
                if 0 <= sx < self.width and 0 <= sy < self.height:
                    self.stations[sy, sx] = int(station)
                    station_positions.append(
                        {"station": station.name, "position": [sx, sy]}
                    )

        # The laboratory may occlude a small sampled biome. Restore a minimum patch
        # using the nearest remaining meadow cells to that biome's sampled center.
        for terrain, center in zip(
            (
                Terrain.FUNGAL_GROVE,
                Terrain.CHITIN_GARDEN,
                Terrain.CELLULOSE_FIELD,
                Terrain.MINERAL_SPRING,
            ),
            centers,
            strict=True,
        ):
            needed = 2 - int(np.count_nonzero(self.terrain == int(terrain)))
            for _ in range(max(0, needed)):
                candidates_y, candidates_x = np.nonzero(
                    self.terrain == int(Terrain.MEADOW)
                )
                if len(candidates_x) == 0:
                    raise RuntimeError("world generator has no cell available for a biome")
                distance = (
                    candidates_x / max(1, self.width - 1) - float(center[0])
                ) ** 2 + (
                    candidates_y / max(1, self.height - 1) - float(center[1])
                ) ** 2
                selected = int(np.argmin(distance))
                self.terrain[candidates_y[selected], candidates_x[selected]] = int(terrain)

        fungal = self.terrain == int(Terrain.FUNGAL_GROVE)
        chitin = self.terrain == int(Terrain.CHITIN_GARDEN)
        cellulose = self.terrain == int(Terrain.CELLULOSE_FIELD)
        mineral = self.terrain == int(Terrain.MINERAL_SPRING)

        self.generator_manifest = {
            "mode": self.config.generator,
            "generation_attempt": _generation_attempt,
            "coast": {
                "base": round(coast_base, 8),
                "amplitude": round(coast_amplitude, 8),
                "frequency": coast_frequency,
                "phase": round(coast_phase, 8),
            },
            "biomes": [
                {
                    "terrain": terrain.name,
                    "center": [round(float(center[0]), 8), round(float(center[1]), 8)],
                    "radius": [round(float(radius[0]), 8), round(float(radius[1]), 8)],
                }
                for terrain, center, radius in zip(
                    (
                        Terrain.FUNGAL_GROVE,
                        Terrain.CHITIN_GARDEN,
                        Terrain.CELLULOSE_FIELD,
                        Terrain.MINERAL_SPRING,
                    ),
                    centers,
                    radii,
                    strict=True,
                )
            ],
            "foundry_center": [cx, cy],
            "workspace_layout": self.config.workspace_layout,
            "stations": station_positions,
        }

        tidal = self.terrain == int(Terrain.TIDAL)
        tidal_pattern = (xx + 2 * yy) % 3 == 0
        self.resource_kind[tidal & tidal_pattern] = int(Resource.SHELL)
        self.resource_kind[tidal & ~tidal_pattern] = int(Resource.KELP)
        # Sparse walkable tidal pools provide water as matter rather than as a
        # globally available assumption. Agents must discover them empirically.
        tidal_pool = tidal & ((3 * xx + 5 * yy) % 17 == 0)
        if not np.any(tidal_pool):
            tidal_y, tidal_x = np.nonzero(tidal)
            if len(tidal_x):
                tidal_pool[tidal_y[0], tidal_x[0]] = True
        self.resource_kind[tidal_pool] = int(Resource.WATER)
        self.resource_kind[self.terrain == int(Terrain.FUNGAL_GROVE)] = int(Resource.FUNGUS)
        self.resource_kind[self.terrain == int(Terrain.CHITIN_GARDEN)] = int(Resource.CHITIN)
        self.resource_kind[self.terrain == int(Terrain.CELLULOSE_FIELD)] = int(
            Resource.CELLULOSE
        )
        self.resource_kind[self.terrain == int(Terrain.MINERAL_SPRING)] = int(Resource.MINERAL)
        # Rare catalytic deposits occupy a subset of the mineral spring. This keeps
        # every schema material reachable while preserving spatial scarcity.
        catalytic_deposit = (
            (self.terrain == int(Terrain.MINERAL_SPRING))
            & ((7 * xx + 3 * yy) % 19 == 0)
        )
        if not np.any(catalytic_deposit):
            mineral_y, mineral_x = np.nonzero(
                self.terrain == int(Terrain.MINERAL_SPRING)
            )
            if len(mineral_x):
                catalytic_deposit[mineral_y[0], mineral_x[0]] = True
        self.resource_kind[catalytic_deposit] = int(Resource.CATALYST)

        productive = self.resource_kind != int(Resource.NONE)
        capacity_noise = rng.uniform(0.8, 1.2, size=self.terrain.shape).astype(np.float32)
        self.resource_capacity[productive] = 3.0 * capacity_noise[productive]
        self.resource_mass[productive] = self.resource_capacity[productive] * rng.uniform(
            0.55, 1.0, size=int(productive.sum())
        )

        noise = rng.normal(0.0, 0.025, size=self.terrain.shape).astype(np.float32)
        self.temperature[:] = np.clip(0.62 - 0.20 * ny + noise, 0.0, 1.0)
        self.moisture[:] = np.clip(0.35 + 0.45 * (1.0 - nx) + noise, 0.0, 1.0)
        self.moisture[self.terrain == int(Terrain.MINERAL_SPRING)] = 0.95
        self.moisture[self.terrain == int(Terrain.FUNGAL_GROVE)] += 0.15
        self.moisture[:] = np.clip(self.moisture, 0.0, 1.0)
        self.nutrients[:] = np.clip(0.25 + 0.45 * fungal + 0.25 * chitin + noise, 0.0, 1.0)
        self.contamination[:] = np.clip(rng.uniform(0.0, 0.04, self.terrain.shape), 0.0, 1.0)
        self.solar[:] = np.clip(0.75 - 0.30 * fungal + 0.12 * np.sin(nx * np.pi), 0.0, 1.0)

        candidates_y, candidates_x = np.nonzero(self.walkable)
        chosen = rng.choice(len(candidates_x), size=3, replace=False)
        self._disturbance_centers = [
            (int(candidates_x[index]), int(candidates_y[index])) for index in chosen
        ]

    @property
    def walkable(self) -> NDArray[np.bool_]:
        return self.terrain != int(Terrain.DEEP_WATER)

    def configure_evaluation_disturbances(self, seed: int) -> dict[str, object]:
        """Install a deterministic held-out stress schedule on an evaluation copy.

        This method changes only future disturbance locations and order.  Evaluators
        call it on deep copies after agents are frozen, so it cannot shape discovery.
        """

        rng = np.random.default_rng(int(seed))
        candidates_y, candidates_x = np.nonzero(self.walkable)
        if len(candidates_x) < 3:
            raise RuntimeError("held-out evaluation requires three walkable cells")
        chosen = rng.choice(len(candidates_x), size=3, replace=False)
        self._disturbance_centers = [
            (int(candidates_x[index]), int(candidates_y[index])) for index in chosen
        ]
        canonical = np.asarray(
            ["contamination_front", "drought", "storm"], dtype=object
        )
        self._disturbance_sequence = tuple(
            str(item) for item in canonical[rng.permutation(len(canonical))]
        )
        self.last_disturbance_mask.fill(False)
        return {
            "seed": int(seed),
            "centers": [list(center) for center in self._disturbance_centers],
            "sequence": list(self._disturbance_sequence),
        }

    @staticmethod
    def _laplacian(field: FloatArray) -> FloatArray:
        padded = np.pad(field, 1, mode="edge")
        return (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            - 4.0 * field
        )

    def step(self, tick: int) -> list[dict[str, object]]:
        diffusion = np.float32(self.config.field_diffusion)
        self.moisture += diffusion * self._laplacian(self.moisture)
        self.nutrients += diffusion * np.float32(0.35) * self._laplacian(self.nutrients)
        self.contamination += diffusion * np.float32(0.18) * self._laplacian(
            self.contamination
        )
        day_phase = np.float32(0.5 + 0.5 * np.sin(2.0 * np.pi * tick / 512.0))
        base_solar = np.float32(0.25 + 0.75 * day_phase)
        fungal_mask = self.terrain == int(Terrain.FUNGAL_GROVE)
        self.solar[:] = base_solar
        self.solar[fungal_mask] *= np.float32(0.55)
        self.temperature += np.float32(0.015) * (self.solar - self.temperature)
        spring = self.terrain == int(Terrain.MINERAL_SPRING)
        before_spring = self.moisture[spring].copy()
        self.moisture[spring] += np.float32(0.01)
        np.clip(
            self.moisture,
            0.0,
            np.float32(self.config.moisture_capacity),
            out=self.moisture,
        )
        self.flux_ledger["spring_water_added"] += float(
            np.sum(self.moisture[spring] - before_spring, dtype=np.float64)
        )
        np.clip(
            self.nutrients,
            0.0,
            np.float32(self.config.nutrient_capacity),
            out=self.nutrients,
        )
        np.clip(self.contamination, 0.0, 1.0, out=self.contamination)
        np.clip(self.temperature, 0.0, 1.0, out=self.temperature)

        regrowth = np.float32(self.config.resource_regrowth)
        productive = self.resource_kind != int(Resource.NONE)
        before_regrowth = self.resource_mass[productive].copy()
        self.resource_mass[productive] += regrowth * (
            self.resource_capacity[productive] - self.resource_mass[productive]
        )
        np.minimum(self.resource_mass, self.resource_capacity, out=self.resource_mass)
        self.flux_ledger["resource_regrowth_added"] += float(
            np.sum(self.resource_mass[productive] - before_regrowth, dtype=np.float64)
        )

        self.last_disturbance_mask.fill(False)
        interval = self.config.disturbance_interval
        if interval <= 0 or tick <= 0 or tick % interval:
            return []

        episode = tick // interval - 1
        kind = self._disturbance_sequence[episode % len(self._disturbance_sequence)]
        center_x, center_y = self._disturbance_centers[episode % len(self._disturbance_centers)]
        yy, xx = np.mgrid[0 : self.height, 0 : self.width]
        sigma = max(4.0, 0.22 * min(self.width, self.height))
        local = np.exp(
            -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma**2)
        ).astype(np.float32)
        # A weak global front plus a stronger local core makes the challenge
        # ecologically distributed while preserving meaningful spatial variation.
        field = np.float32(self.config.disturbance_intensity) * (
            np.float32(0.35) + np.float32(0.65) * local
        )
        before_moisture = self.moisture.copy()
        before_contamination = self.contamination.copy()
        if kind == "contamination_front":
            self.contamination += field
        elif kind == "drought":
            self.moisture -= np.float32(0.62) * field
            self.temperature += np.float32(0.22) * field
        else:
            self.moisture += np.float32(0.78) * field
            self.temperature -= np.float32(0.12) * field
        np.clip(
            self.moisture,
            0.0,
            np.float32(self.config.moisture_capacity),
            out=self.moisture,
        )
        np.clip(self.contamination, 0.0, 1.0, out=self.contamination)
        np.clip(self.temperature, 0.0, 1.0, out=self.temperature)
        moisture_delta = float(
            np.sum(self.moisture - before_moisture, dtype=np.float64)
        )
        self.flux_ledger[
            "disturbance_water_added" if moisture_delta >= 0 else "disturbance_water_removed"
        ] += abs(moisture_delta)
        contamination_delta = float(
            np.sum(self.contamination - before_contamination, dtype=np.float64)
        )
        if contamination_delta > 0:
            self.flux_ledger["disturbance_contamination_added"] += contamination_delta
        self.last_disturbance_mask[:] = field >= np.float32(0.22)
        return [
            {
                "kind": kind,
                "center": [center_x, center_y],
                "intensity": round(float(self.config.disturbance_intensity), 6),
                "affected_cells": int(np.count_nonzero(self.last_disturbance_mask)),
            }
        ]

    def harvest(self, x: int, y: int, amount: float) -> tuple[Resource, float]:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return Resource.NONE, 0.0
        kind = Resource(int(self.resource_kind[y, x]))
        taken = min(float(self.resource_mass[y, x]), max(0.0, amount))
        self.resource_mass[y, x] -= np.float32(taken)
        return kind, taken

    def snapshot(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "terrain": self.terrain.ravel().tolist(),
            "resource_kind": self.resource_kind.ravel().tolist(),
            "resource_mass": np.round(self.resource_mass.ravel(), 4).tolist(),
            "stations": self.stations.ravel().tolist(),
            "temperature": np.round(self.temperature.ravel(), 4).tolist(),
            "moisture": np.round(self.moisture.ravel(), 4).tolist(),
            "nutrients": np.round(self.nutrients.ravel(), 4).tolist(),
            "contamination": np.round(self.contamination.ravel(), 4).tolist(),
            "solar": np.round(self.solar.ravel(), 4).tolist(),
            "generator_manifest": self.generator_manifest,
            "flux_ledger": {key: round(value, 6) for key, value in self.flux_ledger.items()},
        }

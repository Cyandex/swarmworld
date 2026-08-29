"""Sparse empirical evidence accumulated through an agent's own sensors and actions.

This module deliberately stores observations rather than conclusions.  It never infers
that an unobserved material is absent, recommends a material, or exposes authoritative
world state.  Those interpretations belong to the agent.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .types import Resource, Station, Terrain


@dataclass(slots=True)
class CellEvidence:
    """The most recent passive observation of one visible cell."""

    x: int
    y: int
    tick: int
    terrain: int
    resource: int
    resource_mass: float
    station: int


class EmpiricalKnowledge:
    """A sparse, replayable evidence store for one partially observing agent."""

    def __init__(self, width: int) -> None:
        self.width = width
        self.cells: dict[int, CellEvidence] = {}
        self.extraction_attempts: Counter[int] = Counter()
        self.extraction_successes: Counter[int] = Counter()

    def observe(
        self,
        *,
        x: int,
        y: int,
        tick: int,
        terrain: Terrain,
        resource: Resource,
        resource_mass: float,
        station: Station = Station.NONE,
    ) -> None:
        self.cells[y * self.width + x] = CellEvidence(
            x=x,
            y=y,
            tick=tick,
            terrain=int(terrain),
            resource=int(resource),
            resource_mass=max(0.0, float(resource_mass)),
            station=int(station),
        )

    def record_extraction(self, resource: Resource, amount: float) -> None:
        """Record the observed outcome without interpreting why it happened."""

        key = int(resource)
        self.extraction_attempts[key] += 1
        if amount > 1e-7:
            self.extraction_successes[key] += 1

    def directly_observed_resources(self) -> set[Resource]:
        return {
            Resource(cell.resource)
            for cell in self.cells.values()
            if cell.resource != int(Resource.NONE)
        }

    def directly_observed_stations(self) -> set[Station]:
        return {
            Station(cell.station)
            for cell in self.cells.values()
            if cell.station != int(Station.NONE)
        }

    def directly_observed_terrains(self) -> set[Terrain]:
        return {Terrain(cell.terrain) for cell in self.cells.values()}

    def grounded_resources(
        self,
        inventory: NDArray[np.float32],
        depot: NDArray[np.float32] | None = None,
    ) -> set[Resource]:
        """Return matter backed by direct observation or conserved possession."""

        grounded = self.directly_observed_resources()
        for value in Resource:
            if value == Resource.NONE:
                continue
            if float(inventory[int(value)]) > 1e-7:
                grounded.add(value)
            if depot is not None and float(depot[int(value)]) > 1e-7:
                grounded.add(value)
        return grounded

    def view(
        self,
        *,
        x: int,
        y: int,
        inventory: NDArray[np.float32],
        depot: NDArray[np.float32] | None = None,
        terrain_name: Callable[[int], str] | None = None,
        resource_name: Callable[[int], str] | None = None,
        station_name: Callable[[int], str] | None = None,
    ) -> dict[str, Any]:
        """Produce compact evidence, never a claim about unseen parts of the world."""

        terrain_label = terrain_name or (lambda value: Terrain(value).name)
        resource_label = resource_name or (lambda value: Resource(value).name)
        station_label = station_name or (lambda value: Station(value).name)

        terrain_counts: Counter[str] = Counter()
        terrain_sites: dict[Terrain, list[CellEvidence]] = {}
        sites: dict[Resource, list[CellEvidence]] = {}
        station_sites: dict[Station, list[CellEvidence]] = {}
        for cell in self.cells.values():
            terrain = Terrain(cell.terrain)
            terrain_counts[terrain_label(int(terrain))] += 1
            terrain_sites.setdefault(terrain, []).append(cell)
            resource = Resource(cell.resource)
            if resource != Resource.NONE:
                sites.setdefault(resource, []).append(cell)
            station = Station(cell.station)
            if station != Station.NONE:
                station_sites.setdefault(station, []).append(cell)

        grounded = self.grounded_resources(inventory, depot)
        materials: dict[str, Any] = {}
        for resource in sorted(grounded, key=int):
            basis: list[str] = []
            resource_sites = sites.get(resource, [])
            if resource_sites:
                basis.append("direct_observation")
            if float(inventory[int(resource)]) > 1e-7:
                basis.append("inventory")
            if depot is not None and float(depot[int(resource)]) > 1e-7:
                basis.append("shared_depot")

            entry: dict[str, Any] = {
                "id": int(resource),
                "evidence_basis": basis,
                "observed_sites": len(resource_sites),
            }
            if resource_sites:
                nearest = min(
                    resource_sites,
                    key=lambda cell: (
                        abs(cell.x - x) + abs(cell.y - y),
                        -cell.tick,
                    ),
                )
                entry["nearest_observed_site"] = {
                    "position": [nearest.x, nearest.y],
                    "distance": abs(nearest.x - x) + abs(nearest.y - y),
                    "last_seen_tick": nearest.tick,
                    "last_seen_mass": round(nearest.resource_mass, 3),
                }
            materials[resource_label(int(resource))] = entry

        observed_stations: dict[str, Any] = {}
        for station in sorted(station_sites, key=int):
            cells = station_sites[station]
            nearest = min(
                cells,
                key=lambda cell: (
                    abs(cell.x - x) + abs(cell.y - y),
                    -cell.tick,
                ),
            )
            observed_stations[station_label(int(station))] = {
                "observed_sites": len(cells),
                "nearest_observed_site": {
                    "position": [nearest.x, nearest.y],
                    "distance": abs(nearest.x - x) + abs(nearest.y - y),
                    "last_seen_tick": nearest.tick,
                },
            }

        observed_terrains: dict[str, Any] = {}
        for terrain in sorted(terrain_sites, key=int):
            cells = terrain_sites[terrain]
            nearest = min(
                cells,
                key=lambda cell: (
                    abs(cell.x - x) + abs(cell.y - y),
                    -cell.tick,
                ),
            )
            observed_terrains[terrain_label(int(terrain))] = {
                "observed_sites": len(cells),
                "nearest_observed_site": {
                    "position": [nearest.x, nearest.y],
                    "distance": abs(nearest.x - x) + abs(nearest.y - y),
                    "last_seen_tick": nearest.tick,
                },
            }

        extraction: dict[str, Any] = {}
        for key in sorted(self.extraction_attempts):
            name = (
                "NO_LOCAL_MATERIAL"
                if key == int(Resource.NONE)
                else resource_label(key)
            )
            extraction[name] = {
                "attempts": self.extraction_attempts[key],
                "successes": self.extraction_successes[key],
            }

        return {
            "distinct_cells_observed": len(self.cells),
            "observed_terrain_counts": dict(sorted(terrain_counts.items())),
            "observed_terrains": observed_terrains,
            "grounded_materials": materials,
            "observed_stations": observed_stations,
            "extraction_outcomes": extraction,
            "scope_note": "Unlisted materials are unobserved, not proven absent.",
        }

    def authoritative_state(self) -> dict[str, Any]:
        return {
            "cells": [asdict(self.cells[key]) for key in sorted(self.cells)],
            "extraction_attempts": dict(sorted(self.extraction_attempts.items())),
            "extraction_successes": dict(sorted(self.extraction_successes.items())),
        }

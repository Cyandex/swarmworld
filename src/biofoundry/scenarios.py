"""Validated declarative scenario packages and their deterministic 2-D runtime.

The original :class:`biofoundry.world.BioWorld` remains the default implementation.
Scenario packages deliberately reuse the stable terrain, resource, station, and
operation numeric slots so old observations, replays, and clients retain their wire
shape.  A package supplies new names, physics, geometry, recipes, and presentation for
those slots without executing package-authored Python.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from .config import WorldConfig
from .materials import PROPERTIES, MaterialBatch, RecipeValidationError
from .types import ProcessOperation, Resource, Station, Terrain

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int16]

CATALOG_SLOTS: dict[str, tuple[str, ...]] = {
    "terrains": tuple(item.name for item in Terrain),
    "resources": tuple(item.name for item in Resource),
    "facilities": tuple(item.name for item in Station),
    "operations": tuple(item.name for item in ProcessOperation),
}


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _records(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{label} must be a list of mappings")
    return value


def _resolve_package(value: str | Path) -> Path:
    source = Path(value).expanduser()
    repository = Path(__file__).resolve().parents[2]
    candidates = [source, Path.cwd() / source, repository / source]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and resolved.name == "scenario.yaml":
            return resolved.parent
        if (resolved / "scenario.yaml").is_file():
            return resolved
    raise ValueError(f"scenario package not found: {value}")


@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    root: Path
    manifest: dict[str, Any]
    documents: dict[str, dict[str, Any]]
    package_hash: str
    catalogs: dict[str, tuple[dict[str, Any], ...]]

    @property
    def scenario_id(self) -> str:
        return str(self.manifest["id"])

    @property
    def version(self) -> str:
        return str(self.manifest["version"])

    @property
    def name(self) -> str:
        return str(self.manifest.get("name", self.scenario_id))

    def catalog(self, kind: str) -> tuple[dict[str, Any], ...]:
        return self.catalogs[kind]

    def _lookup(self, kind: str, value: str | int) -> dict[str, Any]:
        records = self.catalog(kind)
        if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
            index = int(value)
            if 0 <= index < len(records):
                return records[index]
        normalized = str(value).strip().upper().replace(" ", "_").replace("-", "_")
        for record in records:
            aliases = {
                str(record["slot"]).upper(),
                str(record["id"]).upper(),
                str(record.get("name", "")).upper().replace(" ", "_"),
            }
            if normalized in aliases:
                return record
        raise ValueError(f"unknown {kind[:-1]} {value!r} in scenario {self.scenario_id}")

    def slot(self, kind: str, value: str | int) -> str:
        return str(self._lookup(kind, value)["slot"])

    def identifier(self, kind: str, value: str | int) -> str:
        return str(self._lookup(kind, value)["id"])

    def display_name(self, kind: str, value: str | int) -> str:
        return str(self._lookup(kind, value).get("name", self.identifier(kind, value)))

    def normalize_recipe(self, recipe: dict[str, Any]) -> dict[str, Any]:
        normalized = copy.deepcopy(recipe)
        for item in normalized.get("inputs", []):
            item["resource"] = self.slot("resources", item.get("resource", ""))
        for step in normalized.get("steps", []):
            step["operation"] = self.slot("operations", step.get("operation", ""))
        return normalized

    def externalize_recipe(self, recipe: dict[str, Any]) -> dict[str, Any]:
        external = copy.deepcopy(recipe)
        for item in external.get("inputs", []):
            item["resource"] = self.identifier("resources", item.get("resource", ""))
        for step in external.get("steps", []):
            step["operation"] = self.identifier("operations", step.get("operation", ""))
        return external

    def normalize_action_mapping(self, action: dict[str, Any]) -> dict[str, Any]:
        normalized = copy.deepcopy(action)
        resource = normalized.get("resource")
        if resource not in (None, "", "NONE", 0):
            normalized["resource"] = self.slot("resources", resource)
        recipe = normalized.get("recipe")
        if isinstance(recipe, dict):
            normalized["recipe"] = self.normalize_recipe(recipe)
        return normalized

    def workspace_supports(
        self,
        terrain: int,
        station: int,
        recipe: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        required = {
            self.identifier("operations", step.get("operation", ""))
            for step in recipe.get("steps", [])
        }
        supported: set[str] = set()
        for kind, value in (("terrains", terrain), ("facilities", station)):
            record = self._lookup(kind, value)
            for operation in record.get("operations", []):
                supported.add(self.identifier("operations", operation))
        missing = sorted(required - supported)
        return not missing, missing

    @property
    def default_recipe(self) -> dict[str, Any]:
        mission = self.documents.get("missions", {})
        recipe = _mapping(mission.get("default_recipe", {}), "missions.default_recipe")
        return self.normalize_recipe(recipe)

    @property
    def resource_names(self) -> tuple[str, ...]:
        return tuple(str(item["id"]) for item in self.catalog("resources"))

    @property
    def operation_names(self) -> tuple[str, ...]:
        return tuple(str(item["id"]) for item in self.catalog("operations"))

    @property
    def agent_prompt(self) -> str:
        prompt_path = self.manifest.get("agent_prompt")
        if not prompt_path:
            return ""
        path = (self.root / str(prompt_path)).resolve()
        if self.root not in path.parents or not path.is_file():
            raise ValueError("scenario agent_prompt must be a file inside the package")
        return path.read_text(encoding="utf-8")

    def descriptor(self) -> dict[str, Any]:
        rendering = self.documents.get("rendering", {})
        analysis = self.documents.get("analysis", {})
        mission = self.documents.get("missions", {})
        return {
            "id": self.scenario_id,
            "name": self.name,
            "version": self.version,
            "package_hash": self.package_hash,
            "format_version": int(self.manifest.get("format_version", 1)),
            "description": str(self.manifest.get("description", "")),
            "catalog_limits": {key: len(value) for key, value in CATALOG_SLOTS.items()},
            "catalogs": {
                kind: [dict(item) for item in records]
                for kind, records in self.catalogs.items()
            },
            "fields": copy.deepcopy(self.documents.get("fields", {}).get("fields", [])),
            "services": copy.deepcopy(rendering.get("services", [])),
            "property_names": copy.deepcopy(analysis.get("property_names", {})),
            "mission": {
                "title": mission.get("title", ""),
                "objective": mission.get("objective", ""),
                "milestones": mission.get("milestones", []),
            },
        }


def _load_definition(root_text: str) -> ScenarioDefinition:
    root = Path(root_text)
    manifest_path = root / "scenario.yaml"
    manifest = _mapping(yaml.safe_load(manifest_path.read_text(encoding="utf-8")), "scenario")
    required = {"id", "version", "documents"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"scenario manifest missing keys: {sorted(missing)}")
    if int(manifest.get("format_version", 1)) != 1:
        raise ValueError("unsupported scenario format_version")
    references = _mapping(manifest["documents"], "scenario.documents")
    documents: dict[str, dict[str, Any]] = {}
    files = [manifest_path]
    for name, relative in references.items():
        path = (root / str(relative)).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"scenario document {name!r} must be inside the package")
        documents[str(name)] = _mapping(
            yaml.safe_load(path.read_text(encoding="utf-8")), str(name)
        )
        files.append(path)
    if manifest.get("agent_prompt"):
        prompt_path = (root / str(manifest["agent_prompt"])).resolve()
        if root not in prompt_path.parents or not prompt_path.is_file():
            raise ValueError("scenario agent_prompt must be a file inside the package")
        files.append(prompt_path)

    catalogs: dict[str, tuple[dict[str, Any], ...]] = {}
    for kind, expected_slots in CATALOG_SLOTS.items():
        document_name = "facilities" if kind == "facilities" else kind
        raw = documents.get(document_name, {}).get(kind, [])
        records = _records(raw, kind)
        by_slot: dict[str, dict[str, Any]] = {}
        identifiers: set[str] = set()
        for record in records:
            slot = str(record.get("slot", "")).upper()
            identifier = str(record.get("id", "")).upper()
            if slot not in expected_slots:
                raise ValueError(f"unknown {kind} slot {slot!r}")
            if not identifier or identifier in identifiers:
                raise ValueError(f"duplicate or empty {kind} id {identifier!r}")
            identifiers.add(identifier)
            by_slot[slot] = {**record, "slot": slot, "id": identifier}
        if set(by_slot) != set(expected_slots):
            missing_slots = sorted(set(expected_slots) - set(by_slot))
            raise ValueError(
                f"scenario {kind} must define every stable slot; missing {missing_slots}"
            )
        catalogs[kind] = tuple(by_slot[slot] for slot in expected_slots)

    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    definition = ScenarioDefinition(root, manifest, documents, digest.hexdigest(), catalogs)
    _validate_definition(definition)
    return definition


@lru_cache(maxsize=16)
def _cached_definition(root_text: str) -> ScenarioDefinition:
    return _load_definition(root_text)


def load_scenario(value: str | Path | None) -> ScenarioDefinition | None:
    if value is None or not str(value).strip():
        return None
    return _cached_definition(str(_resolve_package(value)))


def validate_scenario(value: str | Path) -> dict[str, Any]:
    definition = _load_definition(str(_resolve_package(value)))
    return definition.descriptor()


def _validate_definition(definition: ScenarioDefinition) -> None:
    geometry = definition.documents.get("geometry", {})
    definition.slot("terrains", str(geometry.get("base_terrain", "")))
    for feature in _records(geometry.get("features", []), "geometry.features"):
        definition.slot("terrains", feature.get("terrain", ""))
        _mapping(feature.get("shape", {}), "geometry feature shape")
    for kind in ("terrains", "facilities"):
        for record in definition.catalog(kind):
            for operation in record.get("operations", []):
                definition.slot("operations", operation)
    resources = definition.documents.get("resources", {})
    for deposit in _records(resources.get("deposits", []), "resources.deposits"):
        definition.slot("resources", deposit.get("resource", ""))
        _mapping(deposit.get("shape", {}), "resource deposit shape")
    facilities = definition.documents.get("facilities", {})
    for placement in _records(facilities.get("placements", []), "facilities.placements"):
        definition.slot("facilities", placement.get("facility", ""))
        position = placement.get("position")
        if not isinstance(position, list) or len(position) != 2:
            raise ValueError("facility placement position must be [x_fraction, y_fraction]")
    fields = _records(
        definition.documents.get("fields", {}).get("fields", []), "fields.fields"
    )
    field_ids = {str(item.get("id", "")) for item in fields}
    required_fields = {
        "temperature",
        "water_availability",
        "ground_stability",
        "toxic_gas",
        "solar",
    }
    if not required_fields <= field_ids:
        missing_fields = sorted(required_fields - field_ids)
        raise ValueError(f"scenario fields missing compatibility fields {missing_fields}")
    _ = definition.default_recipe


def _shape_mask(
    shape: dict[str, Any],
    nx: NDArray[np.float64],
    ny: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.bool_]:
    kind = str(shape.get("kind", "circle"))
    center = shape.get("center", [0.5, 0.5])
    cx, cy = float(center[0]), float(center[1])
    if kind == "circle":
        radius = float(shape.get("radius", 0.2))
        return (nx - cx) ** 2 + (ny - cy) ** 2 <= radius**2
    if kind == "ring":
        distance = np.sqrt((nx - cx) ** 2 + (ny - cy) ** 2)
        return (distance >= float(shape.get("inner", 0.1))) & (
            distance <= float(shape.get("outer", 0.2))
        )
    if kind == "ellipse":
        radius = shape.get("radius", [0.2, 0.15])
        return ((nx - cx) / float(radius[0])) ** 2 + (
            (ny - cy) / float(radius[1])
        ) ** 2 <= 1.0
    if kind == "rectangle":
        bounds = shape.get("bounds", [0.2, 0.2, 0.8, 0.8])
        return (
            (nx >= float(bounds[0]))
            & (ny >= float(bounds[1]))
            & (nx <= float(bounds[2]))
            & (ny <= float(bounds[3]))
        )
    if kind == "ridge":
        start = shape.get("start", [0.1, 0.1])
        end = shape.get("end", [0.9, 0.9])
        ax, ay = float(start[0]), float(start[1])
        bx, by = float(end[0]), float(end[1])
        dx, dy = bx - ax, by - ay
        length_sq = max(1e-9, dx * dx + dy * dy)
        projection = np.clip(((nx - ax) * dx + (ny - ay) * dy) / length_sq, 0.0, 1.0)
        distance = np.sqrt((nx - (ax + projection * dx)) ** 2 + (ny - (ay + projection * dy)) ** 2)
        return distance <= float(shape.get("width", 0.04))
    if kind == "noise":
        probability = float(shape.get("probability", 0.2))
        return rng.random(nx.shape) < probability
    raise ValueError(f"unsupported declarative shape kind {kind!r}")


class ScenarioWorld:
    """Deterministic rectangular world driven entirely by validated package data."""

    def __init__(
        self,
        config: WorldConfig,
        rng: np.random.Generator,
        scenario: ScenarioDefinition,
    ) -> None:
        self.config = config
        self.scenario = scenario
        self.width = config.width
        self.height = config.height
        shape = (self.height, self.width)
        self.terrain: IntArray = np.zeros(shape, dtype=np.int16)
        self.resource_kind: IntArray = np.zeros(shape, dtype=np.int16)
        self.resource_mass: FloatArray = np.zeros(shape, dtype=np.float32)
        self.resource_capacity: FloatArray = np.zeros(shape, dtype=np.float32)
        self.stations: IntArray = np.zeros(shape, dtype=np.int16)
        self.fields: dict[str, FloatArray] = {}
        self._field_specs: dict[str, dict[str, Any]] = {}
        self._resource_regrowth: FloatArray = np.zeros(shape, dtype=np.float32)
        self.last_disturbance_mask: NDArray[np.bool_] = np.zeros(shape, dtype=np.bool_)
        self._disturbance_centers: list[tuple[int, int]] = []
        self._disturbances = list(
            scenario.documents.get("disturbances", {}).get("disturbances", [])
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
        nx = xx.astype(np.float64) / max(1, self.width - 1)
        ny = yy.astype(np.float64) / max(1, self.height - 1)
        geometry = self.scenario.documents["geometry"]
        base = Terrain[self.scenario.slot("terrains", geometry["base_terrain"])]
        self.terrain.fill(int(base))
        feature_manifest: list[dict[str, Any]] = []
        for feature in geometry.get("features", []):
            mask = _shape_mask(feature["shape"], nx, ny, rng)
            terrain = Terrain[self.scenario.slot("terrains", feature["terrain"])]
            self.terrain[mask] = int(terrain)
            feature_manifest.append(
                {
                    "name": str(feature.get("name", feature["terrain"])),
                    "terrain": self.scenario.identifier("terrains", int(terrain)),
                    "cells": int(np.count_nonzero(mask)),
                }
            )

        for placement in self.scenario.documents["facilities"].get("placements", []):
            px, py = placement["position"]
            x = int(round(float(px) * (self.width - 1)))
            y = int(round(float(py) * (self.height - 1)))
            station = Station[self.scenario.slot("facilities", placement["facility"])]
            if not self.walkable[y, x]:
                candidates_y, candidates_x = np.nonzero(self.walkable)
                nearest = int(np.argmin((candidates_x - x) ** 2 + (candidates_y - y) ** 2))
                x, y = int(candidates_x[nearest]), int(candidates_y[nearest])
            self.stations[y, x] = int(station)

        for deposit in self.scenario.documents["resources"].get("deposits", []):
            mask = _shape_mask(deposit["shape"], nx, ny, rng) & self.walkable
            resource = Resource[self.scenario.slot("resources", deposit["resource"])]
            self.resource_kind[mask] = int(resource)
            capacity = float(deposit.get("capacity", 3.0))
            variation = float(deposit.get("capacity_variation", 0.15))
            values = capacity * rng.uniform(1.0 - variation, 1.0 + variation, size=int(mask.sum()))
            self.resource_capacity[mask] = values.astype(np.float32)
            fill = deposit.get("initial_fill", [0.65, 1.0])
            self.resource_mass[mask] = self.resource_capacity[mask] * rng.uniform(
                float(fill[0]), float(fill[1]), size=int(mask.sum())
            )
            self._resource_regrowth[mask] = np.float32(deposit.get("regrowth", 0.0))

        for spec in self.scenario.documents["fields"].get("fields", []):
            field_id = str(spec["id"])
            initial = spec.get("initial", {})
            values = np.full((self.height, self.width), float(initial.get("constant", 0.0)))
            values += float(initial.get("gradient_x", 0.0)) * nx
            values += float(initial.get("gradient_y", 0.0)) * ny
            for radial in initial.get("radial", []):
                center = radial.get("center", [0.5, 0.5])
                sigma = max(1e-6, float(radial.get("sigma", 0.2)))
                distance_sq = (nx - float(center[0])) ** 2 + (ny - float(center[1])) ** 2
                values += float(radial.get("amplitude", 1.0)) * np.exp(
                    -distance_sq / (2.0 * sigma**2)
                )
            noise = float(initial.get("noise", 0.0))
            if noise:
                values += rng.normal(0.0, noise, size=values.shape)
            for terrain_name, offset in initial.get("terrain_offsets", {}).items():
                terrain = Terrain[self.scenario.slot("terrains", terrain_name)]
                values[self.terrain == int(terrain)] += float(offset)
            low, high = spec.get("range", [0.0, 1.0])
            self.fields[field_id] = np.clip(values, float(low), float(high)).astype(np.float32)
            self._field_specs[field_id] = dict(spec)

        self.temperature = self.fields["temperature"]
        self.moisture = self.fields["water_availability"]
        self.nutrients = self.fields["ground_stability"]
        self.contamination = self.fields["toxic_gas"]
        self.solar = self.fields["solar"]

        candidates_y, candidates_x = np.nonzero(self.walkable)
        count = min(max(1, len(self._disturbances)), len(candidates_x))
        chosen = rng.choice(len(candidates_x), size=count, replace=False)
        self._disturbance_centers = [
            (int(candidates_x[index]), int(candidates_y[index])) for index in np.atleast_1d(chosen)
        ]
        self.generator_manifest = {
            "mode": "scenario_package",
            "scenario": self.scenario.descriptor(),
            "features": feature_manifest,
            "facilities": [
                {
                    "facility": self.scenario.identifier("facilities", int(value)),
                    "position": [int(x), int(y)],
                }
                for y, x in zip(*np.nonzero(self.stations), strict=True)
                if int(value := self.stations[y, x]) != int(Station.NONE)
            ],
        }

    @property
    def walkable(self) -> NDArray[np.bool_]:
        flags = np.asarray(
            [bool(item.get("walkable", True)) for item in self.scenario.catalog("terrains")],
            dtype=np.bool_,
        )
        return flags[self.terrain]

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

    def configure_evaluation_disturbances(self, seed: int) -> dict[str, object]:
        rng = np.random.default_rng(int(seed))
        candidates_y, candidates_x = np.nonzero(self.walkable)
        count = min(max(1, len(self._disturbances)), len(candidates_x))
        chosen = rng.choice(len(candidates_x), size=count, replace=False)
        self._disturbance_centers = [
            (int(candidates_x[index]), int(candidates_y[index])) for index in np.atleast_1d(chosen)
        ]
        if self._disturbances:
            order = rng.permutation(len(self._disturbances))
            self._disturbances = [self._disturbances[int(index)] for index in order]
        self.last_disturbance_mask.fill(False)
        return {
            "seed": int(seed),
            "centers": [list(center) for center in self._disturbance_centers],
            "sequence": [str(item.get("id", "disturbance")) for item in self._disturbances],
        }

    def step(self, tick: int) -> list[dict[str, object]]:
        for field_id, field in self.fields.items():
            spec = self._field_specs[field_id]
            diffusion = float(spec.get("diffusion", 0.0))
            if diffusion:
                field += np.float32(diffusion) * self._laplacian(field)
            decay = float(spec.get("decay", 0.0))
            if decay:
                field *= np.float32(max(0.0, 1.0 - decay))
            for terrain_name, amount in spec.get("sources", {}).items():
                terrain = Terrain[self.scenario.slot("terrains", terrain_name)]
                field[self.terrain == int(terrain)] += np.float32(amount)
            if spec.get("day_cycle"):
                phase = 0.5 + 0.5 * np.sin(2.0 * np.pi * tick / float(spec.get("period", 512)))
                field[:] = np.float32(spec.get("night", 0.2) + phase * spec.get("amplitude", 0.8))
            low, high = spec.get("range", [0.0, 1.0])
            np.clip(field, float(low), float(high), out=field)

        productive = self.resource_kind != int(Resource.NONE)
        before = self.resource_mass[productive].copy()
        self.resource_mass[productive] += self._resource_regrowth[productive] * (
            self.resource_capacity[productive] - self.resource_mass[productive]
        )
        np.minimum(self.resource_mass, self.resource_capacity, out=self.resource_mass)
        self.flux_ledger["resource_regrowth_added"] += float(
            np.sum(self.resource_mass[productive] - before, dtype=np.float64)
        )

        self.last_disturbance_mask.fill(False)
        interval = self.config.disturbance_interval
        if not self._disturbances or interval <= 0 or tick <= 0 or tick % interval:
            return []
        episode = tick // interval - 1
        disturbance = self._disturbances[episode % len(self._disturbances)]
        center_x, center_y = self._disturbance_centers[
            episode % len(self._disturbance_centers)
        ]
        yy, xx = np.mgrid[0 : self.height, 0 : self.width]
        sigma = float(disturbance.get("radius", 0.18)) * min(self.width, self.height)
        local = np.exp(
            -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * max(1.0, sigma) ** 2)
        ).astype(np.float32)
        intensity = np.float32(self.config.disturbance_intensity)
        field = intensity * local
        for field_id, delta in disturbance.get("field_deltas", {}).items():
            if field_id not in self.fields:
                raise ValueError(f"disturbance references unknown field {field_id!r}")
            self.fields[field_id] += field * np.float32(delta)
            spec = self._field_specs[field_id]
            low, high = spec.get("range", [0.0, 1.0])
            np.clip(self.fields[field_id], float(low), float(high), out=self.fields[field_id])
        transform = disturbance.get("terrain_transform")
        threshold = float(disturbance.get("transform_threshold", 0.72))
        if transform:
            terrain = Terrain[self.scenario.slot("terrains", transform)]
            self.terrain[field >= threshold * intensity] = int(terrain)
        self.last_disturbance_mask[:] = field >= np.float32(0.22)
        return [
            {
                "kind": str(disturbance.get("id", "scenario_disturbance")),
                "name": str(disturbance.get("name", disturbance.get("id", "disturbance"))),
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

    def field_values(self, x: int, y: int) -> dict[str, float]:
        return {key: round(float(value[y, x]), 4) for key, value in self.fields.items()}

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
            "fields": {
                key: np.round(value.ravel(), 4).tolist() for key, value in self.fields.items()
            },
            "scenario": self.scenario.descriptor(),
            "generator_manifest": self.generator_manifest,
            "flux_ledger": {key: round(value, 6) for key, value in self.flux_ledger.items()},
        }


class ScenarioMaterialLab:
    """Safe package-defined metallurgy surrogate with the legacy batch interface."""

    def __init__(self, scenario: ScenarioDefinition) -> None:
        self.scenario = scenario
        self.next_batch = 0
        analysis = scenario.documents.get("analysis", {})
        self.component_names = tuple(
            analysis.get(
                "composition_names",
                ["iron", "copper", "carbon", "silicate", "flux", "volatile"],
            )
        )
        if len(self.component_names) != 6:
            raise ValueError("scenario analysis.composition_names must contain six names")

    def validate(self, recipe: dict[str, Any]) -> None:
        normalized = self.scenario.normalize_recipe(recipe)
        inputs = normalized.get("inputs", [])
        steps = normalized.get("steps", [])
        if not 1 <= len(inputs) <= 8:
            raise RecipeValidationError("a scenario recipe must contain 1 to 8 inputs")
        if not 1 <= len(steps) <= 12:
            raise RecipeValidationError("a scenario recipe must contain 1 to 12 process steps")
        seen: set[str] = set()
        for item in inputs:
            slot = self.scenario.slot("resources", item.get("resource", ""))
            mass = float(item.get("mass", 0.0))
            if slot == "NONE" or slot in seen or not 0.0 < mass <= 20.0:
                raise RecipeValidationError(f"invalid or duplicate scenario input: {item!r}")
            seen.add(slot)
        for step in steps:
            self.scenario.slot("operations", step.get("operation", ""))
            intensity = float(step.get("intensity", -1.0))
            if not 0.0 <= intensity <= 1.0:
                raise RecipeValidationError(f"step intensity outside [0, 1]: {step!r}")

    def required_inputs(self, recipe: dict[str, Any]) -> dict[Resource, float]:
        self.validate(recipe)
        required: dict[Resource, float] = {}
        for item in self.scenario.normalize_recipe(recipe)["inputs"]:
            resource = Resource[str(item["resource"])]
            required[resource] = required.get(resource, 0.0) + float(item["mass"])
        return required

    def can_execute(self, recipe: dict[str, Any], inventory: FloatArray) -> bool:
        try:
            required = self.required_inputs(recipe)
        except RecipeValidationError:
            return False
        return all(
            float(inventory[int(resource)]) + 1e-7 >= mass
            for resource, mass in required.items()
        )

    def execute(
        self,
        recipe: dict[str, Any],
        inventory: FloatArray,
        tick: int,
        contributors: list[str],
        causal_parents: list[str] | None = None,
    ) -> MaterialBatch:
        normalized = self.scenario.normalize_recipe(recipe)
        required = self.required_inputs(normalized)
        if not self.can_execute(normalized, inventory):
            raise RecipeValidationError("inventory does not satisfy scenario recipe inputs")
        total_mass = sum(required.values())
        composition = np.zeros(6, dtype=np.float32)
        for resource, mass in required.items():
            record = self.scenario._lookup("resources", int(resource))
            values = record.get("composition", [0, 0, 0, 1, 0, 0])
            if not isinstance(values, list) or len(values) != 6:
                raise RecipeValidationError(f"resource {record['id']} needs six composition values")
            composition += np.asarray(values, dtype=np.float32) * np.float32(mass)
        composition /= np.float32(total_mass)
        state = {
            "purity": float(np.clip(0.25 + 0.65 * (composition[0] + composition[1]), 0, 1)),
            "porosity": 0.48,
            "grain_refinement": 0.18,
            "work_hardening": 0.05,
            "thermal_control": 0.12,
            "oxidation": float(np.clip(0.45 + 0.35 * composition[5], 0, 1)),
            "quality": 0.42,
        }
        severity = 0.0
        operation_records = {
            str(item["slot"]): item for item in self.scenario.catalog("operations")
        }
        for step in normalized["steps"]:
            intensity = float(step["intensity"])
            severity += intensity
            effects = operation_records[str(step["operation"])].get("effects", {})
            for key, coefficient in effects.items():
                if key in state:
                    state[key] += float(coefficient) * intensity
        state["quality"] -= 0.012 * max(0.0, severity - 6.0) ** 2
        for key in state:
            state[key] = float(np.clip(state[key], 0.0, 1.0))
        iron, copper, carbon, silicate, flux, volatile = map(float, composition)
        metallic = float(np.clip(iron + copper, 0.0, 1.0))
        strength = metallic * (
            0.35
            + 0.45 * state["work_hardening"]
            + 0.20 * state["grain_refinement"]
        )
        strength *= 1.0 - 0.58 * state["porosity"]
        toughness = metallic * (0.35 + 0.38 * state["grain_refinement"])
        toughness *= 1.0 - 0.45 * state["oxidation"]
        heat_resistance = 0.45 * silicate + 0.25 * carbon + 0.30 * state["thermal_control"]
        corrosion_resistance = 0.5 * state["purity"] + 0.3 * copper - 0.35 * state["oxidation"]
        properties = {
            "stiffness": float(np.clip(strength, 0, 1)),
            "toughness": float(np.clip(toughness, 0, 1)),
            "permeability": float(np.clip(heat_resistance, 0, 1)),
            "adhesion": float(np.clip(corrosion_resistance, 0, 1)),
            "healing": float(np.clip(0.10 + 0.25 * flux, 0, 1)),
            "responsiveness": float(np.clip(state["thermal_control"], 0, 1)),
            "degradation": float(np.clip(state["oxidation"] + 0.3 * volatile, 0, 1)),
            "quality": state["quality"],
        }
        for resource, mass in required.items():
            inventory[int(resource)] -= np.float32(mass)
        batch = MaterialBatch(
            batch_id=f"batch_{self.next_batch:08d}",
            tick=tick,
            mass=total_mass * max(0.65, 0.96 - 0.04 * severity - 0.08 * volatile),
            composition=composition,
            hydration=state["thermal_control"],
            porosity=state["porosity"],
            alignment=state["grain_refinement"],
            crosslink=state["work_hardening"],
            properties={name: properties[name] for name in PROPERTIES},
            recipe=self.scenario.externalize_recipe(normalized),
            contributors=sorted(set(contributors)),
            causal_parents=list(causal_parents or []),
            component_names=self.component_names,
            process_state=state,
        )
        self.next_batch += 1
        return batch


def scenario_metadata(scenario: ScenarioDefinition | None) -> dict[str, Any]:
    if scenario is None:
        return {"id": "legacy_biofoundry", "version": "1", "backend": "legacy"}
    return {
        "id": scenario.scenario_id,
        "name": scenario.name,
        "version": scenario.version,
        "package_hash": scenario.package_hash,
        "backend": "declarative_slots_v1",
    }


def scenario_json(definition: ScenarioDefinition) -> str:
    return json.dumps(definition.descriptor(), indent=2, sort_keys=True)

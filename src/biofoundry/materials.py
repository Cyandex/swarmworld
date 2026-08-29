"""Typed material recipes and transparent game-level surrogate properties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .types import ArtifactType, ProcessOperation, Resource

COMPONENTS = ("cellulose", "chitin", "protein", "mineral", "lignin", "water")
PROPERTIES = (
    "stiffness",
    "toughness",
    "permeability",
    "adhesion",
    "healing",
    "responsiveness",
    "degradation",
    "quality",
)


def _composition(*values: float) -> NDArray[np.float32]:
    return np.asarray(values, dtype=np.float32)


FEEDSTOCK_COMPOSITION: dict[Resource, NDArray[np.float32]] = {
    Resource.KELP: _composition(0.34, 0.02, 0.14, 0.08, 0.02, 0.40),
    Resource.SHELL: _composition(0.01, 0.10, 0.05, 0.78, 0.00, 0.06),
    Resource.FUNGUS: _composition(0.18, 0.24, 0.22, 0.04, 0.08, 0.24),
    Resource.CHITIN: _composition(0.04, 0.72, 0.10, 0.04, 0.02, 0.08),
    Resource.CELLULOSE: _composition(0.78, 0.02, 0.02, 0.02, 0.12, 0.04),
    Resource.MINERAL: _composition(0.01, 0.00, 0.01, 0.92, 0.00, 0.06),
    Resource.WATER: _composition(0.00, 0.00, 0.00, 0.00, 0.00, 1.00),
    Resource.CATALYST: _composition(0.04, 0.04, 0.42, 0.42, 0.02, 0.06),
}


class RecipeValidationError(ValueError):
    pass


@dataclass(slots=True)
class MaterialBatch:
    batch_id: str
    tick: int
    mass: float
    composition: NDArray[np.float32]
    hydration: float
    porosity: float
    alignment: float
    crosslink: float
    properties: dict[str, float]
    recipe: dict[str, Any]
    contributors: list[str]
    causal_parents: list[str]
    component_names: tuple[str, ...] = COMPONENTS
    process_state: dict[str, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "tick": self.tick,
            "mass": round(self.mass, 6),
            "composition": {
                name: round(float(value), 6)
                for name, value in zip(self.component_names, self.composition, strict=True)
            },
            "hydration": round(self.hydration, 6),
            "porosity": round(self.porosity, 6),
            "alignment": round(self.alignment, 6),
            "crosslink": round(self.crosslink, 6),
            "properties": {key: round(value, 6) for key, value in self.properties.items()},
            "recipe": self.recipe,
            "contributors": self.contributors,
            "causal_parents": self.causal_parents,
            **(
                {
                    "process_state": {
                        key: round(float(value), 6)
                        for key, value in self.process_state.items()
                    }
                }
                if self.process_state is not None
                else {}
            ),
        }


DEFAULT_RECIPES: dict[ArtifactType, dict[str, Any]] = {
    ArtifactType.MATERIAL_SYSTEM: {
        "inputs": [
            {"resource": "KELP", "mass": 1.2},
            {"resource": "SHELL", "mass": 0.4},
        ],
        "steps": [
            {"operation": "WASH", "intensity": 0.7},
            {"operation": "ALIGN", "intensity": 0.75},
            {"operation": "PRESS", "intensity": 0.45},
            {"operation": "DRY", "intensity": 0.55},
        ],
        "output_form": "patterned_membrane",
        "design_principles": ["directional_wettability", "hierarchical_channels"],
    },
}


def required_inputs(recipe: dict[str, Any]) -> dict[Resource, float]:
    required: dict[Resource, float] = {}
    for item in recipe.get("inputs", []):
        try:
            resource = Resource[str(item["resource"]).upper()]
        except (KeyError, TypeError) as exc:
            raise RecipeValidationError(f"unknown input resource: {item!r}") from exc
        mass = float(item.get("mass", 0.0))
        if resource == Resource.NONE or not 0.0 < mass <= 20.0:
            raise RecipeValidationError(f"invalid input mass: {item!r}")
        required[resource] = required.get(resource, 0.0) + mass
    if not required:
        raise RecipeValidationError("a recipe must contain at least one input")
    return required


class MaterialLab:
    """Executes conservative recipes against an agent inventory."""

    def __init__(self) -> None:
        self.next_batch = 0

    @staticmethod
    def validate(recipe: dict[str, Any]) -> None:
        required_inputs(recipe)
        steps = recipe.get("steps", [])
        if not 1 <= len(steps) <= 12:
            raise RecipeValidationError("a recipe must contain 1 to 12 process steps")
        for step in steps:
            try:
                ProcessOperation[str(step["operation"]).upper()]
            except (KeyError, TypeError) as exc:
                raise RecipeValidationError(f"unknown process operation: {step!r}") from exc
            intensity = float(step.get("intensity", -1.0))
            if not 0.0 <= intensity <= 1.0:
                raise RecipeValidationError(f"step intensity outside [0, 1]: {step!r}")

    @staticmethod
    def can_execute(recipe: dict[str, Any], inventory: NDArray[np.float32]) -> bool:
        try:
            required = required_inputs(recipe)
        except RecipeValidationError:
            return False
        return all(
            float(inventory[int(resource)]) + 1e-7 >= mass
            for resource, mass in required.items()
        )

    def execute(
        self,
        recipe: dict[str, Any],
        inventory: NDArray[np.float32],
        tick: int,
        contributors: list[str],
        causal_parents: list[str] | None = None,
    ) -> MaterialBatch:
        self.validate(recipe)
        required = required_inputs(recipe)
        if not self.can_execute(recipe, inventory):
            raise RecipeValidationError("inventory does not satisfy recipe inputs")

        total_mass = sum(required.values())
        composition = np.zeros(len(COMPONENTS), dtype=np.float32)
        for resource, mass in required.items():
            composition += FEEDSTOCK_COMPOSITION[resource] * np.float32(mass)
        composition /= np.float32(total_mass)

        hydration = float(composition[5])
        porosity = 0.45
        alignment = 0.15
        crosslink = 0.10
        quality = 0.55
        operation_counts: dict[ProcessOperation, int] = {}
        process_severity = 0.0
        for step in recipe["steps"]:
            operation = ProcessOperation[str(step["operation"]).upper()]
            intensity = float(step["intensity"])
            prior_uses = operation_counts.get(operation, 0)
            operation_counts[operation] = prior_uses + 1
            # Repeating a unit operation gives diminishing returns and eventually
            # introduces handling damage. This prevents recipe length from being a
            # hidden monotonic objective while retaining freedom to repeat a process.
            effective = intensity / (1.0 + 0.75 * prior_uses)
            process_severity += intensity
            quality -= 0.009 * intensity * prior_uses
            cellulose, chitin, protein, mineral, lignin, _water = map(
                float, composition
            )
            fibrous = float(np.clip(cellulose + chitin + lignin, 0.0, 1.0))
            biological = float(np.clip(cellulose + chitin + protein, 0.0, 1.0))
            if operation == ProcessOperation.WASH:
                quality += 0.10 * effective
                hydration += 0.12 * effective
            elif operation == ProcessOperation.GRIND:
                porosity += 0.20 * effective
                alignment -= 0.08 * effective
                quality -= 0.015 * intensity**2
            elif operation == ProcessOperation.FERMENT:
                fermentation = effective * biological * (0.25 + 0.75 * hydration)
                crosslink += 0.28 * fermentation
                quality += 0.07 * fermentation
            elif operation == ProcessOperation.ALKALINE_TREAT:
                treatment = effective * (0.25 + 0.75 * (chitin + lignin))
                crosslink += 0.16 * treatment
                hydration += 0.06 * treatment
            elif operation == ProcessOperation.MINERALIZE:
                nucleation = effective * (0.2 + 0.8 * (mineral + protein))
                composition[3] += np.float32(0.10 * nucleation)
                crosslink += 0.18 * nucleation
            elif operation == ProcessOperation.ALIGN:
                alignment += 0.48 * effective * fibrous
                porosity -= 0.10 * effective * fibrous
            elif operation == ProcessOperation.WEAVE:
                alignment += 0.20 * effective * fibrous
                toughness_bonus = 0.10 * effective * fibrous
                quality += toughness_bonus
            elif operation == ProcessOperation.PRESS:
                porosity -= 0.32 * effective
                crosslink += 0.10 * effective
            elif operation == ProcessOperation.DRY:
                hydration -= 0.36 * effective
                quality += 0.04 * effective
            elif operation == ProcessOperation.COAT:
                coating_affinity = 0.25 + 0.75 * (protein + chitin)
                porosity -= 0.18 * effective * coating_affinity
                crosslink += 0.15 * effective * coating_affinity

        excess_processing = max(0.0, process_severity - 5.5)
        quality -= 0.012 * excess_processing**2

        composition = np.clip(composition, 0.0, None)
        composition /= max(1e-6, float(composition.sum()))
        hydration = float(np.clip(hydration, 0.0, 1.0))
        porosity = float(np.clip(porosity, 0.02, 0.95))
        alignment = float(np.clip(alignment, 0.0, 1.0))
        crosslink = float(np.clip(crosslink, 0.0, 1.0))
        quality = float(np.clip(quality, 0.0, 1.0))
        properties = self._properties(
            composition, hydration, porosity, alignment, crosslink, quality
        )

        for resource, mass in required.items():
            inventory[int(resource)] -= np.float32(mass)
        batch_id = f"batch_{self.next_batch:08d}"
        self.next_batch += 1
        return MaterialBatch(
            batch_id=batch_id,
            tick=tick,
            mass=total_mass
            * max(0.72, 0.97 - 0.06 * porosity - 0.008 * process_severity),
            composition=composition,
            hydration=hydration,
            porosity=porosity,
            alignment=alignment,
            crosslink=crosslink,
            properties=properties,
            recipe=recipe,
            contributors=sorted(set(contributors)),
            causal_parents=list(causal_parents or []),
        )

    @staticmethod
    def _properties(
        composition: NDArray[np.float32],
        hydration: float,
        porosity: float,
        alignment: float,
        crosslink: float,
        quality: float,
    ) -> dict[str, float]:
        cellulose, chitin, protein, mineral, lignin, _water = map(float, composition)
        fiber_matrix_interface = float(
            np.clip(
                3.0
                * (
                    cellulose * chitin
                    + protein * mineral
                    + protein * (cellulose + chitin)
                ),
                0.0,
                1.0,
            )
        )
        stiffness = (0.34 * cellulose + 0.38 * mineral + 0.22 * chitin + 0.16 * lignin)
        stiffness += 0.08 * fiber_matrix_interface * crosslink
        stiffness *= 0.55 + 0.45 * alignment
        stiffness *= 1.0 - 0.62 * porosity
        toughness = 0.30 * cellulose + 0.32 * chitin + 0.28 * protein + 0.16 * lignin
        toughness += 0.12 * fiber_matrix_interface
        toughness *= 0.65 + 0.35 * crosslink
        permeability = porosity * (0.45 + 0.55 * hydration)
        adhesion = 0.48 * protein + 0.26 * chitin + 0.20 * crosslink + 0.12 * mineral
        healing = 0.50 * protein + 0.24 * chitin + 0.24 * hydration
        responsiveness = 0.38 * cellulose + 0.25 * chitin + 0.30 * hydration
        responsiveness *= 0.45 + 0.55 * alignment
        degradation = 0.38 * protein + 0.20 * cellulose + 0.18 * hydration - 0.26 * mineral
        raw = [
            stiffness,
            toughness,
            permeability,
            adhesion,
            healing,
            responsiveness,
            degradation,
            quality,
        ]
        return {
            name: float(np.clip(value, 0.0, 1.0))
            for name, value in zip(PROPERTIES, raw, strict=True)
        }

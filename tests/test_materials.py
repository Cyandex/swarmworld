import numpy as np
import pytest

from biofoundry.config import ScienceConfig
from biofoundry.materials import (
    DEFAULT_RECIPES,
    MaterialLab,
    RecipeValidationError,
    required_inputs,
)
from biofoundry.science import ResearchMission
from biofoundry.types import ArtifactType, Resource


def test_recipe_consumes_exact_inputs_and_bounds_properties() -> None:
    lab = MaterialLab()
    recipe = DEFAULT_RECIPES[ArtifactType.MATERIAL_SYSTEM]
    inventory = np.zeros(len(Resource), dtype=np.float32)
    required = required_inputs(recipe)
    for resource, mass in required.items():
        inventory[int(resource)] = mass
    batch = lab.execute(recipe, inventory, tick=4, contributors=["agent_000000"])
    assert inventory.sum() == pytest.approx(0.0, abs=1e-6)
    assert batch.mass > 0
    assert all(0.0 <= value <= 1.0 for value in batch.properties.values())
    assert batch.contributors == ["agent_000000"]


def test_recipe_rejects_unknown_operation() -> None:
    inventory = np.ones(len(Resource), dtype=np.float32) * 10
    recipe = {
        "inputs": [{"resource": "KELP", "mass": 1.0}],
        "steps": [{"operation": "TELEPORT", "intensity": 0.5}],
    }
    with pytest.raises(RecipeValidationError):
        MaterialLab().execute(recipe, inventory, tick=0, contributors=[])


def test_combined_recipe_synergy_uses_matched_material_ablation() -> None:
    mission = ResearchMission(ScienceConfig())
    proposal = mission.propose(
        "agent_000000",
        DEFAULT_RECIPES[ArtifactType.MATERIAL_SYSTEM],
        tick=4,
        causal_parents=["kelp_evidence", "shell_evidence"],
        combined=True,
        parent_authors=["agent_000000", "agent_000001"],
        inherited_resources=["SHELL"],
    )
    assert proposal["composition_synergy"] == pytest.approx(0.013017)
    assert proposal["composition_synergy"] != proposal["evaluation_utility"]

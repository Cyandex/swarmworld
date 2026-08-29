"""Policy protocol independent of model or environment implementation."""

from __future__ import annotations

from typing import Protocol

from ..simulation import BioFoundrySimulation
from ..types import AgentAction


class Policy(Protocol):
    def actions(self, simulation: BioFoundrySimulation) -> dict[str, AgentAction]: ...

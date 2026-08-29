"""BioFoundry World public package."""

from .config import GameConfig, load_config
from .simulation import BioFoundrySimulation

__all__ = ["BioFoundrySimulation", "GameConfig", "load_config"]
__version__ = "0.1.0"

# Increment whenever authoritative world dynamics, action semantics, or state-digest
# contents change.  Package versions may also change for renderer or documentation
# updates, so replay compatibility needs its own explicit boundary.
ENGINE_REVISION = 9

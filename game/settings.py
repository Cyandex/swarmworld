"""Game-profile loading: one YAML file with a ``game:`` section plus a standard
biofoundry configuration.

``biofoundry.config.config_from_dict`` reads only its ten known sections, so the
extra ``game:`` section is invisible to the engine, replay headers, and every
existing tool. Nothing in ``src/`` is modified.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from biofoundry.config import GameConfig, config_from_dict


@dataclass(slots=True)
class GameSettings:
    """Human-play pacing and possession settings (see docs/HUMAN_PLAY.md)."""

    # The agent slot a joining player controls. Must exist in the population.
    human_agent: str = "agent_000000"
    # Open a decision window every k ticks (1 = every tick).
    decision_interval: int = 1
    # Wait up to this many seconds for human input per window; 0 = never wait.
    input_timeout_seconds: float = 0.0
    # True = block the world until the human acts (turn-based play). Takes
    # precedence over input_timeout_seconds. Only applies while a player is joined.
    require_response: bool = False
    # Buffered human inputs applied on subsequent ticks.
    queue_limit: int = 4

    def validate(self) -> None:
        if not self.human_agent:
            raise ValueError("game.human_agent must be a non-empty agent id")
        if self.decision_interval < 1:
            raise ValueError("game.decision_interval must be positive")
        if self.input_timeout_seconds < 0:
            raise ValueError("game.input_timeout_seconds cannot be negative")
        if self.queue_limit < 1:
            raise ValueError("game.queue_limit must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "human_agent": self.human_agent,
            "decision_interval": self.decision_interval,
            "input_timeout_seconds": self.input_timeout_seconds,
            "require_response": self.require_response,
            "queue_limit": self.queue_limit,
        }


def settings_from_mapping(raw: dict[str, Any] | None) -> GameSettings:
    data = dict(raw or {})
    allowed = {field.name for field in fields(GameSettings)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown keys in game section: {sorted(unknown)}")
    settings = GameSettings(**data)
    settings.validate()
    return settings


def load_game_profile(path: str | Path) -> tuple[GameConfig, GameSettings]:
    """Load one YAML profile into (engine config, game settings)."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError("game profile must be a mapping")
    settings = settings_from_mapping(data.get("game"))
    config = config_from_dict(data)
    return config, settings

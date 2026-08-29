"""Condition-dependent executable capabilities exposed to decision policies.

The simulator remains the final authority, but model-facing prompts and schemas must
not advertise actions disabled by an experimental treatment.  This module contains
only interface facts derived from configuration; it does not prescribe scientific
strategies or privileged action sequences.
"""

from __future__ import annotations

from .config import GameConfig
from .types import ActionType

_PUBLIC_CULTURE_ACTIONS = {
    ActionType.COMMUNICATE,
    ActionType.PUBLISH,
    ActionType.CLAIM_TASK,
    ActionType.TEACH,
    ActionType.TRADE,
    ActionType.COMBINE_DESIGN,
}


def available_action_types(config: GameConfig) -> tuple[ActionType, ...]:
    """Return exactly the action verbs executable in ``config``.

    Ordering follows :class:`ActionType`, making prompts, schemas, and trace hashes
    stable across processes.
    """

    available = set(ActionType)
    if not config.science.communication:
        available.difference_update(_PUBLIC_CULTURE_ACTIONS)
    if not config.science.shared_depot:
        available.discard(ActionType.DEPOSIT)
    if not config.science.program_forking:
        available.discard(ActionType.FORK_PROGRAM)
    if not config.economy.enabled:
        available.discard(ActionType.METABOLIZE)
    return tuple(action for action in ActionType if action in available)


def addressing_available(config: GameConfig) -> bool:
    """Whether a plan may address another agent explicitly."""

    return bool(config.science.communication and config.science.addressed_communication)


def replies_available(config: GameConfig) -> bool:
    """Whether actions may point to a public request/message record."""

    return bool(config.science.communication and config.science.request_tracking)

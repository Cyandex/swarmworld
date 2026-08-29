"""Minimal provider protocol used by LLM policies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Model text plus provider-reported accounting for budget-matched studies."""

    text: str
    usage: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    async def generate(self, messages: Sequence[ChatMessage]) -> str: ...

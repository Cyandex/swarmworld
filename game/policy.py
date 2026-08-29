"""Human-aware wrapping of the stock LLM policy — composition, not modification.

The stock ``LLMPolicy`` consults ``simulation.scheduled_macro_agents()`` to pick
which agents deliberate, and pops one queued action per agent per tick. To keep
a human-possessed slot out of both paths without editing ``policies/llm.py``,
this module supplies:

- ``ScheduleFilterView``: a delegating facade over ``BioFoundrySimulation`` whose
  only override hides excluded agents from macroturn scheduling;
- ``HumanAwareLLMPolicy``: an ``LLMPolicy`` subclass that plans through the
  filtered view and skips the plan-queue pop for excluded agents, so a human
  action never consumes model calls or silently discards a model plan.

Exclusion is dynamic: agents are excluded only while a player possesses them,
so the society runs fully autonomously before a join and after a release.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any

from biofoundry.policies.llm import LLMPolicy
from biofoundry.simulation import BioFoundrySimulation
from biofoundry.types import AgentAction


class ScheduleFilterView:
    """Delegate everything to the real simulation except macroturn scheduling."""

    def __init__(self, simulation: BioFoundrySimulation, excluded: Iterable[str]):
        self._simulation = simulation
        self._excluded = set(excluded)

    def scheduled_macro_agents(self) -> list[str]:
        return [
            agent_id
            for agent_id in self._simulation.scheduled_macro_agents()
            if agent_id not in self._excluded
        ]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._simulation, name)


class HumanAwareLLMPolicy(LLMPolicy):
    def __init__(self, provider: Any, excluded_agents: Iterable[str] | None = None):
        super().__init__(provider)
        self.excluded_agents: set[str] = set(excluded_agents or ())

    @classmethod
    def adopt(cls, policy: LLMPolicy) -> HumanAwareLLMPolicy:
        """Wrap an already-constructed stock policy, reusing its provider and queues."""

        replacement = cls(policy.provider)
        replacement.action_queues = policy.action_queues
        replacement.last_traces = policy.last_traces
        replacement.calls_made = policy.calls_made
        replacement.provider_attempts = policy.provider_attempts
        replacement.provider_outage = policy.provider_outage
        return replacement

    def visible_scheduled(self, simulation: BioFoundrySimulation) -> list[str]:
        return ScheduleFilterView(simulation, self.excluded_agents).scheduled_macro_agents()

    async def refresh_plans(
        self,
        simulation: BioFoundrySimulation,
        show_progress: bool = True,
    ) -> bool:
        view = (
            ScheduleFilterView(simulation, self.excluded_agents)
            if self.excluded_agents
            else simulation
        )
        return await super().refresh_plans(view, show_progress=show_progress)

    def next_actions(self, simulation: BioFoundrySimulation) -> dict[str, AgentAction]:
        actions: dict[str, AgentAction] = {}
        for agent_id in simulation.agent_ids:
            if agent_id in self.excluded_agents:
                # The caller supplies this agent's action; its queue stays frozen.
                continue
            queue = self.action_queues.setdefault(agent_id, deque())
            actions[agent_id] = queue.popleft() if queue else AgentAction()
        return actions

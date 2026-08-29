import asyncio
from collections import deque
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from biofoundry.config import load_config
from biofoundry.memory import MemoryRecord
from biofoundry.policies.llm import LLMPolicy
from biofoundry.providers.base import ChatMessage, GenerationResult
from biofoundry.providers.openai_compatible import ProviderError
from biofoundry.simulation import BioFoundrySimulation
from biofoundry.types import ActionType, AgentAction

ROOT = Path(__file__).resolve().parents[1]


class FakeProvider:
    async def generate(self, messages: Sequence[ChatMessage]) -> str:
        assert messages[0].role == "system"
        return """{
          "research_state": {
            "goal": "understand local layered structures",
            "hypothesis": "layering redirects cracks",
            "progress": "one candidate insight",
            "next_checkpoint": "inspect a contrasting site",
            "collaboration_need": "seek an independent material observation"
          },
          "plan": [{
            "verb": "DEPOSIT_INSIGHT",
            "direction": "STAY",
            "resource": "NONE",
            "artifact": "NONE",
            "target_x": -1,
            "target_y": -1,
            "target_artifact_id": "",
            "amount": 0,
            "message": "",
            "recipe": null,
            "program": null,
            "artifact_spec": null,
            "insight": {
              "content": "layered cuticle redirects cracks",
              "kind": "observation",
              "salience": 0.8
            },
            "causal_parents": []
          }]
        }"""


def test_llm_policy_parses_and_traces_structured_action() -> None:
    simulation = BioFoundrySimulation(load_config(ROOT / "configs" / "demo.yaml"))
    policy = LLMPolicy(FakeProvider())
    actions = asyncio.run(policy.actions(simulation, show_progress=False))
    assert actions[simulation.agent_ids[0]].verb == ActionType.DEPOSIT_INSIGHT
    assert all(
        action.verb == ActionType.WAIT
        for agent_id, action in actions.items()
        if agent_id != simulation.agent_ids[0]
    )
    assert policy.last_traces[0]["response"].startswith("{")
    assert policy.last_traces[0]["research_state"]["goal"].startswith("understand")
    assert simulation.memories[0].latest("research_state") is not None
    user_prompt = policy.last_traces[0]["messages"][1]["content"]
    assert "Empirically grounded material IDs for this agent" in user_prompt
    assert "WATER=7" not in user_prompt
    assert "CATALYST=8" not in user_prompt
    assert "Unlisted materials are unobserved, not proven absent" in user_prompt
    assert "Prior self-authored research state: none yet" in user_prompt
    simulation.step(actions)
    assert simulation.deposited_insights[0]["author"] == simulation.agent_ids[0]


class OldSingleActionProvider:
    async def generate(self, messages: Sequence[ChatMessage]) -> str:
        return '{"verb": 0}'


def test_llm_policy_rejects_legacy_single_action_shape_safely() -> None:
    simulation = BioFoundrySimulation(load_config(ROOT / "configs" / "demo.yaml"))
    policy = LLMPolicy(OldSingleActionProvider())
    actions = asyncio.run(policy.actions(simulation, show_progress=False))
    assert all(action.verb == ActionType.WAIT for action in actions.values())
    assert "action plan violates schema at root" in policy.last_traces[0]["error"]


class UsageProvider(FakeProvider):
    async def generate_record(self, messages: Sequence[ChatMessage]) -> GenerationResult:
        return GenerationResult(
            text=await self.generate(messages),
            usage={"input_tokens": 321, "output_tokens": 87, "total_tokens": 408},
        )


def test_llm_policy_retains_provider_token_usage() -> None:
    simulation = BioFoundrySimulation(load_config(ROOT / "configs" / "demo.yaml"))
    policy = LLMPolicy(UsageProvider())
    asyncio.run(policy.actions(simulation, show_progress=False))
    assert policy.last_traces[0]["usage"] == {
        "input_tokens": 321,
        "output_tokens": 87,
        "total_tokens": 408,
    }
    assert policy.last_traces[0]["prompt_characters"] > 1000


class BudgetedProvider(FakeProvider):
    def __init__(self) -> None:
        self.config = SimpleNamespace(call_budget=1)
        self.calls = 0

    async def generate(self, messages: Sequence[ChatMessage]) -> str:
        self.calls += 1
        return await super().generate(messages)


def test_llm_policy_enforces_episode_call_budget_during_replanning() -> None:
    simulation = BioFoundrySimulation(load_config(ROOT / "configs" / "demo.yaml"))
    provider = BudgetedProvider()
    policy = LLMPolicy(provider)
    first = asyncio.run(policy.actions(simulation, show_progress=False))
    simulation.step(first)
    simulation.needs_replan[0] = True

    second = asyncio.run(policy.actions(simulation, show_progress=False))

    assert provider.calls == 1
    assert policy.calls_made == 1
    assert policy.last_traces == []
    assert all(action.verb == ActionType.WAIT for action in second.values())
    assert not simulation.needs_replan[0]


def test_llm_policy_records_actions_abandoned_by_evidence_replanning() -> None:
    simulation = BioFoundrySimulation(load_config(ROOT / "configs" / "demo.yaml"))
    policy = LLMPolicy(FakeProvider())
    agent_id = simulation.agent_ids[0]
    policy.action_queues[agent_id] = deque([AgentAction(verb=ActionType.MOVE)])
    simulation.needs_replan[0] = True

    asyncio.run(policy.actions(simulation, show_progress=False))

    assert policy.last_traces[0]["queued_actions_replaced"] == 1


class TransientProvider(FakeProvider):
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            call_budget=None,
            freeze_on_provider_outage=True,
            experience_attention=False,
        )
        self.failing = True

    async def generate(self, messages: Sequence[ChatMessage]) -> str:
        if self.failing:
            raise ProviderError("temporary DNS failure", retryable=True)
        return await super().generate(messages)


def test_transient_provider_failure_preserves_world_time_and_queued_actions() -> None:
    simulation = BioFoundrySimulation(load_config(ROOT / "configs" / "demo.yaml"))
    provider = TransientProvider()
    policy = LLMPolicy(provider)
    agent = simulation.agent_ids[0]
    policy.action_queues[agent] = deque([AgentAction(verb=ActionType.MOVE)])

    committed = asyncio.run(policy.refresh_plans(simulation, show_progress=False))

    assert committed is False
    assert simulation.tick == 0
    assert policy.calls_made == 0
    assert policy.provider_attempts == 1
    assert list(policy.action_queues[agent])[0].verb == ActionType.MOVE
    assert policy.last_traces[0]["error_kind"] == "provider_transient"
    assert policy.last_traces[0]["queue_preserved_during_outage"] == 1
    assert policy.last_traces[0]["planning_committed"] is False
    assert simulation.memories[0].latest("research_state") is None
    assert simulation.memories[0].retrieval_stats == {}

    provider.failing = False
    assert asyncio.run(policy.refresh_plans(simulation, show_progress=False)) is True
    assert policy.calls_made == 1
    assert policy.last_traces[0]["planning_committed"] is True


class AttentionProvider(FakeProvider):
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            call_budget=None,
            freeze_on_provider_outage=True,
            experience_attention=True,
            context_budget_characters=12_000,
            nearby_artifact_detail_limit=3,
            public_archive_retrieval_limit=2,
            memory_retrieval_limit=4,
            memory_record_characters=700,
            retrieval_feedback_weight=0.45,
            retrieval_exploration_weight=0.12,
            max_plan_actions=16,
        )


def test_experience_attention_bounds_prompt_under_large_private_history() -> None:
    simulation = BioFoundrySimulation(load_config(ROOT / "configs" / "demo.yaml"))
    for number in range(80):
        simulation.memories[0].remember(
            MemoryRecord(
                f"large_{number}",
                number,
                "observation",
                "wet interface experiment " + "x" * 8_000,
                salience=0.8,
            ),
            notebook=True,
        )
    policy = LLMPolicy(AttentionProvider())

    asyncio.run(policy.refresh_plans(simulation, show_progress=False))

    context = policy.last_traces[0]["context"]
    assert context["experience_attention"] is True
    assert context["original_prompt_characters"] > 12_000
    assert context["final_prompt_characters"] <= 12_000
    assert policy.last_traces[0]["prompt_characters"] <= 12_000


class MixedTransientProvider(TransientProvider):
    async def generate(self, messages: Sequence[ChatMessage]) -> str:
        if "Agent: agent_000000" in messages[1].content:
            raise ProviderError("one shard unavailable", retryable=True)
        return await FakeProvider.generate(self, messages)


def test_mixed_provider_failure_rolls_back_successful_peers_as_one_macroturn() -> None:
    base = load_config(ROOT / "configs" / "demo.yaml")
    simulation = BioFoundrySimulation(
        replace(base, population=replace(base.population, macro_interval=1))
    )
    policy = LLMPolicy(MixedTransientProvider())

    assert asyncio.run(policy.refresh_plans(simulation, show_progress=False)) is False

    assert policy.calls_made == 0
    assert policy.provider_attempts == simulation.population.size
    assert all(memory.latest("research_state") is None for memory in simulation.memories)
    assert all(trace["planning_committed"] is False for trace in policy.last_traces)

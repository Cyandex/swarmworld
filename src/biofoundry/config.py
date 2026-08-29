"""Configuration loading with explicit defaults and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class WorldConfig:
    width: int = 48
    height: int = 36
    field_diffusion: float = 0.06
    resource_regrowth: float = 0.002
    disturbance_interval: int = 96
    disturbance_intensity: float = 0.45
    generator: str = "fixed"
    workspace_layout: str = "centralized"
    procedural_attempts: int = 32
    # Environmental fields are normalized inventories per cell. Natural recharge
    # is an explicit external source; artifact transfers are locally conservative.
    moisture_capacity: float = 1.0
    nutrient_capacity: float = 1.0
    # Optional declarative scenario package. ``None`` preserves the original
    # BioFoundry generator, vocabulary, physics, and serialized configuration.
    scenario_package: str | None = None


@dataclass(slots=True)
class PhysicsConfig:
    """Experiment-switchable material and artifact accounting semantics."""

    closed_artifact_fluxes: bool = False
    dismantle_recovery: bool = False
    dismantle_efficiency: float = 0.55


@dataclass(slots=True)
class EconomyConfig:
    """Optional agent metabolism and generational turnover treatment."""

    enabled: bool = False
    mortality_enabled: bool = False
    respawn_enabled: bool = False
    cultural_inheritance: bool = True
    initial_energy: float = 1.0
    maximum_energy: float = 1.0
    passive_cost: float = 0.0002
    default_action_cost: float = 0.001
    move_cost: float = 0.002
    build_cost: float = 0.012
    operate_cost: float = 0.008
    communication_cost: float = 0.0015
    metabolize_efficiency: float = 0.65
    respawn_delay: int = 32
    inherited_skill_limit: int = 4


@dataclass(slots=True)
class TraceConfig:
    """Lossless trace-storage controls; neither option changes simulation state."""

    deduplicate_prompts: bool = False
    compression: str = "none"


@dataclass(slots=True)
class PopulationConfig:
    agents: int = 12
    macro_interval: int = 16
    # Offset of agent 0 in the staggered macroturn schedule. Matched isolated
    # controls use the corresponding swarm agent's phase.
    macro_phase_offset: int = 0
    communication_radius: int = 6
    inventory_capacity: float = 12.0
    memory_capacity: int = 64
    # Optional experiment-supplied spawn coordinates. Normal runs leave this unset.
    initial_positions: list[list[int]] | None = None


@dataclass(slots=True)
class SimulationConfig:
    seed: int = 17
    max_ticks: int = 2000
    snapshot_interval: int = 32
    artifact_limit: int = 4096
    # Optional episode-level cap on admitted non-WAIT actions. Failed actions still
    # consume one attempt, so conditions cannot gain physical opportunity by
    # emitting invalid plans. Admission is deterministic and rotates across agents.
    action_attempt_budget: int | None = None


@dataclass(slots=True)
class EvaluationConfig:
    """Frozen, agent-free ecological assay applied after a discovery episode."""

    enabled: bool = False
    suite: str = "native_disturbance_cycle_v1"
    horizon: int = 288


@dataclass(slots=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    ticks_per_second: float = 8.0
    render_every: int = 1


@dataclass(slots=True)
class LLMConfig:
    enabled: bool = False
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "google/gemma-4-E2B-it"
    api_key_env: str = "BIOFOUNDRY_API_KEY"
    concurrency: int = 4
    timeout_seconds: float = 120.0
    temperature: float = 0.7
    max_tokens: int = 512
    max_plan_actions: int = 16
    call_budget: int | None = None
    reasoning_effort: str | None = None
    structured_output: bool = False
    # Work around constrained decoders that cannot terminate JSON number tokens.
    # Numbers travel as short decimal strings and are coerced before validation.
    grammar_safe_numbers: bool = False
    # Some local tokenizers strongly prefer legal JSON whitespace forever under a
    # grammar.  A negative bias on tokenizer tokens that decode only to whitespace
    # preserves compact JSON while leaving content tokens unconstrained.
    json_whitespace_logit_bias: float | None = None
    tokenizer: str | None = None
    tokenizer_local_files_only: bool = True
    # General, agent-directed context selection. The agent's own research state is
    # the retrieval query; fixed budgets bound scaling without prescribing content.
    experience_attention: bool = False
    context_budget_characters: int = 64_000
    nearby_artifact_detail_limit: int = 6
    public_archive_retrieval_limit: int = 4
    memory_retrieval_limit: int = 4
    memory_record_characters: int = 2_000
    retrieval_feedback_weight: float = 0.45
    retrieval_exploration_weight: float = 0.12
    # A total provider failure is not an environmental event. When enabled, the
    # runner retries the same macroturn without consuming queued actions or ticks.
    freeze_on_provider_outage: bool = False
    provider_retry_seconds: float = 5.0
    # Legacy JSON mode is retained so older replay headers remain loadable.
    json_mode: bool = False


@dataclass(slots=True)
class ScienceConfig:
    enabled: bool = False
    mission: str = "open_habitat_resilience"
    test_scale: float = 0.25
    target_material_utility: float = 0.30
    target_artifact_performance: float = 0.15
    target_behavioral_novelty: float = 0.08
    minimum_contributors: int = 2
    feedback_history: int = 8
    action_feedback: bool = True
    immediate_replanning: bool = True
    communication: bool = True
    social_awareness: bool = True
    shared_depot: bool = True
    # Fixed laboratories are public infrastructure; biological matter remains local.
    public_infrastructure_map: bool = True
    global_landmarks: bool = False
    addressed_communication: bool = False
    program_forking: bool = False
    skill_library: bool = False
    retrieval_diagnostics: bool = False
    experience_reuse: bool = False
    request_tracking: bool = False
    # Preserve legacy eager interruption by default. Scalable profiles can batch
    # messages into the structured inbox until the next scheduled macroturn.
    message_interrupts: bool = True
    # Legacy profiles also replan after intermediate fabrication/publication events.
    # Scalable profiles wait for measured evidence or a completed embodiment.
    selective_replanning: bool = False
    # Confirmatory studies use fixed opportunities; interactive runs may replan on
    # events. This changes scheduling, never the content of an agent's decisions.
    decision_schedule: str = "event-driven"


@dataclass(slots=True)
class GameConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    science: ScienceConfig = field(default_factory=ScienceConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    economy: EconomyConfig = field(default_factory=EconomyConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)

    def validate(self) -> None:
        if self.world.width < 16 or self.world.height < 16:
            raise ValueError("world dimensions must both be at least 16")
        if self.population.agents < 1:
            raise ValueError("population.agents must be positive")
        if self.population.macro_interval < 1:
            raise ValueError("population.macro_interval must be positive")
        if self.population.macro_phase_offset < 0:
            raise ValueError("population.macro_phase_offset cannot be negative")
        if self.population.communication_radius < 0:
            raise ValueError("communication radius cannot be negative")
        if self.population.initial_positions is not None:
            positions = self.population.initial_positions
            if len(positions) != self.population.agents:
                raise ValueError(
                    "population.initial_positions must contain exactly one position "
                    "per agent"
                )
            for position in positions:
                if len(position) != 2:
                    raise ValueError(
                        "each population.initial_positions entry must be [x, y]"
                    )
                x, y = position
                if not isinstance(x, int) or not isinstance(y, int):
                    raise ValueError("initial-position coordinates must be integers")
                if not (0 <= x < self.world.width and 0 <= y < self.world.height):
                    raise ValueError("initial positions must lie inside the world")
        if not 0.0 <= self.world.field_diffusion <= 0.24:
            raise ValueError("field_diffusion must be in [0, 0.24] for stability")
        if self.world.disturbance_interval < 0:
            raise ValueError("world.disturbance_interval cannot be negative")
        if not 0.0 <= self.world.disturbance_intensity <= 1.0:
            raise ValueError("world.disturbance_intensity must be in [0, 1]")
        if self.world.generator not in {"fixed", "procedural"}:
            raise ValueError("world.generator must be 'fixed' or 'procedural'")
        if self.world.workspace_layout not in {"centralized", "distributed"}:
            raise ValueError(
                "world.workspace_layout must be 'centralized' or 'distributed'"
            )
        if self.world.procedural_attempts < 1:
            raise ValueError("world.procedural_attempts must be positive")
        if self.world.moisture_capacity <= 0 or self.world.nutrient_capacity <= 0:
            raise ValueError("world field capacities must be positive")
        if self.simulation.max_ticks < 1:
            raise ValueError("simulation.max_ticks must be positive")
        if (
            self.simulation.action_attempt_budget is not None
            and self.simulation.action_attempt_budget < 1
        ):
            raise ValueError(
                "simulation.action_attempt_budget must be positive when specified"
            )
        if self.evaluation.suite != "native_disturbance_cycle_v1":
            raise ValueError("unsupported evaluation.suite")
        if self.evaluation.horizon < 2:
            raise ValueError("evaluation.horizon must be at least 2")
        if self.server.ticks_per_second <= 0:
            raise ValueError("server.ticks_per_second must be positive")
        if self.llm.reasoning_effort not in {
            None,
            "none",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ValueError("unsupported llm.reasoning_effort")
        if not 0.0 <= self.llm.temperature <= 2.0:
            raise ValueError("llm.temperature must be in [0, 2]")
        if not 1 <= self.llm.max_plan_actions <= 16:
            raise ValueError("llm.max_plan_actions must be in [1, 16]")
        if self.llm.structured_output and self.llm.json_mode:
            raise ValueError("llm.structured_output and llm.json_mode are mutually exclusive")
        if self.llm.json_whitespace_logit_bias is not None:
            if not self.llm.structured_output:
                raise ValueError(
                    "llm.json_whitespace_logit_bias requires llm.structured_output"
                )
            if not self.llm.tokenizer:
                raise ValueError("llm.tokenizer is required for JSON whitespace bias")
            if not -100.0 <= self.llm.json_whitespace_logit_bias <= 100.0:
                raise ValueError("llm.json_whitespace_logit_bias must be in [-100, 100]")
        if self.llm.call_budget is not None and self.llm.call_budget < 1:
            raise ValueError("llm.call_budget must be positive when specified")
        if self.llm.context_budget_characters < 12_000:
            raise ValueError("llm.context_budget_characters must be at least 12000")
        for name in (
            "nearby_artifact_detail_limit",
            "public_archive_retrieval_limit",
            "memory_retrieval_limit",
            "memory_record_characters",
        ):
            if getattr(self.llm, name) < 1:
                raise ValueError(f"llm.{name} must be positive")
        for name in ("retrieval_feedback_weight", "retrieval_exploration_weight"):
            if getattr(self.llm, name) < 0:
                raise ValueError(f"llm.{name} cannot be negative")
        if self.llm.provider_retry_seconds <= 0:
            raise ValueError("llm.provider_retry_seconds must be positive")
        if not 0.0 < self.science.test_scale <= 1.0:
            raise ValueError("science.test_scale must be in (0, 1]")
        if self.science.minimum_contributors < 1:
            raise ValueError("science.minimum_contributors must be positive")
        if self.science.feedback_history < 1:
            raise ValueError("science.feedback_history must be positive")
        if self.science.decision_schedule not in {"event-driven", "fixed"}:
            raise ValueError(
                "science.decision_schedule must be 'event-driven' or 'fixed'"
            )
        for name, value in (
            ("target_material_utility", self.science.target_material_utility),
            ("target_artifact_performance", self.science.target_artifact_performance),
            ("target_behavioral_novelty", self.science.target_behavioral_novelty),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"science.{name} must be in [0, 1]")
        if not 0.0 <= self.physics.dismantle_efficiency <= 1.0:
            raise ValueError("physics.dismantle_efficiency must be in [0, 1]")
        if not 0.0 < self.economy.maximum_energy:
            raise ValueError("economy.maximum_energy must be positive")
        if not 0.0 <= self.economy.initial_energy <= self.economy.maximum_energy:
            raise ValueError("economy.initial_energy must be within maximum_energy")
        for name in (
            "passive_cost",
            "default_action_cost",
            "move_cost",
            "build_cost",
            "operate_cost",
            "communication_cost",
        ):
            if getattr(self.economy, name) < 0:
                raise ValueError(f"economy.{name} cannot be negative")
        if not 0.0 < self.economy.metabolize_efficiency <= 1.0:
            raise ValueError("economy.metabolize_efficiency must be in (0, 1]")
        if self.economy.respawn_delay < 1:
            raise ValueError("economy.respawn_delay must be positive")
        if self.economy.inherited_skill_limit < 0:
            raise ValueError("economy.inherited_skill_limit cannot be negative")
        if self.economy.respawn_enabled and not self.economy.mortality_enabled:
            raise ValueError("economy.respawn_enabled requires mortality_enabled")
        if self.economy.mortality_enabled and not self.economy.enabled:
            raise ValueError("economy.mortality_enabled requires economy.enabled")
        if self.trace.compression not in {"none", "gzip"}:
            raise ValueError("trace.compression must be 'none' or 'gzip'")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        # Do not alter legacy replay headers, hashes, or user-facing configuration
        # dumps merely because scenario support is installed.
        if value["world"].get("scenario_package") is None:
            value["world"].pop("scenario_package", None)
        return value


def _section(cls: type[Any], data: dict[str, Any], name: str) -> Any:
    raw = data.get(name, {}) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"configuration section {name!r} must be a mapping")
    allowed = set(cls.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown keys in {name}: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path) -> GameConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return config_from_dict(data)


def config_from_dict(data: dict[str, Any]) -> GameConfig:
    """Reconstruct a validated configuration from a replay header or mapping."""
    if not isinstance(data, dict):
        raise TypeError("configuration must be a mapping")
    config = GameConfig(
        world=_section(WorldConfig, data, "world"),
        population=_section(PopulationConfig, data, "population"),
        simulation=_section(SimulationConfig, data, "simulation"),
        evaluation=_section(EvaluationConfig, data, "evaluation"),
        server=_section(ServerConfig, data, "server"),
        llm=_section(LLMConfig, data, "llm"),
        science=_section(ScienceConfig, data, "science"),
        physics=_section(PhysicsConfig, data, "physics"),
        economy=_section(EconomyConfig, data, "economy"),
        trace=_section(TraceConfig, data, "trace"),
    )
    config.validate()
    return config

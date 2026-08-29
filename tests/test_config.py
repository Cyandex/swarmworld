from pathlib import Path

import pytest

from biofoundry.config import config_from_dict, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_demo_config_loads() -> None:
    config = load_config(ROOT / "configs" / "demo.yaml")
    assert config.population.agents == 12
    assert config.world.width == 48
    assert config.llm.enabled is False
    assert config.science.enabled is True
    assert config.science.minimum_contributors == 2


def test_config_round_trips_through_replay_mapping() -> None:
    config = load_config(ROOT / "configs" / "demo.yaml")
    rebuilt = config_from_dict(config.as_dict())
    assert rebuilt == config


def test_openai_luna_config_enables_real_llm_agents() -> None:
    config = load_config(ROOT / "configs" / "openai-gpt-5.6-luna.yaml")
    assert config.llm.enabled is True
    assert config.llm.base_url == "https://api.openai.com/v1"
    assert config.llm.model == "gpt-5.6-luna"
    assert config.llm.api_key_env == "OPENAI_API_KEY"
    assert config.llm.reasoning_effort == "low"
    assert config.llm.structured_output is True
    assert config.llm.json_mode is False
    assert config.llm.max_tokens == 4096
    assert config.llm.max_plan_actions == 16


def test_local_gemma_config_uses_responses_and_safe_numeric_transport() -> None:
    config = load_config(ROOT / "configs" / "local-gemma-4-mistralrs.yaml")
    assert config.llm.enabled is True
    assert config.llm.base_url == "http://127.0.0.1:8000/v1"
    assert config.llm.model == "default"
    assert config.llm.structured_output is True
    assert config.llm.grammar_safe_numbers is True
    assert config.llm.json_whitespace_logit_bias == -100.0
    assert config.llm.tokenizer == "google/gemma-4-E2B-it"
    assert config.llm.tokenizer_local_files_only is True


def test_local_gemma_e4b_config_selects_matching_tokenizer() -> None:
    config = load_config(ROOT / "configs" / "local-gemma-4-e4b-mistralrs.yaml")
    assert config.llm.model == "default"
    assert config.llm.tokenizer == "google/gemma-4-E4B-it"
    assert config.llm.structured_output is True


def test_local_gemma_12b_config_selects_matching_tokenizer() -> None:
    config = load_config(ROOT / "configs" / "local-gemma-4-12b-mistralrs.yaml")
    assert config.llm.model == "default"
    assert config.llm.tokenizer == "google/gemma-4-12B-it"
    assert config.llm.structured_output is True
    assert config.llm.max_plan_actions == 6


def test_gemma_e4b_technology_ecology_profiles_match_scientific_design() -> None:
    hosted = load_config(
        ROOT / "configs" / "openai-gpt-5.6-luna-technology-ecology-50.yaml"
    )
    mistralrs = load_config(
        ROOT / "configs" / "mistralrs-gemma-4-e4b-technology-ecology-50.yaml"
    )
    vllm = load_config(
        ROOT / "configs" / "vllm-gemma-4-e4b-technology-ecology-50.yaml"
    )

    assert mistralrs.population.agents == vllm.population.agents == 50
    assert mistralrs.world == vllm.world
    assert mistralrs.population == vllm.population
    assert mistralrs.simulation == vllm.simulation
    assert mistralrs.evaluation == vllm.evaluation
    assert mistralrs.science == vllm.science
    assert mistralrs.physics == vllm.physics
    assert mistralrs.economy == vllm.economy
    assert mistralrs.trace == vllm.trace

    assert mistralrs.llm.base_url == "http://127.0.0.1:8000/v1"
    assert mistralrs.llm.model == "default"
    assert mistralrs.llm.structured_output is True
    assert mistralrs.llm.grammar_safe_numbers is True
    assert mistralrs.llm.tokenizer == "google/gemma-4-E4B-it"
    assert mistralrs.llm.tokenizer_local_files_only is False
    assert mistralrs.llm.max_plan_actions == hosted.llm.max_plan_actions == 12

    assert vllm.llm.base_url == "http://127.0.0.1:8000/v1"
    assert vllm.llm.model == "swarm-gemma-4-e4b"
    assert vllm.llm.structured_output is True
    assert vllm.llm.grammar_safe_numbers is False
    assert vllm.llm.tokenizer is None
    assert vllm.llm.max_plan_actions == hosted.llm.max_plan_actions


def test_sampling_temperature_is_bounded() -> None:
    with pytest.raises(ValueError, match="temperature"):
        config_from_dict({"llm": {"temperature": 2.1}})
    with pytest.raises(ValueError, match="max_plan_actions"):
        config_from_dict({"llm": {"max_plan_actions": 17}})


def test_strict_and_legacy_json_modes_cannot_both_be_enabled() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        config_from_dict({"llm": {"structured_output": True, "json_mode": True}})


def test_json_whitespace_bias_requires_strict_output_and_tokenizer() -> None:
    with pytest.raises(ValueError, match="requires llm.structured_output"):
        config_from_dict({"llm": {"json_whitespace_logit_bias": -100.0}})
    with pytest.raises(ValueError, match="tokenizer is required"):
        config_from_dict(
            {
                "llm": {
                    "structured_output": True,
                    "json_whitespace_logit_bias": -100.0,
                }
            }
        )

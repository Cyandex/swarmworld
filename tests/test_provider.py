import http.client
import io
import urllib.error

import pytest

import biofoundry.providers.openai_compatible as provider_module
from biofoundry.config import LLMConfig
from biofoundry.providers.base import ChatMessage
from biofoundry.providers.openai_compatible import (
    OpenAICompatibleProvider,
    ProviderError,
    _extract_output_text,
)
from biofoundry.structured_output import ACTION_PLAN_SCHEMA


def test_gpt_5_6_uses_responses_request_shape() -> None:
    provider = OpenAICompatibleProvider(
        LLMConfig(
            model="gpt-5.6-luna",
            max_tokens=4096,
            reasoning_effort="low",
            structured_output=True,
        )
    )
    body = provider._request_body([ChatMessage("user", "Return JSON.")])
    assert provider.endpoint == "http://127.0.0.1:8000/v1/responses"
    assert body["model"] == "gpt-5.6-luna"
    assert body["input"] == [{"role": "user", "content": "Return JSON."}]
    assert body["max_output_tokens"] == 4096
    assert body["store"] is False
    assert "messages" not in body
    assert "max_completion_tokens" not in body
    assert "max_tokens" not in body
    assert body["reasoning"] == {"effort": "low"}
    assert "reasoning_effort" not in body
    assert "temperature" not in body
    assert body["text"] == {
        "format": {
            "type": "json_schema",
            "name": "biofoundry_agent_plan",
            "strict": True,
            "schema": ACTION_PLAN_SCHEMA,
        }
    }
    assert "response_format" not in body


def test_legacy_json_mode_remains_available_for_old_replay_configs() -> None:
    provider = OpenAICompatibleProvider(LLMConfig(json_mode=True))
    body = provider._request_body([ChatMessage("user", "Return JSON.")])
    assert body["text"] == {"format": {"type": "json_object"}}


def test_local_compatible_models_use_same_responses_shape() -> None:
    provider = OpenAICompatibleProvider(LLMConfig(model="local-gemma", max_tokens=512))
    messages = [
        ChatMessage("system", "Act locally."),
        ChatMessage("user", "Choose an action."),
    ]
    body = provider._request_body(messages)
    assert body["input"] == [
        {"role": "system", "content": "Act locally."},
        {"role": "user", "content": "Choose an action."},
    ]
    assert body["max_output_tokens"] == 512
    assert body["temperature"] == 0.7
    assert "text" not in body


def test_grammar_safe_numbers_are_transport_only() -> None:
    provider = OpenAICompatibleProvider(
        LLMConfig(structured_output=True, grammar_safe_numbers=True)
    )
    body = provider._request_body([ChatMessage("user", "Return JSON.")])
    schema = body["text"]["format"]["schema"]
    action = schema["$defs"]["action"]["anyOf"][0]["properties"]
    assert action["amount"]["type"] == "string"
    assert action["target_x"]["type"] == "string"
    canonical = ACTION_PLAN_SCHEMA["$defs"]["action"]["anyOf"][0]["properties"]
    assert canonical["amount"]["type"] == "number"


def test_optional_whitespace_bias_is_sent_only_for_strict_local_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provider_module, "_whitespace_token_ids", lambda *_: (107, 138))
    provider = OpenAICompatibleProvider(
        LLMConfig(
            structured_output=True,
            tokenizer="local-tokenizer",
            json_whitespace_logit_bias=-100.0,
        )
    )
    body = provider._request_body([ChatMessage("user", "Return JSON.")])
    assert body["logit_bias"] == {"107": -100.0, "138": -100.0}


def test_extracts_nested_responses_output_text() -> None:
    data = {
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": []},
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": '{"verb":3}'},
                ],
            },
        ],
    }
    assert _extract_output_text(data) == '{"verb":3}'


def test_accepts_compatible_server_output_text_convenience_field() -> None:
    assert _extract_output_text(
        {"status": "completed", "output_text": '{"verb":0}'}
    ) == '{"verb":0}'


def test_connection_reset_is_classified_as_retryable_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAICompatibleProvider(LLMConfig())

    def disconnected(*_args: object, **_kwargs: object) -> None:
        raise ConnectionResetError("remote disconnected")

    monkeypatch.setattr(provider_module.urllib.request, "urlopen", disconnected)
    with pytest.raises(ProviderError) as raised:
        provider._generate_sync_record([ChatMessage("user", "Choose.")])
    assert raised.value.retryable is True


def test_truncated_response_is_classified_as_retryable_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAICompatibleProvider(LLMConfig())

    class TruncatedResponse:
        def __enter__(self) -> "TruncatedResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            raise http.client.IncompleteRead(b'{"status":', 64)

    monkeypatch.setattr(
        provider_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: TruncatedResponse(),
    )
    with pytest.raises(ProviderError) as raised:
        provider._generate_sync_record([ChatMessage("user", "Choose.")])
    assert raised.value.retryable is True
    assert "IncompleteRead" in str(raised.value)


@pytest.mark.parametrize(
    ("error_code", "retryable"),
    [
        ("rate_limit_exceeded", True),
        ("credit_balance_exhausted", False),
        ("insufficient_quota", False),
        ("billing_hard_limit_reached", False),
    ],
)
def test_429_retry_semantics_distinguish_throttling_from_quota_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
    error_code: str,
    retryable: bool,
) -> None:
    provider = OpenAICompatibleProvider(LLMConfig())
    detail = (
        '{"error":{"message":"request rejected","code":"'
        + error_code
        + '"}}'
    ).encode()

    def rejected(request: object, timeout: float) -> None:
        raise urllib.error.HTTPError(
            url="http://127.0.0.1:8000/v1/responses",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=io.BytesIO(detail),
        )

    monkeypatch.setattr(provider_module.urllib.request, "urlopen", rejected)
    with pytest.raises(ProviderError) as raised:
        provider._generate_sync_record([ChatMessage("user", "Choose.")])
    assert raised.value.retryable is retryable


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
            "incomplete",
        ),
        (
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "refusal", "refusal": "not allowed"}],
                    }
                ],
            },
            "refused",
        ),
        ({"error": {"message": "backend unavailable"}}, "backend unavailable"),
    ],
)
def test_rejects_non_text_responses(data: dict[str, object], message: str) -> None:
    with pytest.raises(ProviderError, match=message):
        _extract_output_text(data)

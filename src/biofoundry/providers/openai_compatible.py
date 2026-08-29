"""Responses API boundary for OpenAI, vLLM, mistral.rs, and compatible servers."""

from __future__ import annotations

import asyncio
import http.client
import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import Any

from ..config import LLMConfig
from ..structured_output import bounded_action_plan_text_format
from ..types import ActionType
from .base import ChatMessage, GenerationResult


class ProviderError(RuntimeError):
    """Provider-boundary failure with explicit retry semantics.

    Only transient transport/service failures may pause world time. Invalid requests,
    refusals, and malformed completed responses remain ordinary failed decisions so a
    bad configuration cannot deadlock an episode forever.
    """

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


_TERMINAL_429_CODES = {
    "billing_hard_limit_reached",
    "credit_balance_exhausted",
    "insufficient_quota",
}


def _http_error_is_retryable(status: int, detail: str) -> bool:
    """Distinguish transient throttling from terminal account failures."""

    if status != 429:
        return status in {408, 409, 425} or status >= 500
    try:
        payload = json.loads(detail)
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        code = error.get("code") if isinstance(error, dict) else None
    except (json.JSONDecodeError, TypeError):
        code = None
    return str(code or "").lower() not in _TERMINAL_429_CODES


@lru_cache(maxsize=8)
def _whitespace_token_ids(
    tokenizer_name: str, local_files_only: bool
) -> tuple[int, ...]:
    """Resolve tokenizer-specific whitespace tokens for compact strict JSON.

    This is deliberately optional: remote APIs and local engines without the
    whitespace-loop pathology never import Transformers.
    """

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on optional local extra
        raise ProviderError(
            "llm.json_whitespace_logit_bias requires the optional 'transformers' "
            "package; install biofoundry-world[local]"
        ) from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            local_files_only=local_files_only,
        )
    except Exception as exc:  # pragma: no cover - external tokenizer/cache failure
        raise ProviderError(
            f"could not load tokenizer {tokenizer_name!r} for JSON whitespace bias: {exc}"
        ) from exc
    token_ids: list[int] = []
    for token_id in range(len(tokenizer)):
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded and decoded.isspace():
            token_ids.append(token_id)
    if not token_ids:
        raise ProviderError(
            f"tokenizer {tokenizer_name!r} exposed no whitespace-only tokens"
        )
    return tuple(token_ids)


def _extract_output_text(data: dict[str, Any]) -> str:
    """Extract assistant text from a non-streaming Responses API payload."""
    error = data.get("error")
    if error:
        if isinstance(error, dict):
            detail = str(error.get("message") or error)
        else:
            detail = str(error)
        raise ProviderError(f"model response failed: {detail[:1200]}")

    status = data.get("status")
    if status not in (None, "completed"):
        details = data.get("incomplete_details")
        suffix = f": {details}" if details else ""
        raise ProviderError(f"model response status is {status!r}{suffix}")

    # Some compatible servers expose the SDK convenience field directly.
    direct = data.get("output_text")
    if isinstance(direct, str) and direct:
        return direct

    texts: list[str] = []
    refusals: list[str] = []
    output = data.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    texts.append(part["text"])
                elif part.get("type") == "refusal":
                    refusals.append(str(part.get("refusal", "request refused")))

    if texts:
        return "".join(texts)
    if refusals:
        raise ProviderError(f"model refused the request: {'; '.join(refusals)[:1200]}")
    raise ProviderError(f"Responses payload contains no output text: {data}")


class OpenAICompatibleProvider:
    def __init__(
        self,
        config: LLMConfig,
        *,
        allowed_action_types: Iterable[ActionType | str] | None = None,
        allow_addressing: bool = True,
        allow_replies: bool = True,
        resource_names: Iterable[str] | None = None,
        operation_names: Iterable[str] | None = None,
    ):
        self.config = config
        self.allowed_action_types = (
            None if allowed_action_types is None else tuple(allowed_action_types)
        )
        self.allow_addressing = allow_addressing
        self.allow_replies = allow_replies
        self.resource_names = None if resource_names is None else tuple(resource_names)
        self.operation_names = None if operation_names is None else tuple(operation_names)
        self.endpoint = config.base_url.rstrip("/") + "/responses"
        self.semaphore = asyncio.Semaphore(config.concurrency)
        self._json_whitespace_logit_bias: dict[str, float] = {}
        if config.json_whitespace_logit_bias is not None:
            assert config.tokenizer is not None
            token_ids = _whitespace_token_ids(
                config.tokenizer,
                config.tokenizer_local_files_only,
            )
            self._json_whitespace_logit_bias = {
                str(token_id): float(config.json_whitespace_logit_bias)
                for token_id in token_ids
            }

    async def generate(self, messages: Sequence[ChatMessage]) -> str:
        async with self.semaphore:
            return await asyncio.to_thread(self._generate_sync, messages)

    async def generate_record(
        self, messages: Sequence[ChatMessage]
    ) -> GenerationResult:
        """Generate text while retaining Responses API token accounting."""
        async with self.semaphore:
            return await asyncio.to_thread(self._generate_sync_record, messages)

    def _request_body(self, messages: Sequence[ChatMessage]) -> dict[str, Any]:
        body = {
            "model": self.config.model,
            "input": [
                {"role": message.role, "content": message.content} for message in messages
            ],
            "max_output_tokens": self.config.max_tokens,
            # Swarm decisions are stateless at the provider layer. Memory is explicit in
            # each prompt and recorded by the simulator, so server-side storage is not used.
            "store": False,
        }
        if self.config.reasoning_effort is not None:
            body["reasoning"] = {"effort": self.config.reasoning_effort}
        else:
            body["temperature"] = self.config.temperature
        if self.config.structured_output:
            output_format = bounded_action_plan_text_format(
                self.config.max_plan_actions,
                grammar_safe_numbers=self.config.grammar_safe_numbers,
                allowed_action_types=self.allowed_action_types,
                allow_addressing=self.allow_addressing,
                allow_replies=self.allow_replies,
                resource_names=self.resource_names,
                operation_names=self.operation_names,
            )
            body["text"] = {"format": output_format}
            if self._json_whitespace_logit_bias:
                body["logit_bias"] = dict(self._json_whitespace_logit_bias)
        elif self.config.json_mode:
            body["text"] = {"format": {"type": "json_object"}}
        return body

    def _generate_sync(self, messages: Sequence[ChatMessage]) -> str:
        return self._generate_sync_record(messages).text

    def _generate_sync_record(
        self, messages: Sequence[ChatMessage]
    ) -> GenerationResult:
        body = self._request_body(messages)
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.config.api_key_env, "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:1200]
            except Exception:
                detail = ""
            suffix = f": {detail}" if detail else ""
            retryable = _http_error_is_retryable(exc.code, detail)
            raise ProviderError(
                f"model request failed: HTTP {exc.code}{suffix}",
                retryable=retryable,
            ) from exc
        except (
            urllib.error.URLError,
            ConnectionError,
            TimeoutError,
            http.client.IncompleteRead,
            json.JSONDecodeError,
        ) as exc:
            raise ProviderError(
                f"model request failed: {exc}", retryable=True
            ) from exc
        usage = data.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        return GenerationResult(
            text=_extract_output_text(data),
            usage={str(key): value for key, value in usage.items()},
            metadata={
                "response_id": str(data.get("id", "")),
                "model": str(data.get("model", self.config.model)),
                "status": str(data.get("status", "completed")),
            },
        )

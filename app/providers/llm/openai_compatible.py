"""OpenAI-compatible chat-completions language-model provider."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Literal

import httpx
from pydantic import SecretStr

from app.core.exceptions import ProviderError, RateLimitError
from app.providers.base import GenerationResponse, LLMProvider, TokenUsage
from app.providers.http_client import validate_model_catalog

_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenAICompatibleLLMProvider(LLMProvider):
    """Generate and stream through the portable Chat Completions contract."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: SecretStr | None,
        model: str,
        timeout_seconds: float,
        max_retries: int,
        max_concurrency: int,
        client: httpx.AsyncClient | None = None,
        request_headers: Mapping[str, str] | None = None,
        healthcheck_mode: Literal["catalog", "inference"] = "catalog",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._max_retries = max_retries
        self._healthcheck_mode = healthcheck_mode
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._owns_client = client is None
        headers = {"Accept": "application/json"}
        if api_key is not None and api_key.get_secret_value():
            headers["Authorization"] = f"Bearer {api_key.get_secret_value()}"
        if request_headers is not None:
            headers.update(request_headers)
        self._request_headers = headers
        self._client = client or httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds),
        )

    @property
    def name(self) -> str:
        return "openai_compatible"

    @property
    def model(self) -> str:
        """Return the configured upstream model identifier."""
        return self._model

    async def healthcheck(self) -> None:
        """Verify configured-model availability with a safe selectable probe."""
        if self._healthcheck_mode == "inference":
            await self.generate(
                messages=[{"role": "user", "content": "healthcheck"}],
                max_tokens=1,
            )
            return
        decoded = await self._request_json("GET", self._models_endpoint())
        validate_model_catalog(decoded, self._model)

    async def generate(
        self,
        *,
        messages: Sequence[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> GenerationResponse:
        """Return one validated Chat Completions response with token usage."""
        _validate_generation_input(messages, max_tokens)
        payload = self._payload(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        if response_format is not None:
            payload["response_format"] = response_format
        decoded = await self._request_json("POST", self._completions_endpoint(), payload)
        text, finish_reason = _parse_completion(decoded)
        return GenerationResponse(
            text=text,
            model=_response_model(decoded, self._model),
            usage=_parse_usage(decoded.get("usage")),
            finish_reason=finish_reason,
            raw=dict(decoded),
        )

    def stream(
        self,
        *,
        messages: Sequence[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Yield validated text deltas from an SSE Chat Completions response."""
        _validate_generation_input(messages, max_tokens)
        payload = self._payload(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        return self._stream(payload)

    async def _stream(self, payload: Mapping[str, Any]) -> AsyncIterator[str]:
        emitted = False
        async with self._semaphore:
            for attempt in range(self._max_retries + 1):
                try:
                    async with self._client.stream(
                        "POST",
                        self._completions_endpoint(),
                        json=dict(payload),
                        headers={**self._request_headers, "Accept": "text/event-stream"},
                    ) as response:
                        if response.status_code in _RETRYABLE_STATUS_CODES:
                            if attempt < self._max_retries:
                                await asyncio.sleep(_retry_delay(response, attempt))
                                continue
                            _raise_retry_exhausted(response.status_code)
                        try:
                            response.raise_for_status()
                        except httpx.HTTPStatusError as exc:
                            raise ProviderError(
                                "Language-model provider rejected the streaming request",
                                details={"status_code": exc.response.status_code},
                            ) from exc
                        async for line in response.aiter_lines():
                            event = _sse_data(line)
                            if event is None:
                                continue
                            if event == "[DONE]":
                                return
                            delta = _parse_stream_delta(event)
                            if delta:
                                emitted = True
                                yield delta
                        return
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    if emitted or attempt >= self._max_retries:
                        raise ProviderError(
                            "Language-model stream failed after bounded retries"
                        ) from exc
                    await asyncio.sleep(0.25 * (2**attempt))

    async def _request_json(
        self,
        method: str,
        endpoint: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        async with self._semaphore:
            for attempt in range(self._max_retries + 1):
                try:
                    response = await self._client.request(
                        method,
                        endpoint,
                        json=dict(payload) if payload is not None else None,
                        headers=self._request_headers,
                    )
                    if response.status_code in _RETRYABLE_STATUS_CODES:
                        if attempt < self._max_retries:
                            await asyncio.sleep(_retry_delay(response, attempt))
                            continue
                        _raise_retry_exhausted(response.status_code)
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        raise ProviderError(
                            "Language-model provider rejected the request",
                            details={"status_code": exc.response.status_code},
                        ) from exc
                    try:
                        decoded = response.json()
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        raise ProviderError(
                            "Language-model provider returned invalid JSON"
                        ) from exc
                    if not isinstance(decoded, Mapping):
                        raise ProviderError(
                            "Language-model provider returned a non-object JSON response"
                        )
                    return decoded
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    if attempt >= self._max_retries:
                        raise ProviderError(
                            "Language-model request failed after bounded retries"
                        ) from exc
                    await asyncio.sleep(0.25 * (2**attempt))
        raise ProviderError("Language-model request failed")

    def _payload(
        self,
        *,
        messages: Sequence[dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [dict(message) for message in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _completions_endpoint(self) -> str:
        if self._base_url.endswith("/chat/completions"):
            return self._base_url
        return f"{self._base_url}/chat/completions"

    def _models_endpoint(self) -> str:
        if self._base_url.endswith("/chat/completions"):
            root = self._base_url[: -len("/chat/completions")]
            return f"{root}/models"
        return f"{self._base_url}/models"

    async def close(self) -> None:
        """Release the HTTP connection pool when this provider created it."""
        if self._owns_client:
            await self._client.aclose()


def _validate_generation_input(messages: Sequence[dict[str, str]], max_tokens: int) -> None:
    if not messages:
        raise ProviderError("Language-model messages must not be empty")
    if max_tokens < 1:
        raise ProviderError("Language-model max_tokens must be positive")
    if any(not message.get("role") or not message.get("content") for message in messages):
        raise ProviderError("Language-model messages require non-empty role and content")


def _parse_completion(decoded: Mapping[str, Any]) -> tuple[str, str | None]:
    choices = decoded.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ProviderError("Language-model provider returned no completion choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise ProviderError("Language-model provider returned an invalid completion message")
    text = _content_text(message.get("content"))
    if text is None:
        raise ProviderError("Language-model provider returned non-text completion content")
    finish_reason = choice.get("finish_reason")
    return text, finish_reason if isinstance(finish_reason, str) else None


def _parse_stream_delta(raw_event: str) -> str:
    try:
        decoded = json.loads(raw_event)
    except json.JSONDecodeError as exc:
        raise ProviderError("Language-model provider returned invalid SSE JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ProviderError("Language-model provider returned a non-object SSE event")
    choices = decoded.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ProviderError("Language-model provider returned an invalid SSE choice")
    delta = choice.get("delta")
    if not isinstance(delta, Mapping):
        return ""
    return _content_text(delta.get("content")) or ""


def _content_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for block in content:
        if not isinstance(block, Mapping):
            return None
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts) if parts else None


def _parse_usage(raw_usage: Any) -> TokenUsage:
    if not isinstance(raw_usage, Mapping):
        return TokenUsage()
    prompt_tokens = _nonnegative_int(raw_usage.get("prompt_tokens", raw_usage.get("input_tokens")))
    completion_tokens = _nonnegative_int(
        raw_usage.get("completion_tokens", raw_usage.get("output_tokens"))
    )
    if prompt_tokens == 0 and completion_tokens == 0:
        prompt_tokens = _nonnegative_int(raw_usage.get("total_tokens"))
    return TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


def _nonnegative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _response_model(decoded: Mapping[str, Any], fallback: str) -> str:
    model = decoded.get("model")
    return model if isinstance(model, str) and model else fallback


def _sse_data(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith(":") or not stripped.startswith("data:"):
        return None
    return stripped[len("data:") :].strip()


def _raise_retry_exhausted(status_code: int) -> None:
    if status_code == 429:
        raise RateLimitError("Language-model provider rate limit was exhausted")
    raise ProviderError(
        "Language-model provider remained unavailable after bounded retries",
        details={"status_code": status_code},
    )


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        try:
            return min(max(float(retry_after), 0.0), 10.0)
        except ValueError:
            pass
    return float(min(0.25 * (2**attempt), 5.0))

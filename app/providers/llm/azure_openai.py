"""Azure OpenAI chat-completions provider using the Azure data-plane contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import httpx
from pydantic import SecretStr

from app.providers.azure_openai import (
    azure_api_key_headers,
    azure_deployment_url,
    azure_models_url,
)
from app.providers.http_client import validate_model_catalog
from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider


class AzureOpenAILLMProvider(OpenAICompatibleLLMProvider):
    """Call an Azure deployment with ``api-key`` and ``api-version`` semantics."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: SecretStr | None,
        api_version: str,
        deployment: str,
        model: str | None,
        timeout_seconds: float,
        max_retries: int,
        max_concurrency: int,
        client: httpx.AsyncClient | None = None,
        healthcheck_mode: Literal["catalog", "inference"] = "inference",
    ) -> None:
        self._azure_catalog_model = model
        self._azure_completions_endpoint = azure_deployment_url(
            endpoint=endpoint,
            deployment=deployment,
            resource="chat/completions",
            api_version=api_version,
        )
        self._azure_models_endpoint = azure_models_url(
            endpoint=endpoint,
            api_version=api_version,
        )
        super().__init__(
            base_url=endpoint,
            api_key=None,
            model=model or deployment,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            max_concurrency=max_concurrency,
            client=client,
            request_headers=azure_api_key_headers(api_key),
            healthcheck_mode=healthcheck_mode,
        )

    @property
    def name(self) -> str:
        return "azure_openai"

    async def healthcheck(self) -> None:
        """Probe the deployment by default or use an explicitly selected catalog check."""
        if self._healthcheck_mode == "inference":
            await super().healthcheck()
            return
        decoded = await self._request_json("GET", self._azure_models_endpoint)
        if self._azure_catalog_model is not None:
            validate_model_catalog(decoded, self._azure_catalog_model)

    def _completions_endpoint(self) -> str:
        return self._azure_completions_endpoint

    def _models_endpoint(self) -> str:
        return self._azure_models_endpoint

    def _payload(
        self,
        *,
        messages: Sequence[dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool,
    ) -> dict[str, Any]:
        payload = super()._payload(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )
        # Azure routes to the model deployment encoded in the URL. Keeping a
        # second model field is both redundant and rejected by some API versions.
        payload.pop("model", None)
        payload.pop("stream_options", None)
        return payload

    async def _request_json(
        self,
        method: str,
        endpoint: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Retain the explicit method for a typed Azure adapter surface."""
        return await super()._request_json(method, endpoint, payload)

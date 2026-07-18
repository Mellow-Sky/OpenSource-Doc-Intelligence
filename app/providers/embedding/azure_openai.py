"""Azure OpenAI embedding provider."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import httpx
from pydantic import SecretStr

from app.providers.azure_openai import (
    azure_api_key_headers,
    azure_deployment_url,
    azure_models_url,
)
from app.providers.embedding.openai_compatible import OpenAICompatibleEmbeddingProvider
from app.providers.http_client import validate_model_catalog


class AzureOpenAIEmbeddingProvider(OpenAICompatibleEmbeddingProvider):
    """Embed through a deployment-scoped Azure OpenAI data-plane URL."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: SecretStr | None,
        api_version: str,
        deployment: str,
        model: str | None,
        dimension: int,
        batch_size: int,
        timeout_seconds: float,
        max_retries: int,
        max_concurrency: int,
        client: httpx.AsyncClient | None = None,
        healthcheck_mode: Literal["catalog", "inference"] = "inference",
    ) -> None:
        self._azure_catalog_model = model
        self._azure_embeddings_endpoint = azure_deployment_url(
            endpoint=endpoint,
            deployment=deployment,
            resource="embeddings",
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
            dimension=dimension,
            batch_size=batch_size,
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
        """Validate the deployment by default; catalog mode is an opt-in cheap probe."""
        if self._healthcheck_mode == "inference":
            await self.embed(["healthcheck"])
            return
        decoded = await self._http.request_json("GET", self._azure_models_endpoint)
        if self._azure_catalog_model is not None:
            validate_model_catalog(decoded, self._azure_catalog_model)

    def _embedding_resource(self) -> str:
        return self._azure_embeddings_endpoint

    def _embedding_payload(self, texts: Sequence[str]) -> dict[str, object]:
        # Deployment selection is carried by the URL in Azure OpenAI.
        return {"input": list(texts)}

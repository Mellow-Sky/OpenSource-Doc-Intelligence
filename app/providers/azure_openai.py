"""Azure OpenAI data-plane URL and authentication helpers."""

from __future__ import annotations

from urllib.parse import quote, urlencode, urlsplit, urlunsplit

from pydantic import SecretStr

from app.core.exceptions import ConfigurationError


def azure_api_key_headers(api_key: SecretStr | None) -> dict[str, str]:
    """Return Azure's required ``api-key`` header without logging the secret."""
    secret = api_key.get_secret_value().strip() if api_key is not None else ""
    if not secret:
        raise ConfigurationError("An Azure OpenAI API key is required")
    return {"api-key": secret}


def azure_deployment_url(
    *,
    endpoint: str,
    deployment: str,
    resource: str,
    api_version: str,
) -> str:
    """Build an Azure OpenAI deployment-scoped data-plane endpoint."""
    root = _azure_resource_root(endpoint)
    deployment_name = deployment.strip()
    if not deployment_name:
        raise ConfigurationError("An Azure OpenAI deployment name is required")
    normalized_resource = resource.strip("/")
    if normalized_resource not in {"chat/completions", "embeddings"}:
        raise ConfigurationError("Unsupported Azure OpenAI deployment resource")
    query = urlencode({"api-version": _api_version(api_version)})
    return (
        f"{root}/openai/deployments/{quote(deployment_name, safe='')}/{normalized_resource}?{query}"
    )


def azure_models_url(*, endpoint: str, api_version: str) -> str:
    """Build the authenticated model-list endpoint used by readiness checks."""
    root = _azure_resource_root(endpoint)
    return f"{root}/openai/models?{urlencode({'api-version': _api_version(api_version)})}"


def _azure_resource_root(endpoint: str) -> str:
    parsed = urlsplit(endpoint.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigurationError("Azure OpenAI endpoint must be an absolute HTTP(S) resource URL")
    if parsed.query or parsed.fragment:
        raise ConfigurationError("Azure OpenAI endpoint must not include a query or fragment")
    if parsed.path.rstrip("/"):
        raise ConfigurationError(
            "Azure OpenAI endpoint must be the resource root, without /openai/deployments"
        )
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")


def _api_version(api_version: str) -> str:
    normalized = api_version.strip()
    if not normalized:
        raise ConfigurationError("Azure OpenAI API version must not be empty")
    return normalized

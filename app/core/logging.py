"""Structured logging configuration with secret-safe processors."""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import structlog

_SENSITIVE_KEYS = {
    "admin_api_key",
    "api_key",
    "authorization",
    "embedding_api_key",
    "github_token",
    "judge_api_key",
    "llm_api_key",
    "password",
    "reranker_api_key",
    "secret",
    "token",
}
_AUTH_PATTERN = re.compile(r"(?i)(bearer\s+|(?:api[_ -]?key|token|password|secret)\s*[=:]\s*)\S+")
_URL_CREDENTIAL_PATTERN = re.compile(r"(https?://)[^/@\s]+@", re.IGNORECASE)


def _redact_secrets(
    _logger: Any, _method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    for key in tuple(event_dict):
        if key.casefold() in _SENSITIVE_KEYS and event_dict[key]:
            event_dict[key] = "[REDACTED]"
        else:
            event_dict[key] = _sanitize_value(event_dict[key])
    return event_dict


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: (
                "[REDACTED]"
                if str(key).casefold() in _SENSITIVE_KEYS and item
                else _sanitize_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, str):
        return _URL_CREDENTIAL_PATTERN.sub(
            r"\1[REDACTED]@",
            _AUTH_PATTERN.sub(r"\1[REDACTED]", value),
        )
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_sanitize_value(item) for item in value]
    return value


def configure_logging(level: str) -> None:
    """Configure stdlib and structlog to emit JSON events."""
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level, force=True)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _redact_secrets,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

from __future__ import annotations

from app.core.logging import _redact_secrets


def test_logging_redacts_nested_keys_authorization_strings_and_url_credentials() -> None:
    event = {
        "config": {
            "github_token": "secret-value",
            "safe": "keep",
        },
        "error": "Bearer abc123 failed at https://user:pass@example.test/path",
        "token_usage": {"total": 42},
    }

    sanitized = _redact_secrets(None, "info", event)

    assert sanitized["config"] == {"github_token": "[REDACTED]", "safe": "keep"}
    assert "abc123" not in sanitized["error"]
    assert "user:pass" not in sanitized["error"]
    assert sanitized["token_usage"] == {"total": 42}

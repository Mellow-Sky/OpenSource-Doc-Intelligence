"""Optional OpenTelemetry setup loaded only when explicitly enabled."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from fastapi import FastAPI

from app.core.config import Settings
from app.core.exceptions import ConfigurationError


def configure_telemetry(app: FastAPI, settings: Settings) -> Any | None:
    """Instrument FastAPI and OTLP export, or remain dependency-free when disabled."""
    if not settings.enable_telemetry:
        return None
    try:
        trace_api = import_module("opentelemetry.trace")
        exporter_module = import_module("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")
        fastapi_module = import_module("opentelemetry.instrumentation.fastapi")
        resource_module = import_module("opentelemetry.sdk.resources")
        trace_sdk = import_module("opentelemetry.sdk.trace")
        export_sdk = import_module("opentelemetry.sdk.trace.export")
    except ImportError as exc:
        raise ConfigurationError(
            "OpenTelemetry dependencies are missing; install the telemetry extra"
        ) from exc

    resource = resource_module.Resource.create({"service.name": settings.otel_service_name})
    provider = trace_sdk.TracerProvider(resource=resource)
    exporter_options: dict[str, Any] = {}
    if settings.otel_exporter_otlp_endpoint:
        exporter_options["endpoint"] = settings.otel_exporter_otlp_endpoint
    exporter = exporter_module.OTLPSpanExporter(**exporter_options)
    provider.add_span_processor(export_sdk.BatchSpanProcessor(exporter))
    trace_api.set_tracer_provider(provider)
    fastapi_module.FastAPIInstrumentor.instrument_app(app)
    return provider

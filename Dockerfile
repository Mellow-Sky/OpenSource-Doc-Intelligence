# syntax=docker/dockerfile:1.7
FROM python:3.12-slim-bookworm AS builder

ARG INSTALL_LOCAL_MODELS=false

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /build
COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock README.md ./
RUN if [ "$INSTALL_LOCAL_MODELS" = "true" ]; then \
      uv sync --frozen --no-dev --no-install-project --extra models --extra telemetry; \
    else \
      uv sync --frozen --no-dev --no-install-project --extra telemetry; \
    fi

COPY app ./app
COPY evaluation ./evaluation
RUN if [ "$INSTALL_LOCAL_MODELS" = "true" ]; then \
      uv sync --frozen --no-dev --no-editable --extra models --extra telemetry; \
    else \
      uv sync --frozen --no-dev --no-editable --extra telemetry; \
    fi

FROM python:3.12-slim-bookworm AS runtime

ARG GIT_COMMIT=""

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GIT_COMMIT=${GIT_COMMIT} \
    PATH=/build/.venv/bin:$PATH

LABEL org.opencontainers.image.revision=${GIT_COMMIT}

RUN apt-get update \
    && apt-get install --no-install-recommends --yes ca-certificates git libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 odi \
    && useradd --uid 10001 --gid odi --create-home odi

# Keep the virtual environment at the builder path: generated console-script
# shebangs are absolute and a relocated venv would make uvicorn/alembic fail.
COPY --from=builder /build/.venv /build/.venv

WORKDIR /app
COPY --chown=odi:odi alembic.ini ./
COPY --chown=odi:odi migrations ./migrations
COPY --chown=odi:odi prompts ./prompts
COPY --chown=odi:odi config ./config
COPY --chown=odi:odi evaluation/datasets ./evaluation/datasets
COPY --chown=odi:odi scripts ./scripts
COPY --chown=odi:odi docker/entrypoint.sh /usr/local/bin/odi-entrypoint

RUN chmod 0555 /usr/local/bin/odi-entrypoint \
    && mkdir -p \
      /app/.cache/ingestion \
      /app/evaluation/reports \
      /home/odi/.cache/huggingface \
    && chown -R odi:odi /app /home/odi/.cache
USER odi

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

ENTRYPOINT ["odi-entrypoint"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

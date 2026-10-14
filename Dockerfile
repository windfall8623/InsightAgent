# syntax=docker/dockerfile:1

# ---------- Stage 1: Frontend Build ----------
FROM node:20-alpine AS frontend-builder
WORKDIR /build
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---------- Stage 2: Python Dependencies ----------
FROM python:3.12-slim AS python-builder

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first so edits to source do not invalidate layer cache
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install api, graph engine, and dependencies
RUN uv pip install --system --no-cache ".[api,full]" langgraph redis

# ---------- Stage 3: Runtime ----------
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="InsightAgent" \
      org.opencontainers.image.description="Enterprise Multi-Modal Causal Analytics Agent Platform" \
      org.opencontainers.image.licenses="MIT"

# Run unprivileged for container security
RUN useradd --create-home --uid 10001 insight

WORKDIR /app
COPY --from=python-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin
COPY src ./src
COPY --from=frontend-builder /build/dist ./frontend/dist

RUN mkdir -p /app/data && chown -R insight:insight /app
USER insight

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["sh", "-c", "uvicorn --factory insight_agent.api:get_app --host 0.0.0.0 --port ${PORT}"]
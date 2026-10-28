# Multi-stage hardened distroless container for XLAxiom compiler forensics
# Build stage
FROM python:3.12-slim-bookworm AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, reproducible dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Build wheel distribution
RUN uv build --wheel --out-dir /dist

# Runtime stage: hardened distroless/minimal non-root container
FROM python:3.12-slim-bookworm AS runner

LABEL org.opencontainers.image.title="XLAxiom" \
      org.opencontainers.image.description="Evidence-first differential testing and failure minimization for JAX and OpenXLA" \
      org.opencontainers.image.vendor="Google Ecosystem ML Infrastructure" \
      org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# Create non-root user
RUN groupadd -g 10001 xlaxiom && \
    useradd -u 10001 -g xlaxiom -m -s /bin/bash xlaxiom

COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl

USER 10001:10001

ENTRYPOINT ["xlaxiom"]
CMD ["doctor"]


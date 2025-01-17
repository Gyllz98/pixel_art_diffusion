FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies with caching
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc

# Install Python dependencies with caching
COPY requirements.txt requirements_dev.txt ./
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt -r requirements_dev.txt

# Copy source code and other files
COPY src/ src/
COPY configs/ configs/
COPY data/ data/
COPY models/ models/
COPY README.md pyproject.toml ./

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-u", "src/pixel_art_diffusion/train.py"]

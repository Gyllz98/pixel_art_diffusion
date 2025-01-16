# Base image
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY configs/ configs/
COPY data/ data/
COPY models/ models/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install --no-cache-dir --verbose -r requirements_dev.txt
RUN pip install . --no-deps --no-cache-dir --verbose

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-u", "src/pixel_art_diffusion/train.py"]

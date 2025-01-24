# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Initialize git repository (required for DVC)
RUN git init

# Copy requirements first for caching
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy the entire project
COPY . .

# Install the pixel_art_diffusion package in development mode
RUN pip install -e .

# Configure git and perform DVC operations during build
RUN git config --global --add safe.directory /app && \
    dvc pull && \
    PYTHONPATH=/app/src python -m pixel_art_diffusion.data

# Set environment variables
ENV PORT=8080
ENV HOST=0.0.0.0
ENV PYTHONPATH=/app/src

# Expose the port
EXPOSE 8080

# Run the API server
CMD ["uvicorn", "pixel_art_diffusion.api:app", "--host", "0.0.0.0", "--port", "8080"]
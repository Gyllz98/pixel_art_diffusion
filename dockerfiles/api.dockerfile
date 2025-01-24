# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy DVC files
COPY .dvc/ .dvc/
COPY .dvcignore .
COPY dvc.yaml .
COPY dvc.lock .

# Copy the application code
COPY src/pixel_art_diffusion /app/pixel_art_diffusion/

# Set environment variables
ENV PORT=8080
ENV HOST=0.0.0.0

# Expose the port
EXPOSE 8080

# Run the FastAPI server
CMD dvc pull && uvicorn pixel_art_diffusion.api:app --host 0.0.0.0 --port 8080

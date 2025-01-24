# Pixel Art Diffusion Documentation

## Overview
Pixel Art Diffusion is an MLOps project for generating pixel art using diffusion models. The system focuses on generating item-category pixel art using the HuggingFace Diffusers library.

## Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-compatible GPU (optional but recommended)
- Git

### Setup
Step 1. Clone the repository:
```bash
git clone <repository-url>
cd pixel_art_diffusion
```
Step 2. Install the package:
```bash
pip install -e .
# For development dependencies:
pip install -e .[dev]
```
Step 3. Launch the application:
```bash
pixel-art-app
```

## Project Structure

### Core Components
- `api.py`: FastAPI server for model inference
- `data.py`: Data loading and preprocessing pipeline
- `model.py`: PixelArtDiffusion model implementation
- `train.py`: Training pipeline
- `frontend.py`: Streamlit web interface

### Configuration
The project uses Hydra for configuration management. Key config files:

- `configs/config.yaml`: Main configuration file
- `configs/model/`: Model-specific configurations
- `configs/training/`: Training configurations

## Data Setup

To set up the dataset for training:

1. Pull data from DVC:
```bash
dvc pull
```

2. Extract and preprocess data:
```bash
python -m src.pixel_art_diffusion.data
```

This will:

- Download the raw pixel art dataset from cloud storage
- Extract the archive to data/processed/
- Create necessary data structures for training

## Usage

### Training
Train the model using:
```bash
python src/pixel_art_diffusion/train.py --run-name "my_training_run" --num-epochs 100 --label-subset 3
```

Parameters:

- `run_name`: Name for the training run
- `num_epochs`: Number of training epochs
- `label_subset`: Label category to train on (3 for items)

### Generation
Generate samples using:
```bash
python src/pixel_art_diffusion/generate_sample.py generate --num-samples 32
```

### Web Interface
Launch the web interface:
```bash
python src/pixel_art_diffusion/run_app.py
```

### API
The FastAPI server provides the following endpoints:

#### GET /
Health check endpoint

#### GET /model_info
Returns model information:

- Model name
- Loading status
- Device type

#### POST /generate
Generates pixel art images:

- Parameter: `num_samples` (optional, default: 4)
- Returns: PNG image containing generated samples

#### GET /data_drift
Performs and returns drift analysis of generated samples

## Development

### Testing
Run tests:
```bash
invoke test
```

### Documentation
Build documentation:
```bash
invoke build-docs
```

Serve documentation locally:
```bash
invoke serve-docs
```

### Docker
Build Docker images:
```bash
invoke docker-build
```

## Monitoring

### Metrics
The API includes Prometheus metrics:

- `pixel_art_generate_requests_total`: Total generation requests
- `pixel_art_memory_usage_bytes`: Application memory usage
- `pixel_art_generation_time_seconds`: Generation time histogram

### Data Drift Monitoring
The system monitors data drift between generated samples and the training dataset using Evidently AI, tracking:

- RGB channel statistics
- Image brightness and contrast
- Data quality metrics

## Deployment
The project supports deployment to Google Cloud Run:

1. Build the container image
2. Push to Google Container Registry
3. Deploy to Cloud Run

Configuration is handled through `cloudbuild.yaml`.

## Contributing
1. Fork the repository
2. Create a feature branch
3. Run tests and linting
4. Submit a pull request

The project uses pre-commit hooks for code quality. Install them:
```bash
pip install pre-commit
pre-commit install
```

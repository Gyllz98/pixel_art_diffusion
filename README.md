# pixel_art_diffusion
This repo is made as an exercise in MLOps for the course 02476 Machine learning operations at DTU.

## Project description
Our project aims to create an MLOps pipeline for generating categorized pixel art using diffusion models. Initially, we will focus specifically on label 3 (items) from our dataset, which contains 35,000 diverse images. This focused approach will allow us to better tune our diffusion model and establish our MLOps practices before potentially expanding to other categories.

We'll use [HuggingFace Diffusers library](https://huggingface.co/docs/diffusers/index) with the following specific integrations:
- Unconditional diffusion model implementation optimized for item generation
- Data pipeline handling pre-processed numpy arrays
- MLflow for tracking generation quality and model performance
- Docker containerization with support for numpy array processing
- The framework will need to handle sprites.npy and the corresponding labels from sprites_labels.npy

We are going to use the [Pixel Art](https://www.kaggle.com/datasets/ebrahimelgazar/pixel-art) dataset. The dataset structure provides multiple formats:
- 89,400 16x16 JPEG images in the images directory
- Labels in both CSV (labels.csv) and numpy format (sprites_labels.npy)
- Pre-processed numpy array of sprites (sprites.npy) - 90MB total
- We'll use the numpy format (sprites.npy) for efficient data loading
- Label distribution:
    - Label 0 (characters): 8000 images
    - Label 1 (creatures): 32400 images
    - Label 2 (food): 6000 images
    - Label 3 (items): 35000 images
    - Label 4 (character sideview): 8000 images

For this project, we'll focus exclusively on Label 3 (items) due to its large sample size and diverse representation of pixel art items.
Specifically, we plan to use the following models:

1. Unconditional DDPM (Denoising Diffusion Probabilistic Model):
    - Input: 16x16x3 RGB images from the items category
    - Modified to respect the discrete nature of pixel art
    - Optimized for item-specific features and patterns

2. DDIM (Denoising Diffusion Implicit Models):
    - Faster sampling while maintaining quality
    - Attention mechanisms optimized for item generation

## Project structure

The directory structure of the project looks like this:
```txt
|   .dvcignore
|   .gitignore
|   .pre-commit-config.yaml
|   cloudbuild.yaml
|   LICENSE
|   pyproject.toml
|   README.md
|   requirements.txt
|   requirements_dev.txt
|   tasks.py
|
+---.dvc
|       .gitignore
|       config
|
+---.github
|   |   dependabot.yaml
|   |
|   \---workflows
|           cml_data.yaml
|           codecheck.yaml
|           docs.yaml
|           pre_commit.yaml
|           tests.yaml
|
+---configs
|   |   .gitkeep
|   |   cloud_config.yaml
|   |   config.yaml
|   |   sweep.yaml
|   |   test_config.yaml
|   |
|   \---experiments
|           exp1.yaml
|           exp2.yaml
|
+---data
|   +---processed
|   |       .gitkeep
|   |
|   \---raw
|           .gitkeep
|           archive.zip.dvc
|
+---dockerfiles
|       api.dockerfile
|       train.dockerfile
|
+---docs
|   |   mkdocs.yaml
|   |   README.md
|   |
|   \---source
|           index.md
|
+---models
|       .gitkeep
|       pixel_art_diffusion-checkpoint-epoch-50.pt
|       pixel_art_diffusion-final.pt
|
+---notebooks
|       .gitkeep
|       generated_samples.png
|       prototyping.ipynb
|
+---reports
|   |   .gitkeep
|   |   README.md
|   |   report.html
|   |   report.py
|   |
|   \---figures
|           .gitkeep
|           ARshot1.jpeg
|           ARshot2.jpeg
|           bucket.png
|           bucketshot1.jpeg
|           bucketshot2.jpeg
|           build.png
|           cloudbuild.jpeg
|           overview.png
|           registry.png
|           wandb.jpeg
|           wandb.png
|
+---src
|   |   __init__.py
|   |
|   \---pixel_art_diffusion
|           api.py
|           cloud_function.py
|           data.py
|           data_drift.py
|           frontend.py
|           generate_sample.py
|           model.py
|           run_app.py
|           run_profiling.py
|           train.py
|           visualize.py
|           __init__.py
|
\---tests
    |   pytest.ini
    |   test_api.py
    |   test_data.py
    |   test_model.py
    |   test_train.py
    |   __init__.py
    |
    \---performancetests
            locustfile.py
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

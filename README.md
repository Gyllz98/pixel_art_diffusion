# pixel_art_diffusion

Our project aims to create an MLOps pipeline for generating categorized pixel art using diffusion models. Beyond just generating pixel art, we'll leverage the labeled nature of the dataset to implement conditional generation - allowing us to generate specific types of characters or objects. This adds an interesting supervised learning aspect to the traditional diffusion model approach.

We'll use HuggingFace Diffusers library with the following specific integrations:

- Conditional diffusion model implementation to utilize the label information
- Data pipeline handling both images (JPEG) and labels (from CSV and numpy formats)
- MLflow for tracking both generation quality and classification accuracy
- Docker containerization with support for numpy array processing
- The framework will need to handle both the image files and their corresponding labels from labels.csv and sprites_labels.npy.

We are going to use the [Pixel Art](https://www.kaggle.com/datasets/ebrahimelgazar/pixel-art/data) dataset. The dataset structure provides multiple formats we can use:
- 89,000 16x16 JPEG images in the images directory
- Labels in both CSV (labels.csv) and numpy format (sprites_labels.npy)
- Pre-processed numpy array of sprites (sprites.npy) - 90MB total
- We'll start with the numpy format (sprites.npy) for efficient data loading, with labels from sprites_labels.npy for conditioning.

Specificly, we plan to use the following models:
1. Conditional DDPM:
    - Input: 16x16x3 RGB images with label conditioning
    - Modified to respect the discrete nature of pixel art
    - Integrated with the label information for category-specific generation

2. Conditional DDIM:
    - Faster sampling while maintaining category accuracy
    - Label-aware attention mechanisms

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

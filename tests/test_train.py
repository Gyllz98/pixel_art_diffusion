import pytest
import os
from pathlib import Path
from unittest.mock import patch

from pixel_art_diffusion.train import train_model, get_models_dir
from tests import _PATH_MODELS


def test_get_models_dir():
    """Test that models directory path is correct"""
    models_dir = get_models_dir()
    assert models_dir.exists(), f"Directory path {models_dir} does not exist."
    assert models_dir.is_dir(), f"Directory path {models_dir} is not a directory."
    assert Path(models_dir) == Path(
        _PATH_MODELS
    ), f"Directory path {models_dir} is not the correct path {_PATH_MODELS}."
    print("\nModel directory path is correct.")


def test_train_model_cli():
    """Test CLI wrapper function"""
    with patch("pixel_art_diffusion.train.train_model_hydra"), patch.dict(os.environ, clear=True):
        try:
            train_model(run_name="test-cli-run", num_epochs=1, label_subset=[3])
            print("\nCLI wrapper function works correctly.")
        except Exception as e:
            pytest.fail(f"CLI training failed: {e}")

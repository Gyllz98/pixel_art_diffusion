import pytest
import torch
from pathlib import Path
from tests import _PATH_MODELS
from src.pixel_art_diffusion.model import PixelArtDiffusion # Model

PATH = Path(_PATH_MODELS)
model = PixelArtDiffusion()


# Skips tests if the data files does not exist
@pytest.mark.skipif(not (PATH / "pixel_art_diffusion-checkpoint-epoch-50.pt").exists(), reason="Missing model checkpoint, model has not been initialized.")

def test_model_init():
    ### Is the model correctly initialized? ###
    assert model.image_size == 16, f"Model should have image size = 16, received {model.image_size}"
    assert model.num_channels == 3, f"Model should have 3 channels (RGB), received {model.num_channels}"
    assert hasattr(model, "noise_scheduler"), f"Model should have noise scheduler, received {model.noise_scheduler}"
    print(" Correctly initialized model.")

def test_model_output():
    ### Is the model output shape correct? ###
    samples = 1
    output = model.generate_samples(samples)
    expected_output_shape = (samples, 3, 16, 16)
    assert output.shape == expected_output_shape, f"Output shape is not 4, received {output.shape}"
    print(" Correct model shape.")

# Test model device handling?
# Test model checkpoint loading and saving?
# Test model pixel art quantization?
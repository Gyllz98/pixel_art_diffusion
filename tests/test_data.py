import pytest
import torch
from pathlib import Path
from tests import _PATH_DATA_RAW, _PATH_DATA_PROCESSED
from src.pixel_art_diffusion.data import PixelArtDataset  


PATH = Path(_PATH_DATA_PROCESSED)

# Skips tests if the data files does not exist
# pytestmark = pytest.mark.skipif(not (PATH / "labels1.csv").exists(), reason="labels.csv not found")
pytestmark = pytest.mark.skipif(not (PATH / "labels.csv").exists(), reason="labels1.csv not found",
)

dataset = PixelArtDataset(PATH) # Set up path + dataset

def test_dataset_init():
    ### Is the dataset initialized correctly?\n ###
    assert len(dataset) == 89400, f"Dataset ({len(dataset)} images) should contain 89400 images."
    label_distribution = dataset.get_label_distribution()
    assert label_distribution[3] == 35000, f"Expected 35000 images in label 3, got {label_distribution[3]}"
    print(" Correctly initialized")

def test_data_format():
    ### Are the first item from the dataset in the correct format? ###
    item = dataset[0]
    assert isinstance(item, dict), f"Dataset item should be a dictionary"
    assert "pixel_values" in item, "Item should contain pixel_values"
    assert "label" in item, "Item should contain label"
    pixel_value = item["pixel_values"]
    assert pixel_value.shape == (3, 16, 16), f"Image should be 3x16x16, received {pixel_value.shape}"
    label = item["label"]
    assert label.shape == (5,), f"Label should be one-hot encoded with 5 categories, got shape {label.shape}"
    assert label.sum() == 1, "One-hot encoded label should sum to 1"
    print(" Correct dataset format.")
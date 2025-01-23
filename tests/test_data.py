import pytest
import os
from pathlib import Path
from pixel_art_diffusion.data import PixelArtDataset
from test import _PROJECT_ROOT


ROOT = Path(__file__).resolve().parent.parent.parent
PATH_RAW = ROOT / "data"
PATH_PROC = ROOT / "processed"

# Skips tests if the data files does not exist
pytestmark = pytest.mark.skipif(
    not (PATH_RAW / "archive.zip.dvc").exists(),
    reason="archive.zip.dvc not found",
)
pytestmark = pytest.mark.skipif(
    not (PATH_PROC / "labels.csv").exists(),
    reason="label.csv not found",
)


dataset = PixelArtDataset(PATH_PROC)  # Set up path + dataset


def test_dataset_init():
    ### Is the dataset initialized correctly?###
    assert len(dataset) == 89400, f"Dataset ({len(dataset)} images) should contain 89400 images."
    label_distribution = dataset.get_label_distribution()
    assert label_distribution[3] == 35000, f"Expected 35000 images in label 3, got {label_distribution[3]}"
    print("\nCorrectly initialized")


def test_data_format():
    ### Are the first item from the dataset in the correct format? ###
    item = dataset[0]
    assert isinstance(item, dict), "Dataset item should be a dictionary"
    assert "pixel_values" in item, "Item should contain pixel_values"
    assert "label" in item, "Item should contain label"
    pixel_value = item["pixel_values"]
    assert pixel_value.shape == (3, 16, 16), f"Image should be 3x16x16, received {pixel_value.shape}"
    label = item["label"]
    assert label.shape == (5,), f"Label should be one-hot encoded with 5 categories, got shape {label.shape}"
    assert label.sum() == 1, "One-hot encoded label should sum to 1"
    print("\nCorrect dataset format")

if __name__ == "__main__":
    print("Current working directory:", Path.cwd())
    print("ROOT", PROJECT_ROOT)
    print("_PATH_DATA_RAW:", PATH_RAW)
    print("_PATH_DATA_PROCESSED:", PATH_PROC)
    print("Resolved PATH_RAW:", PATH_RAW.resolve())
    print("Resolved PATH_PROC:", PATH_PROC.resolve())
    print("Does labels.csv exist?", (PATH_PROC / "labels.csv").exists())
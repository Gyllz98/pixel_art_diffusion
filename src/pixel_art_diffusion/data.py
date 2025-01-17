from pathlib import Path
import zipfile
import typer
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from PIL import Image

def get_data_dir():
    """Helper function to get the data directory path"""
    return Path(__file__).parent.parent.parent / "data"

class PixelArtDataset(Dataset):
    """Dataset for pixel art sprites and their labels."""

    def __init__(self, 
                 data_path: str | Path,  # Made more flexible to accept string or Path
                 transform=None,
                 label_subset=None,
                 calculate_stats=False
                ) -> None:
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the data folder containing labels.csv and images/images/*.jpg
            transform: Optional transforms to be applied on images
            label_subset: Optional list of indices to include (0-4 for the 5 categories)
            calculate_stats: Whether to recalculate dataset statistics
        """
        self.data_path = Path(data_path)
        self.transform = transform
        
        # Load labels from CSV
        labels_path = self.data_path / "labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found at {labels_path}")
            
        self.labels_df = pd.read_csv(labels_path)
        
        # Convert string representation of arrays to numpy arrays
        self.labels_df['Label'] = self.labels_df['Label'].apply(
            lambda x: np.array([float(n) for n in x.strip('[]').split()])
        )
        
        # Filter by label subset if specified
        if label_subset is not None:
            mask = self.labels_df['Label'].apply(
                lambda x: any(x[i] == 1.0 for i in label_subset)
            )
            self.labels_df = self.labels_df[mask].reset_index(drop=True)
        
        # Build image paths
        images_dir = self.data_path / "images" / "images"
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found at {images_dir}")
            
        self.image_paths = [
            images_dir / f"image_{idx-1}.JPEG"
            for idx in self.labels_df['Image Index']
        ]
        
        # Verify all images exist
        missing_images = [path for path in self.image_paths if not path.exists()]
        if missing_images:
            raise FileNotFoundError(f"Missing {len(missing_images)} images, first few: {missing_images[:3]}")
        
        # Calculate statistics if needed
        if calculate_stats:
            self.mean, self.std = self.calculate_statistics()
        
        # Set default transform if none provided
        if self.transform is None:
            self.transform = self.get_default_transforms()
            
        # Print dataset info
        print(f"Loaded dataset with {len(self.image_paths)} sprites")
        label_counts = self.get_label_distribution()
        print("Label distribution:")
        for idx, count in enumerate(label_counts):
            print(f"Category {idx}: {count} images")

    # ... [rest of the dataset methods remain unchanged] ...

    def preprocess(self, output_folder: Path = None) -> None:
        """Preprocess the raw data and save it to the output folder."""
        if output_folder is None:
            output_folder = get_data_dir() / "processed"
            
        # Create output folder if it doesn't exist
        output_folder.mkdir(parents=True, exist_ok=True)

        # Check if archive exists
        archive_path = self.data_path / "archive.zip"
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found at {archive_path}")

        # Unzip the archive
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            print(f"Extracting contents to {output_folder}")
            zip_ref.extractall(output_folder)

        print("Extraction complete!")


@typer.command()
def preprocess(
    raw_data_path: Path = None,
    output_folder: Path = None
) -> None:
    """
    Preprocess the raw data and save it to the processed folder.
    
    Args:
        raw_data_path: Path to raw data directory. Defaults to data/raw
        output_folder: Path to output directory. Defaults to data/processed
    """
    if raw_data_path is None:
        raw_data_path = get_data_dir() / "raw"
    if output_folder is None:
        output_folder = get_data_dir() / "processed"

    print(f"Preprocessing data from {raw_data_path} to {output_folder}")
    dataset = PixelArtDataset(raw_data_path)
    dataset.preprocess(output_folder)
    print("Preprocessing complete!")


if __name__ == "__main__":
    typer.run(preprocess)
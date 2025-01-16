from pathlib import Path
import zipfile
import typer
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from PIL import Image

class PixelArtDataset(Dataset):
    """Dataset for pixel art sprites and their labels."""

    def __init__(self, 
                 data_path: Path,
                 transform=None,
                 label_subset=None,  # Optional: filter for specific label indices
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
        self.labels_df = pd.read_csv(self.data_path / "labels.csv")
        
        # Convert string representation of arrays to numpy arrays
        self.labels_df['Label'] = self.labels_df['Label'].apply(
            lambda x: np.array([float(n) for n in x.strip('[]').split()])
        )
        
        # Filter by label subset if specified
        if label_subset is not None:
            # For one-hot encoded labels, we need to check the specific indices
            mask = self.labels_df['Label'].apply(
                lambda x: any(x[i] == 1.0 for i in label_subset)
            )
            self.labels_df = self.labels_df[mask].reset_index(drop=True)
        
        # Build image paths - using Image Index to construct filenames
        # Update the image path construction in the __init__ method:
        self.image_paths = [
            self.data_path / "images" / "images" / f"image_{idx-1}.JPEG"
            for idx in self.labels_df['Image Index']
        ]
        
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

    def get_label_distribution(self):
        """Calculate the distribution of labels in the dataset."""
        # Sum up one-hot vectors to get counts for each category
        return np.sum(np.stack(self.labels_df['Label'].values), axis=0)

    def calculate_statistics(self):
        """Calculate dataset mean and std."""
        print("Calculating dataset statistics...")
        
        pixel_sum = torch.zeros(3)
        pixel_sq_sum = torch.zeros(3)
        num_pixels = 0

        for img_path in self.image_paths:
            img = Image.open(img_path)
            img_tensor = transforms.ToTensor()(img)
            
            pixel_sum += img_tensor.sum(dim=[1, 2])
            pixel_sq_sum += (img_tensor ** 2).sum(dim=[1, 2])
            num_pixels += img_tensor.shape[1] * img_tensor.shape[2]

        mean = pixel_sum / num_pixels
        std = torch.sqrt(pixel_sq_sum/num_pixels - mean**2)
        
        return mean.numpy(), std.numpy()

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        # Load image
        img_path = self.image_paths[index]
        image = Image.open(img_path)
        
        # Get one-hot encoded label
        label = torch.tensor(self.labels_df.iloc[index]['Label'], dtype=torch.float)
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        return {
            "pixel_values": image,
            "label": label
        }

    def get_default_transforms(self):
        """Return transforms using calculated dataset statistics."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.mean if hasattr(self, 'mean') else [0.5, 0.5, 0.5],
                std=self.std if hasattr(self, 'std') else [0.5, 0.5, 0.5]
            )
        ])

    def preprocess(self, output_folder: Path = Path("data/processed")) -> None:
        """Preprocess the raw data and save it to the output folder."""
        # Create output folder if it doesn't exist
        output_folder.mkdir(parents=True, exist_ok=True)

        # Unzip the archive
        with zipfile.ZipFile(self.data_path / "archive.zip", "r") as zip_ref:
            print(f"Extracting contents to {output_folder}")
            zip_ref.extractall(output_folder)

        print("Extraction complete!")
        # TODO: Add any additional preprocessing steps here
        pass


def preprocess(raw_data_path: Path = Path("data/raw"), output_folder: Path = Path("data/processed")) -> None:
    print("Preprocessing data...")
    dataset = PixelArtDataset(raw_data_path)
    dataset.preprocess(output_folder)

    pass


if __name__ == "__main__":
    typer.run(preprocess)

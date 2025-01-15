from pathlib import Path
import zipfile
import typer
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path = Path("data/raw")) -> None:
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""
        # TODO: Implement dataset length
        return 0

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        # TODO: Implement getting individual samples
        return None

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
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)

    pass


if __name__ == "__main__":
    typer.run(preprocess)

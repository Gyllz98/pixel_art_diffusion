from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report
import pandas as pd
import numpy as np
from PIL import Image
import os
import base64
from io import BytesIO
import subprocess
from pathlib import Path
from pixel_art_diffusion.data import extract
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_image_dataframe(image_dir):
    """Convert directory of images to dataframe with pixel statistics"""
    image_path = Path(image_dir)

    # Check if directory exists and has JPEG files
    if not image_path.exists() or not any(f.endswith(".JPEG") for f in os.listdir(image_dir)):
        logging.info("Extracting data...")
        try:
            subprocess.run(["dvc", "pull"], check=True)
            extract()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to pull data with DVC: {e}")

    data = []
    for img_file in os.listdir(image_dir):
        if img_file.endswith(".JPEG"):
            img_path = os.path.join(image_dir, img_file)
            img = Image.open(img_path)
            img_array = np.array(img)

            stats = calculate_image_stats(img_array)
            data.append(stats)

    if not data:
        raise RuntimeError(f"No JPEG images found in {image_dir} after DVC pull")

    return pd.DataFrame(data)


def create_generated_dataframe(generated_images):
    """Convert base64 images to features dataframe for drift analysis"""
    image_features = []

    for img_data in generated_images:
        # Decode base64 image
        img_bytes = base64.b64decode(img_data["image"])
        img = Image.open(BytesIO(img_bytes))
        img_array = np.array(img)

        # Extract features
        features = {
            "mean_r": np.mean(img_array[:, :, 0]),
            "mean_g": np.mean(img_array[:, :, 1]),
            "mean_b": np.mean(img_array[:, :, 2]),
            "std_r": np.std(img_array[:, :, 0]),
            "std_g": np.std(img_array[:, :, 1]),
            "std_b": np.std(img_array[:, :, 2]),
            "brightness": np.mean(img_array),
            "contrast": np.std(img_array),
        }
        image_features.append(features)

    return pd.DataFrame(image_features)


def run_report(report_path, current_data, reference_data):
    # Create drift report
    drift_report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])

    drift_report.run(current_data=current_data, reference_data=reference_data)
    drift_report.save_html(report_path)


def calculate_image_stats(img_array):
    """Calculate RGB statistics for an image array"""
    return {
        "mean_r": img_array[:, :, 0].mean(),
        "mean_g": img_array[:, :, 1].mean(),
        "mean_b": img_array[:, :, 2].mean(),
        "std_r": img_array[:, :, 0].std(),
        "std_g": img_array[:, :, 1].std(),
        "std_b": img_array[:, :, 2].std(),
    }

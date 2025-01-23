import requests
import base64
import torch
from io import BytesIO
from pixel_art_diffusion.model import PixelArtDiffusion
from diffusers import DDPMScheduler
from pixel_art_diffusion.visualize import visualize_samples
from pixel_art_diffusion.generate_sample import cpu_map_location
from datetime import datetime
from PIL import Image
import numpy as np
import json
from typing import Union

CLOUD_URL_MODEL = "https://europe-west1-rare-mechanic-447712-n4.cloudfunctions.net/function-4-pixel-art"
CLOUD_URL_DATA = "https://europe-west1-rare-mechanic-447712-n4.cloudfunctions.net/pure-chaos"


def load_model_from_cloud():
    # Get model file from cloud function
    response = requests.get(CLOUD_URL_MODEL)
    if response.ok:
        # Decode base64 string back to bytes
        model_bytes = base64.b64decode(response.json()["model"])

        # Initialize your model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        if device.type == "cpu":
            checkpoint = torch.load(BytesIO(model_bytes), map_location=cpu_map_location, weights_only=False)
        else:
            checkpoint = torch.load(BytesIO(model_bytes), map_location=device, weights_only=False)
        model = PixelArtDiffusion(device=device)
        model.model.load_state_dict(checkpoint["model_state_dict"])
        model.noise_scheduler = DDPMScheduler.from_config(checkpoint["scheduler_config"])

        return model
    else:
        raise Exception(f"Failed to load model: {response.text}")


def get_samples_from_cloud():
    try:
        response = requests.get(f"{CLOUD_URL_DATA}/samples")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to get samples: {str(e)}")


def get_images_from_cloud():
    try:
        response = requests.get(f"{CLOUD_URL_DATA}/extract_images")
        response.raise_for_status()
        data = response.json()

        if "images" not in data:
            raise ValueError("Response missing 'images' field")

        return data["images"]

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to get images: {str(e)}")
    except (ValueError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Invalid response format: {str(e)}")


def visualize_single_sample(sample_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert a single sample to a viewable image"""
    # Ensure correct shape and scaling
    if isinstance(sample_data, torch.Tensor):
        sample_data = sample_data.cpu().numpy()

    # Add batch dimension if needed
    if len(sample_data.shape) == 3:
        sample_data = np.expand_dims(sample_data, 0)

    return visualize_samples(torch.from_numpy(sample_data), api=True)


def store_generated_samples(samples: torch.Tensor, num_samples: int):
    try:
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        uploaded_samples = []

        for i in range(samples.shape[0]):
            sample_data = samples[i]
            grid_array = visualize_single_sample(sample_data)
            image = Image.fromarray(grid_array)

            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)

            # Convert bytes to base64 for transmission
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

            response = requests.post(
                f"{CLOUD_URL_DATA}/upload",
                json={"image": img_base64, "filename": f"generated/{batch_id}_sample_{i}.png"},
            )

            if not response.ok:
                raise Exception(f"Upload failed: {response.text}")

            uploaded_samples.append(response.json())

        return uploaded_samples
    except Exception:
        raise


if __name__ == "__main__":
    model = load_model_from_cloud()
    samples = model.generate_samples(num_samples=32)
    visualize_samples(samples)

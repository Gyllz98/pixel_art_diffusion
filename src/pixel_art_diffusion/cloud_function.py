import requests
import base64
import torch
from io import BytesIO
from pixel_art_diffusion.model import PixelArtDiffusion
from diffusers import DDPMScheduler
from pixel_art_diffusion.visualize import visualize_samples

CLOUD_URL = "https://europe-west1-rare-mechanic-447712-n4.cloudfunctions.net/function-4-pixel-art"


def load_model_from_cloud():
    # Get model file from cloud function
    response = requests.get(CLOUD_URL)
    if response.ok:
        # Decode base64 string back to bytes
        model_bytes = base64.b64decode(response.json()["model"])

        # Load model using BytesIO to avoid saving to disk
        checkpoint = torch.load(BytesIO(model_bytes), map_location="cpu")

        # Initialize your model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PixelArtDiffusion(device=device)
        model.model.load_state_dict(checkpoint["model_state_dict"])
        model.noise_scheduler = DDPMScheduler.from_config(checkpoint["scheduler_config"])

        return model
    else:
        raise Exception(f"Failed to load model: {response.text}")


if __name__ == "__main__":
    model = load_model_from_cloud()
    samples = model.generate_samples(num_samples=32)
    visualize_samples(samples)

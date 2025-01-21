import functions_framework
from google.cloud import storage
import torch
from pathlib import Path
import tempfile
import json
import base64
import io
from pixel_art_diffusion.model import PixelArtDiffusion
from diffusers import DDPMScheduler
from torchvision.transforms import ToPILImage

# Cloud Storage configuration
BUCKET_NAME = "pixel-art-bucket"
MODEL_FILE = "pixel_art_diffusion-final.pt"

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)

def download_model():
    """Download the model file from Cloud Storage to a temporary file"""
    temp_model_path = Path(tempfile.gettempdir()) / MODEL_FILE
    
    if not temp_model_path.exists():
        print(f"Downloading model to {temp_model_path}")
        blob.download_to_filename(str(temp_model_path))
    
    return temp_model_path

@functions_framework.http
def art_generate(request):
    """Generate pixel art samples and return as base64 encoded images"""
    # Set CORS headers for the preflight request
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
    }

    try:
        # Parse request parameters
        request_json = request.get_json(silent=True)
        num_samples = int(request_json.get('num_samples', 1)) if request_json else 1
        
        if num_samples < 1 or num_samples > 32:
            return (json.dumps({'error': 'num_samples must be between 1 and 32'}), 400, headers)

        # Set up device
        device = torch.device("cpu")
        
        # Initialize model
        model = PixelArtDiffusion(device=device.type)
        
        # Download and load model
        temp_model_path = download_model()
        checkpoint = torch.load(str(temp_model_path), map_location=device)
        model.model.load_state_dict(checkpoint["model_state_dict"])
        model.noise_scheduler = DDPMScheduler.from_config(checkpoint["scheduler_config"])

        # Generate samples
        print(f"Generating {num_samples} samples...")
        samples = model.generate_samples(num_samples=num_samples)
        
        # Convert tensors to base64 encoded images
        to_pil = ToPILImage()
        images_base64 = []
        
        for i in range(samples.shape[0]):
            img = to_pil(samples[i])
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            images_base64.append(img_str)

        return (json.dumps({'images': images_base64}), 200, headers)

    except Exception as e:
        print(f"Error: {str(e)}")
        return (json.dumps({'error': str(e)}), 500, headers)
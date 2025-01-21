import torch
import io
import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from http import HTTPStatus
from pydantic import BaseModel
from PIL import Image
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from diffusers import DDPMScheduler

from pixel_art_diffusion.model import PixelArtDiffusion # Model
from pixel_art_diffusion.generate_sample import generate_samples


"""
FastAPI for our model. 
To run the API file: uvicorn pixel_art_diffusion.api:app --reload --port 8000
    Launches a server: http://localhost:8000/
    Checkout these as well:
        http://localhost:8000/docs - Use the interactive docs 
        http://localhost:8000/redoc -  provides automatic API documentation using the ReDoc (Reference Documentation) format
        http://localhost:8000/predict - Make predictions by sending POST requests
        http://localhost:8000/openapi.json - JSON file of overall specifications of your program

"""

# Image grid for the generated pixel art images
def create_image_grid(samples, rows, cols):
    """Create a grid of images from generated samples"""
    # Convert from tensor to numpy arrays
    images = []
    for sample in samples:
        # Convert from [-1,1] to [0,255]
        img_array = ((sample.cpu().numpy() + 1) * 127.5).astype(np.uint8)
        img_array = img_array.transpose(1, 2, 0)  # CHW to HWH
        images.append(img_array)
    
    # Calculate image dimensions
    image_size = images[0].shape[0]
    
    # Create empty grid
    grid = np.zeros((rows * image_size, cols * image_size, 3), dtype=np.uint8)
    
    # Fill grid with images
    for idx, image in enumerate(images):
        i = idx // cols
        j = idx % cols
        grid[i * image_size:(i + 1) * image_size,
             j * image_size:(j + 1) * image_size] = image
    
    return Image.fromarray(grid)

# Global variables for model and components
model = None
device = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the ML model"""
    global model, device
    
    try:
        print("Loading model and components...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        model = PixelArtDiffusion(
            image_size=16,
            num_channels=3,
            num_train_timesteps=500,
            device=device
        )
        
        # Load the latest checkpoint
        models_dir = Path(__file__).parent.parent / "models"
        checkpoints = list(models_dir.glob("*.pt"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print(f"Loading checkpoint: {latest_checkpoint}")
            
            # Load checkpoint with proper device mapping
            checkpoint = torch.load(str(latest_checkpoint), 
                                 map_location=device)
            model.model.load_state_dict(checkpoint['model_state_dict'])
            model.noise_scheduler = DDPMScheduler.from_config(checkpoint['scheduler_config'])
            print("Checkpoint loaded successfully")
        else:
            print("No checkpoint found, using initialized model")
            
        print(f"Model loaded successfully. Using device: {device}")
        
        yield
    
    except Exception as e:
        print(f"Error during model loading: {str(e)}")
        raise
    finally:
        # Cleanup
        print("Cleaning up resources...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(
    title="Pixel Art Diffusion API",
    description="API for generating pixel art using diffusion models",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    ### Health check endpoint ###
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "model_status": "loaded and ready" if model is not None else "not loaded"
    }

@app.get("/model_info")
async def get_model_info():
    ### Get information about the model ###
    return {
        "model_name": "pixel_art_diffusion",
        "status": "loaded",
        "version": "1.0.0"
    }

@app.post("/generate/")
async def generate_art(num_samples: Optional[int] = 16, rows: Optional[int] = 2):
    """Generate a grid of pixel art images"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Calculate columns based on number of samples and rows
        cols = num_samples // rows
        
        # Generate samples
        samples = model.generate_samples(num_samples=num_samples)
        
        # Create grid image
        grid_image = create_image_grid(samples, rows, cols)
        
        # Save grid image
        output_path = "generated_grid.png"
        grid_image.save(output_path)
        
        return FileResponse(
            output_path,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=generated_grid.png"}
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )
        

# @app.get("/sample/{image_id}")
# async def get_generated_image(image_id: int):
#     """
#     Retrieve a generated image by ID
#     """
#     image_path = f"temp_generated_{image_id}.png"
#     if not os.path.exists(image_path):
#         raise HTTPException(status_code=404, detail="Image not found")
#     return FileResponse(image_path)


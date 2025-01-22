import torch
import tempfile
import os
import numpy 
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from http import HTTPStatus
from contextlib import asynccontextmanager
from typing import Optional
from io import BytesIO
from PIL import Image
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

from pixel_art_diffusion.cloud_function import load_model_from_cloud
from pixel_art_diffusion.visualize import visualize_samples



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

# Global variable to store the loaded model
model = None
path = Path(__file__).parent.parent.parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    ### Loads model on API startup ###
    global model
    try:
        model = load_model_from_cloud()
        print("Model from cloud is successfully loaded.")
        yield
    except Exception as e:
        print("Failed to load model from cloud.")
        raise RuntimeError(f"Startup (lifespan) failed: {e}")
    
app = FastAPI(
    title="Pixel Art Diffusion API",
    description="API for generating pixel art using diffusion models",
    version="1.0.0",
    lifespan=lifespan
)
app.mount("/metrics", make_asgi_app()) # Monitoring


@app.get("/")
async def root():
    ### Health check endpoint ###
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "model_status": "loaded" if model is not None else "not loaded"
    }

@app.get("/model_info")
async def get_model_info():
    ### Get information about the model ###
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_name": "Pixel Art Diffusion",
        "status": "loaded",
        "device": "CPU",
    }

@app.post("/generate")
async def generate_images(num_samples: Optional[int] = 16):
    ### Generate pixel art images using the generate_sample.py ###
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    try:
        samples = model.generate_samples(num_samples=num_samples)
        buffer = BytesIO() # Save the image to an in-memory buffer
        visualize_samples(samples, save_path=None)
        return StreamingResponse(buffer, media_type="image/png") # Return the image as a streaming response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from http import HTTPStatus
from contextlib import asynccontextmanager
from typing import Optional

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

app = FastAPI(
    title="Pixel Art Diffusion API",
    description="API for generating pixel art using diffusion models",
    version="1.0.0",
)


@app.get("/")
async def root():
    ### Health check endpoint ###
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }

@app.get("/model_info")
async def get_model_info():
    ### Get information about the model ###
    return {
        "model_name": "pixel_art_diffusion",
        "status": "loaded",
        "version": "1.0.0"
    }

@app.post("/generate")
async def generate_images(num_samples: Optional[int] = 32):
    ### Generate pixel art images using the generate_sample.py ###
    try:
        generate_samples(num_samples=num_samples)

        return FileResponse(
            media_type="image/png",
            filename="generated_samples.png"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )
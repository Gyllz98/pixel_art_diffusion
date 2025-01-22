from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from http import HTTPStatus
from contextlib import asynccontextmanager
from typing import Optional
from io import BytesIO
from PIL import Image
import logging
import traceback

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
    lifespan=lifespan,
)


@app.get("/")
async def root():
    ### Health check endpoint ###
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "model_status": "loaded" if model is not None else "not loaded",
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


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.post("/generate")
async def generate_images(num_samples: Optional[int] = 4):
    if model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        logger.info(f"Generating {num_samples} samples...")

        samples = model.generate_samples(num_samples=num_samples)
        logger.info(f"Samples generated. Shape: {samples.shape if samples is not None else None}")

        if samples is None:
            logger.error("Model returned None for samples")
            raise HTTPException(status_code=500, detail="Model failed to generate samples")

        logger.info("Visualizing samples...")
        grid_array = visualize_samples(samples, api=True)
        logger.info(f"Grid array shape: {grid_array.shape}")

        # Convert to PIL Image and then to bytes
        image = Image.fromarray(grid_array)

        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()

        # Use StreamingResponse for better memory handling with large images
        return StreamingResponse(BytesIO(img_byte_arr), media_type="image/png")

    except Exception as e:
        logger.error(f"Error in generate_images: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}\nTraceback: {traceback.format_exc()}")

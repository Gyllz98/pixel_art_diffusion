from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from http import HTTPStatus
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, AsyncGenerator
from io import BytesIO
from PIL import Image
import logging
import traceback
import uvicorn

from pixel_art_diffusion.cloud_function import load_model_from_cloud
from pixel_art_diffusion.visualize import visualize_samples

# Add type for the model
from pixel_art_diffusion.model import PixelArtDiffusion  # Import your model class

# Global variable with type annotation
model: Optional[PixelArtDiffusion] = None
path: Path = Path(__file__).parent.parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global model
    try:
        logging.info("Starting model loading from cloud...")
        model = load_model_from_cloud()
        logging.info("Model loaded successfully")
        yield
    except Exception as e:
        logging.error(f"Model loading failed: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise RuntimeError(f"Startup failed: {str(e)}")


app = FastAPI(
    title="Pixel Art Diffusion API",
    description="API for generating pixel art using diffusion models",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "model_status": "loaded" if model is not None else "not loaded",
    }


@app.get("/model_info")
async def get_model_info() -> Dict[str, str]:
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
async def generate_images(num_samples: Optional[int] = 4) -> StreamingResponse:
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

        # Create a new BytesIO object for saving the image
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)  # Reset the pointer to the start

        # Return the StreamingResponse directly with the BytesIO object
        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        logger.error(f"Error in generate_images: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}\nTraceback: {traceback.format_exc()}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

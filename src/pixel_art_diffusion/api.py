import os
import psutil
import time
import logging
import traceback
import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from http import HTTPStatus
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, AsyncGenerator
from io import BytesIO
from PIL import Image
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
)
from evidently.report import Report
import pickle

from pixel_art_diffusion.cloud_function import (
    load_model_from_cloud,
    get_samples_from_cloud,
    store_generated_samples,
    get_images_from_cloud,
)
from pixel_art_diffusion.visualize import visualize_samples
from pixel_art_diffusion.data_drift import create_generated_dataframe, create_image_dataframe

# Initialize Prometheus metrics
GENERATE_REQUESTS = Counter("pixel_art_generate_requests_total", "Total number of image generation requests:")

MEMORY_USAGE = Gauge("pixel_art_memory_usage_bytes", "Memory usage of the application:")

CURRENT_NUM_SAMPLES = Gauge("pixel_art_current_num_samples", "Number of samples generated in the last request:")

CURRENT_GENERATION_TIME = Gauge(
    "pixel_art_current_generation_time_seconds", "Time taken to generate the last batch of images (seconds):"
)

GENERATION_TIME = Histogram("pixel_art_generation_time_seconds", "Time taken to generate images (seconds):")

REFERENCE_DATA_PATH = "reference_data.pkl"

model = None
path = Path(__file__).parent.parent.parent


# Function to update memory usage
def update_memory_usage():
    process = psutil.Process(os.getpid())
    MEMORY_USAGE.set(process.memory_info().rss)


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

app.mount("/metrics", make_asgi_app())


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


@app.get("/samples")
async def get_stored_samples(limit: Optional[int] = 10):
    """Endpoint to retrieve stored samples from GCP bucket"""
    try:
        samples = get_samples_from_cloud()
        return samples
    except Exception as e:
        logger.error(f"Error retrieving samples: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve samples: {str(e)}")


@app.post("/generate")
async def generate_images(background_tasks: BackgroundTasks, num_samples: float = 4) -> StreamingResponse:
    if model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        GENERATE_REQUESTS.inc()
        update_memory_usage()
        start_time = time.time()
        logger.info(f"Generating {num_samples} samples...")

        samples = model.generate_samples(num_samples=num_samples)

        logger.info(f"Samples generated. Shape: {samples.shape if samples is not None else None}")

        if samples is None:
            logger.error("Model returned None for samples")
            raise HTTPException(status_code=500, detail="Model failed to generate samples")

        # Only store samples, remove drift analysis
        background_tasks.add_task(store_generated_samples, samples, num_samples)

        grid_array = visualize_samples(samples, api=True)

        generation_time = time.time() - start_time
        GENERATION_TIME.observe(generation_time)
        CURRENT_GENERATION_TIME.set(generation_time)
        CURRENT_NUM_SAMPLES.set(num_samples)
        update_memory_usage()

        logger.info(f"Grid array shape: {grid_array.shape}")

        image = Image.fromarray(grid_array)

        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)  # Reset the pointer to the start

        return StreamingResponse(BytesIO(img_byte_arr.getvalue()), media_type="image/png")

    except Exception as e:
        logger.error(f"Error in generate_images: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/data_drift")
async def get_drift_analysis():
    """Endpoint to perform and return drift analysis"""
    try:
        logger.info("Starting drift analysis")

        reports_dir = Path("drift_reports")
        reports_dir.mkdir(exist_ok=True)

        reference_data = load_or_create_reference_data()
        logger.info("Reference dataframe loaded")

        generated_images = get_images_from_cloud()
        logger.info("Got images from cloud - count: %d", len(generated_images))

        current_data = create_generated_dataframe(generated_images)
        logger.info("Loaded generated dataframe")

        drift_report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])

        drift_report.run(current_data=current_data, reference_data=reference_data)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"drift_report_{timestamp}.html"
        drift_report.save_html(str(report_path))

        logger.info("Drift analysis completed and saved: %s", report_path)

    except Exception as e:
        logger.error("Error in drift analysis: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Drift analysis failed: {str(e)}")


def load_or_create_reference_data():
    if os.path.exists(REFERENCE_DATA_PATH):
        with open(REFERENCE_DATA_PATH, "rb") as f:
            return pickle.load(f)

    reference_data = create_image_dataframe("data/processed/images/images")
    with open(REFERENCE_DATA_PATH, "wb") as f:
        pickle.dump(reference_data, f)
    return reference_data


# @app.get("/samples/{sample_id}")
# async def get_sample_image(sample_id: str):
#     """Endpoint to retrieve a specific sample image from GCP"""
#     try:
#         blobs = list(bucket.list_blobs(prefix="generated/"))
#         blob = next((b for b in blobs if f"_sample_{sample_id}." in b.name), None)

#         if blob is None:
#             raise HTTPException(status_code=404, detail="Sample not found")

#         # Download blob data
#         img_byte_arr = BytesIO()
#         blob.download_to_file(img_byte_arr)
#         img_byte_arr.seek(0)

#         return StreamingResponse(img_byte_arr, media_type="image/png")

#     except Exception as e:
#         logger.error(f"Error retrieving sample image: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to retrieve sample image: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

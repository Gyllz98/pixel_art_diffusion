from fastapi.testclient import TestClient
from src.pixel_art_diffusion.api import app


client = TestClient(app)  # Initialize TestClient


def test_read_root():
    ### Test the root endpoint ###
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "OK", "status-code": 200, "model_status": "loaded"}
        print(" API has started as intended.")


def test_load():
    ### Basic load test ###
    with TestClient(app) as client:
        for _ in range(10):  # Send 10 consecutive requests
            response = client.get("/")
            assert response.status_code == 200
    print(" API load test completed.")


def test_generate_default():
    ## Test image generation with default parameters ###
    with TestClient(app) as client:
        response = client.post("/generate")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        # Verify we got actual image data
        image_data = response.content
        assert len(image_data) > 0
    print(" API generated image correctly.")

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.pixel_art_diffusion.api import app

# Initialize TestClient
client = TestClient(app)

@pytest.fixture
def mock_generate_samples():
    ### Mock the generate_samples function from generate_sample.py. ### 
    with patch("src.pixel_art_diffusion.generate_sample.generate_samples") as mock_func:
        mock_func.return_value = None  # Simulate successful execution
        yield mock_func


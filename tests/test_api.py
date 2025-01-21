import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from pathlib import Path
from src.pixel_art_diffusion.api import app

# Initialize TestClient
client = TestClient(app)


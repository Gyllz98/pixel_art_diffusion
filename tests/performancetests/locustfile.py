import random
from locust import HttpUser, between, task

"""
locust -f tests/performancetests/locustfile.py \
    --headless --users 10 --spawn-rate 1 --run-time 1m --host %MYENDPOINT%
"""

class PixelArtUser(HttpUser):
    """
    A Locust user class that simulates users interacting with the Pixel Art Diffusion API.
    """
    # Wait between 1 to 3 seconds between tasks
    wait_time = between(1, 3)

    @task(3)  # Higher weight for health check endpoint
    def test_root(self):
        """Simulates users checking API health"""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(2)
    def test_model_info(self):
        """Simulates users checking model information"""
        with self.client.get("/model_info", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(1)  # Lower weight since this is more resource-intensive
    def test_generate(self):
        """Simulates users generating images"""
        # Randomly choose number of samples between 1 and 16
        num_samples = random.randint(1, 16)
        with self.client.post(
            f"/generate?num_samples={num_samples}", 
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
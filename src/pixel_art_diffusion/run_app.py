import subprocess
import time
import webbrowser
import sys
import signal
import logging
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def wait_for_server(url: str, max_retries: int = 5) -> bool:
    """Wait for server to be ready"""
    for i in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            logger.info(f"Waiting for server at {url} (attempt {i+1}/{max_retries})...")
            time.sleep(3)
    return False


def get_repo_root():
    """Get the root directory of the repository"""
    current_path = Path(__file__).resolve().parent.parent.parent
    return current_path


def run_app():
    try:
        repo_root = get_repo_root()
        frontend_path = repo_root / "src" / "pixel_art_diffusion" / "frontend.py"

        logger.info(f"Repository root: {repo_root}")
        logger.info(f"Frontend path: {frontend_path}")

        # Start the FastAPI backend
        logger.info("Starting backend server...")
        backend = subprocess.Popen(
            ["uvicorn", "pixel_art_diffusion.api:app", "--reload", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for backend to be ready
        if not wait_for_server("http://localhost:8000"):
            logger.error("Backend failed to start")
            backend.terminate()
            sys.exit(1)

        logger.info("Backend is ready!")

        # Start Streamlit frontend
        logger.info("Starting frontend...")
        frontend = subprocess.Popen(
            ["streamlit", "run", str(frontend_path), "--server.headless", "true"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for frontend to be ready
        if not wait_for_server("http://localhost:8501"):
            logger.error("Frontend failed to start")
            backend.terminate()
            frontend.terminate()
            sys.exit(1)

        logger.info("Frontend is ready!")
        webbrowser.open("http://localhost:8501")

        logger.info("Both servers are running!")
        logger.info("Frontend is at: http://localhost:8501")
        logger.info("Backend is at: http://localhost:8000")

        try:
            while True:
                # Check if either process has terminated
                if backend.poll() is not None:
                    logger.error("Backend server stopped unexpectedly")
                    break
                if frontend.poll() is not None:
                    logger.error("Frontend server stopped unexpectedly")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down servers...")
        finally:
            backend.terminate()
            frontend.terminate()
            backend.wait()
            frontend.wait()
            logger.info("Servers shut down successfully")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        if "backend" in locals():
            backend.terminate()
        if "frontend" in locals():
            frontend.terminate()
        sys.exit(1)


def signal_handler(sig, frame):
    logger.info("Received signal to shutdown...")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, signal_handler)
    run_app()


if __name__ == "__main__":
    main()

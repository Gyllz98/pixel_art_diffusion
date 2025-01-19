import os

# Define root paths
_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src", "pixel_art_diffusion") # root of source code

# Define data paths
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data
_PATH_DATA_RAW = os.path.join(_PATH_DATA, "raw")  # raw data directory
_PATH_DATA_PROCESSED = os.path.join(_PATH_DATA, "processed")  # processed data directory

# Define model paths
_PATH_MODELS = os.path.join(_PROJECT_ROOT, "models")  # trained models directory

# Additional paths that might be useful for testing
_PATH_CONFIGS = os.path.join(_PROJECT_ROOT, "configs")  # configuration files
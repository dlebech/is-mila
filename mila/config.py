"""Global configuration."""
import os

FLICKR_API_KEY = os.environ.get('FLICKR_API_KEY', '')
FLICKR_API_SECRET = os.environ.get('FLICKR_API_SECRET', '')
IMAGE_DIRECTORY = os.environ.get('IMAGE_DIR', './images')
OUTPUT_DIRECTORY = os.environ.get('OUTPUT_DIR', './output')

# Web
PORT = int(os.environ.get('PORT', 8000))
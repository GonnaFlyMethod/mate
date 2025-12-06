"""
Pytest configuration - loads environment variables from .env
"""
from pathlib import Path
from dotenv import load_dotenv

# Load .env file before any tests run
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

import os
from dotenv import load_dotenv
load_dotenv()

BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
API_KEY = os.getenv("NVIDIA_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")


def validate():
    missing = [k for k, v in {
        "NVIDIA_API_KEY": API_KEY,
        "TAVILY_API_KEY": TAVILY_API_KEY,
        "DATABASE_URL": DATABASE_URL,
    }.items() if not v]
    if missing:
        raise EnvironmentError(f"Missing env vars: {missing}. Copy .env.example to .env")

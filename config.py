import os
from dotenv import load_dotenv
load_dotenv()

BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
API_KEY = os.getenv("NVIDIA_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

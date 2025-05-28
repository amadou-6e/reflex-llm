from pathlib import Path

DEFAULT_CACHE = Path(Path.home(), ".cache", "localai")
MODEL_PATH = Path(DEFAULT_CACHE, "models")
DATA_PATH = Path(DEFAULT_CACHE, "compiled")
CONFIG_PATH = Path(DEFAULT_CACHE, "config")
WORK_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = Path(PACKAGE_DIR, "config")
DEFAULT_MODEL_MAPPINGS = {
    "gpt-3.5-turbo": "llama3.2:3b",
    "gpt-3.5-turbo-16k": "llama3.2:3b",
    "gpt-4": "llama3.1:8b",
    "gpt-4-turbo": "gemma3:4b",
    "gpt-4o": "gemma3:4b",
    "gpt-4o-mini": "gemma3:1b",
    "o1-preview": "phi3:reasoning",
    "o1-mini": "phi3:reasoning-mini",
    "o3-mini": "phi3:reasoning-mini",
    "o3": "phi3:reasoning",
    "o4-mini": "phi3:reasoning-mini",
    "text-embedding-ada-002": "nomic-embed-text",
    "text-embedding-3-small": "nomic-embed-text",
    "text-embedding-3-large": "mxbai-embed-large"
}

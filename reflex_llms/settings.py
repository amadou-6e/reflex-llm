from pathlib import Path

DEFAULT_CACHE = Path(Path.home(), ".cache", "localai")
MODEL_PATH = Path(DEFAULT_CACHE, "models")
DATA_PATH = Path(DEFAULT_CACHE, "compiled")
CONFIG_PATH = Path(DEFAULT_CACHE, "config")
WORK_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = Path(PACKAGE_DIR, "config")

import os
from pathlib import Path
from pathlib import Path
from IPython.display import Markdown, display
from dotenv import load_dotenv
from typing import List, Optional


def display_json(script_path):
    """
    Display JSON script content in a nicely formatted markdown code block.
    
    Args:
        script_path (Path or str): Path to the JSON script file
        
    Raises:
        FileNotFoundError: If the script file does not exist
    """
    script_path = Path(script_path)

    if not script_path.exists():
        raise FileNotFoundError(f"Script file {script_path} does not exist.")

    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Create markdown with JSON syntax highlighting
    markdown_content = f"```json\n{content}\n```"
    display(Markdown(markdown_content))


def load_and_verify_env(env_file: str = '.env', required_vars: Optional[List[str]] = None) -> bool:
    """
    Load environment variables and display .env file with blurred secrets if loading fails.
    
    Args:
        env_file: Path to the .env file
        required_vars: List of required environment variable names
    
    Returns:
        bool: True if successful, False otherwise
    """

    def find_env_file(filename: str) -> Optional[Path]:
        """Search for .env file in current and parent directories"""
        current_dir = Path.cwd()

        # Search up the directory tree
        for parent in [current_dir] + list(current_dir.parents):
            env_path = parent / filename
            if env_path.exists():
                return env_path
        return None

    def blur_secret(value: str) -> str:
        """Blur secret values"""
        if len(value) <= 4:
            return '*' * len(value)
        return value[:2] + '*' * (len(value) - 4) + value[-2:]

    def is_secret(key: str) -> bool:
        """Check if key likely contains secret"""
        secret_words = ['password', 'secret', 'key', 'token', 'auth', 'api']
        return any(word in key.lower() for word in secret_words)

    def display_env_file(env_file: str = '.env'):
        """Display .env file with blurred secrets"""
        env_file = find_env_file(env_file)
        if env_file is None:
            print(f"File '{env_file}' not found")
            return

        print(f"\nContents of '{env_file}':")
        print("-" * 40)

        with open(env_file, 'r') as f:
            for i, line in enumerate(f, 1):
                line = line.rstrip()

                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    if is_secret(key):
                        clean_value = value.strip('"\'')
                        blurred = blur_secret(clean_value)
                        if value.startswith(('"', "'")):
                            value = f"{value[0]}{blurred}{value[0]}"
                        else:
                            value = blurred
                        line = f"{key}={value}"

                print(f"{i:2d}: {line}")
        print("-" * 40)

    # Main execution
    print(f"Loading environment from '{env_file}'...")

    if not load_dotenv():
        print("Failed to load environment file")
        display_env_file()
        return False

    # Check required variables
    if required_vars:
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            print(f"Missing required variables: {', '.join(missing)}")
            display_env_file(env_file)
            return False

        print(f"Successfully loaded {len(required_vars)} required variables")
    else:
        print("Environment file loaded successfully")

    return True

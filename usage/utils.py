import os
import numpy as np
from pathlib import Path
from IPython.display import Markdown, display
from dotenv import load_dotenv
from typing import Any, Iterator, Union, List, Optional


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


def display_message(message: Union[str, Any], as_markdown: bool = True) -> None:
    """Display a message, extracting content from OpenAI responses if needed."""

    # Extract content
    if isinstance(message, str):
        content = message
    elif hasattr(message, 'choices') and message.choices:
        content = message.choices[0].message.content
    else:
        content = str(message)

    if as_markdown:
        display(Markdown(content))
    else:
        print(content)


def display_stream(stream: Iterator[Any]) -> str:
    """Display streaming response in real-time and return full content."""

    full_response = ""

    for chunk in stream:
        if hasattr(chunk, 'choices') and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content is not None:
                content = delta.content
                print(content, end="", flush=True)
                full_response += content

    return full_response


def display_embeddings(embeddings_response: Any, show_stats: bool = False) -> None:
    """Display basic embedding information."""

    if not hasattr(embeddings_response, 'data') or not embeddings_response.data:
        print("No embedding data found")
        return

    embeddings = [item.embedding for item in embeddings_response.data]
    model = getattr(embeddings_response, 'model', 'unknown')

    print(f"Model: {model}")
    print(f"Embeddings: {len(embeddings)}")
    print(f"Dimensions: {len(embeddings[0]) if embeddings else 0}")

    if show_stats and embeddings:
        arr = np.array(embeddings[0])
        print(f"Mean: {arr.mean():.6f}")
        print(f"Std: {arr.std():.6f}")
        print(f"Preview: {arr[:5].tolist()}")

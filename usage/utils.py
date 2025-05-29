from pathlib import Path
from IPython.display import Markdown, display


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

import json
import requests
from typing import List


class OllamaModelManager:
    """
    Manages Ollama models and OpenAI compatibility mappings.
    """

    def __init__(self, ollama_url: str = "http://127.0.0.1:11434"):
        """
        Initialize model manager.
        
        Args:
            ollama_url: Base URL for Ollama API
        """
        self.ollama_url = ollama_url

        # OpenAI model mappings to Ollama models
        self.model_mappings = {
            "gpt-3.5-turbo": "llama3.2:3b",
            "gpt-3.5-turbo-16k": "llama3.2:3b",
            "gpt-4": "llama3.1:8b",
            "gpt-4-turbo": "llama3.1:70b",
            "gpt-4o": "llama3.1:70b",
            "gpt-4o-mini": "llama3.2:3b",
            "text-embedding-ada-002": "nomic-embed-text",
            "text-embedding-3-small": "nomic-embed-text",
            "text-embedding-3-large": "mxbai-embed-large"
        }

    def _make_request(self, endpoint: str, method: str = "GET", data: dict = None) -> dict:
        url = f"{self.ollama_url}/api/{endpoint}"

        try:
            if method == "GET":
                response = requests.get(url, timeout=30)
            elif method == "POST":
                if endpoint == "pull":
                    # Handle streaming NDJSON response
                    response = requests.post(url, json=data, timeout=300, stream=True)
                    response.raise_for_status()

                    # Process each JSON line, return final status
                    final_result = {}
                    for line in response.iter_lines():
                        if line:
                            line_data = json.loads(line.decode('utf-8'))
                            final_result = line_data  # Keep last status

                    return final_result
                else:
                    response = requests.post(url, json=data, timeout=300)

            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama API request failed: {e}")

    def list_models(self) -> List[dict]:
        """List all available Ollama models."""
        response = self._make_request("tags")
        return response.get("models", [])

    def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama registry.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Pulling model: {model_name}")
            self._make_request("pull", "POST", {"name": model_name})
            print(f"Successfully pulled: {model_name}")
            return True
        except Exception as e:
            print(f"Failed to pull model {model_name}: {e}")
            return False

    def copy_model(self, source: str, destination: str) -> bool:
        """
        Copy/tag a model with a new name.
        
        Args:
            source: Source model name
            destination: Destination model name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Tagging model: {source} -> {destination}")
            self._make_request("copy", "POST", {"source": source, "destination": destination})
            print(f"Successfully tagged: {source} -> {destination}")
            return True
        except Exception as e:
            print(f"Failed to tag model {source} -> {destination}: {e}")
            return False

    def model_exists(self, model_name: str) -> bool:
        """Check if a model exists locally."""
        models = self.list_models()
        return any(model["name"].startswith(model_name) for model in models)

    def setup_openai_models(self) -> bool:
        """
        Set up OpenAI-compatible model mappings.
        
        Returns:
            True if all models set up successfully
        """
        print("Setting up OpenAI-compatible models...")
        success_count = 0

        for openai_name, ollama_name in self.model_mappings.items():
            print(f"\nProcessing: {openai_name} -> {ollama_name}")

            # Check if OpenAI-named model already exists
            if self.model_exists(openai_name):
                print(f"Model {openai_name} already exists, skipping...")
                success_count += 1
                continue

            # Pull the Ollama model if it doesn't exist
            if not self.model_exists(ollama_name):
                if not self.pull_model(ollama_name):
                    print(f"Failed to pull {ollama_name}, skipping {openai_name}")
                    continue

            # Tag with OpenAI name
            if self.copy_model(ollama_name, openai_name):
                success_count += 1

        total_models = len(self.model_mappings)
        print(f"\nModel setup complete: {success_count}/{total_models} models configured")
        return success_count == total_models

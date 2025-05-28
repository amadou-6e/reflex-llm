"""
Ollama Model Management Module

This module provides comprehensive management capabilities for Ollama models,
including model pulling, tagging, and OpenAI compatibility mapping. It handles
the complexities of model lifecycle management and provides seamless integration
with OpenAI-compatible applications.

The OllamaModelManager class facilitates automatic setup of OpenAI model mappings,
allowing applications designed for OpenAI's API to work with local Ollama models
without code changes.

Key Features
------------
- Model pulling from Ollama registry
- Model tagging and copying operations
- OpenAI-compatible model name mappings
- Comprehensive model existence checking
- Batch model setup operations
- Streaming download progress handling

Examples
--------
Basic usage for model management:

>>> from ollama_model_manager import OllamaModelManager
>>> manager = OllamaModelManager()
>>> manager.setup_openai_models()
>>> models = manager.list_models()

Custom Ollama URL:

>>> manager = OllamaModelManager(ollama_url="http://localhost:8080")
>>> manager.pull_model("llama3.2:3b")
>>> manager.copy_model("llama3.2:3b", "my-custom-model")

Dependencies
------------
- requests : HTTP client for API communications
- json : JSON parsing for API responses
- typing : Type annotations support
"""

import json
import requests
from typing import List, Dict, Any, Optional
from reflex_llms.settings import DEFAULT_MODEL_MAPPINGS


class OllamaModelManager:
    """
    Manages Ollama models and OpenAI compatibility mappings.
    
    This class provides a high-level interface for managing Ollama models,
    including pulling models from registries, creating model aliases, and
    setting up OpenAI-compatible model mappings for seamless integration
    with existing OpenAI-based applications.

    Parameters
    ----------
    ollama_url : str, default "http://127.0.0.1:11434"
        Base URL for the Ollama API endpoint

    Attributes
    ----------
    ollama_url : str
        Base URL for Ollama API communications
    model_mappings : dict[str, str]
        Dictionary mapping OpenAI model names to Ollama model names

    Examples
    --------
    Basic model management:

    >>> manager = OllamaModelManager()
    >>> manager.pull_model("llama3.2:3b")
    >>> models = manager.list_models()
    >>> manager.setup_openai_models()

    Custom configuration:

    >>> manager = OllamaModelManager(ollama_url="http://custom-host:8080")
    >>> success = manager.copy_model("source-model", "target-model")
    """

    def __init__(self, ollama_url: str = "http://127.0.0.1:11434") -> None:
        """
        Initialize the Ollama model manager.

        Parameters
        ----------
        ollama_url : str, default "http://127.0.0.1:11434"
            Base URL for Ollama API endpoint
        """
        self.ollama_url = ollama_url

        # OpenAI model mappings to Ollama models
        self.model_mappings: Dict[str, str] = DEFAULT_MODEL_MAPPINGS

    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make HTTP requests to the Ollama API.

        Parameters
        ----------
        endpoint : str
            API endpoint path (without /api/ prefix)
        method : str, default "GET"
            HTTP method to use ("GET" or "POST")
        data : dict or None, default None
            JSON data to send with POST requests

        Returns
        -------
        dict
            JSON response from the API

        Raises
        ------
        RuntimeError
            If the API request fails or returns an error status
        """
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
                    final_result: Dict[str, Any] = {}
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

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available Ollama models.

        Returns
        -------
        list of dict
            List of model information dictionaries, each containing
            model metadata such as name, size, and modification date

        Raises
        ------
        RuntimeError
            If the API request fails

        Examples
        --------
        >>> manager = OllamaModelManager()
        >>> models = manager.list_models()
        >>> for model in models:
        ...     print(f"Model: {model['name']}, Size: {model.get('size', 'Unknown')}")
        """
        response = self._make_request("tags")
        return response.get("models", [])

    def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from the Ollama registry.

        Downloads the specified model from the Ollama model registry and
        makes it available for local use. Handles streaming responses
        and provides progress feedback.

        Parameters
        ----------
        model_name : str
            Name of the model to pull (e.g., "llama3.2:3b")

        Returns
        -------
        bool
            True if the model was successfully pulled, False otherwise

        Examples
        --------
        >>> manager = OllamaModelManager()
        >>> success = manager.pull_model("llama3.2:3b")
        >>> if success:
        ...     print("Model pulled successfully")
        """
        try:
            print(f"Pulling model: {model_name}")
            result = self._make_request("pull", "POST", {"name": model_name})
            assert "error" not in result, f"Error pulling model: {result.get('error', 'Unknown error')}"
            print(f"Successfully pulled: {model_name}")
            return True
        except Exception as e:
            print(f"Failed to pull model {model_name}: {e}")
            return False

    def copy_model(self, source: str, destination: str) -> bool:
        """
        Copy or tag a model with a new name.

        Creates a new reference to an existing model without duplicating
        the model data. This is useful for creating aliases or OpenAI-compatible
        model names.

        Parameters
        ----------
        source : str
            Source model name (must exist locally)
        destination : str
            Destination model name (new alias)

        Returns
        -------
        bool
            True if the model was successfully copied/tagged, False otherwise

        Examples
        --------
        >>> manager = OllamaModelManager()
        >>> success = manager.copy_model("llama3.2:3b", "gpt-3.5-turbo")
        >>> if success:
        ...     print("Model tagged successfully")
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
        """
        Check if a model exists locally.

        Verifies whether a model with the specified name is available
        in the local Ollama installation.

        Parameters
        ----------
        model_name : str
            Name of the model to check

        Returns
        -------
        bool
            True if the model exists locally, False otherwise

        Examples
        --------
        >>> manager = OllamaModelManager()
        >>> if manager.model_exists("llama3.2:3b"):
        ...     print("Model is available")
        ... else:
        ...     print("Model needs to be pulled")
        """
        models = self.list_models()
        return any(model["name"].startswith(model_name) for model in models)

    def setup_openai_models(self) -> bool:
        """
        Set up OpenAI-compatible model mappings.

        Automatically pulls required Ollama models and creates OpenAI-compatible
        aliases for seamless integration with OpenAI-based applications. This
        method processes all models defined in the model_mappings dictionary.

        Returns
        -------
        bool
            True if all models were successfully set up, False if any failed

        Examples
        --------
        >>> manager = OllamaModelManager()
        >>> success = manager.setup_openai_models()
        >>> if success:
        ...     print("All OpenAI models configured successfully")
        ... else:
        ...     print("Some models failed to configure")

        Notes
        -----
        This method will:
        1. Check if OpenAI-named models already exist (skip if present)
        2. Pull missing Ollama models from the registry
        3. Create OpenAI-compatible aliases using model tagging
        4. Provide detailed progress feedback during setup
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

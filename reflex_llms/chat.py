# https://localai.io/basics/container/#usage
import os
import requests
import time
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI
# -- Ours --
from reflex_llms.settings import *
from reflex_llms.containers import ContainerHandler


class ChatClient:
    """
    A client that automatically chooses between Azure OpenAI and LocalAI
    based on available environment variables with model mapping and download management.
    """

    def __init__(
        self,
        local_ai_url: str = "http://127.0.0.1:8080/v1",
        local_ai_model: str = "gpt-4",
        auto_start_localai: bool = True,
        config_path: Optional[Path] = None,
        preload_models: Optional[List[Literal[
            "gpt-4",
            "gpt-4-0613",
            "gpt-4-0314",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4-turbo-preview",
            "gpt-4o",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-instruct",
            "gpt-4-reasoning",
            "gpt-4-code",
            "qwen-8b",
            "qwen-14b",
            "qwen-30b",
            "text-embedding-3-large",
            "text-embedding-3-small",
            "text-embedding-ada-002",
            "whisper-1",
            "tts-1",
            "tts-1-hd",
            "dall-e-3",
            "dall-e-2",
            "gpt-4-vision-preview",
            "gpt-4o-vision",
            "text-davinci-003",
            "text-davinci-002",
            "code-davinci-002",
        ]]] = [
            "gpt-4o", "gpt-4-turbo", "gpt-4o-mini", "gpt-3.5-turbo", "text-embedding-3-small",
            "whisper-1", "dall-e-3"
        ],
        wait_for_download: bool = False,
        host: str = "127.0.0.1",
        port: int = 8080,
        image: str = "localai/localai:latest-cpu",
        container_name: str = "methodsheet-localai",
        models_path: Optional[Path] = None,
        cache_path: Optional[Path] = None,
        startup_timeout: int = 120,
        rebuild: bool = False,
        preload: bool = False,
        n_threads: int = 4,
    ):
        """
        Initialize the adaptive client.
        
        Args:
            local_ai_url: URL for LocalAI instance
            local_ai_model: Default model name (should be OpenAI model name for mapping)
            auto_start_localai: Whether to automatically start LocalAI container
            config_path: Path to model configuration YAML file
            preload_models: List of OpenAI model names to preload on startup
            wait_for_download: If True, wait for model downloads; if False, return error when model not ready
        """
        # Load environment variables
        load_dotenv()

        # Azure OpenAI environment variables
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_deploy_name = os.getenv("AZURE_OPENAI_DEPLOY_NAME")
        self.azure_model_name = os.getenv("AZURE_OPENAI_MODEL_NAME")

        # LocalAI configuration
        self.local_ai_url = local_ai_url
        self.local_ai_model = local_ai_model
        self.auto_start_localai = auto_start_localai
        self.wait_for_download = wait_for_download

        # Model mapping from config
        self.model_mapping = {}

        # Container handler for LocalAI
        self.localai = None
        if not self._is_azure_configured() and auto_start_localai:
            # Extract host and port from URL
            url_parts = local_ai_url.replace("http://", "").replace("/v1", "").split(":")
            host = url_parts[0]
            port = int(url_parts[1]) if len(url_parts) > 1 else 8080

            self.localai = ContainerHandler(
                host=host,
                port=port,
                image=image,
                container_name=container_name,
                models_path=models_path,
                cache_path=cache_path,
                startup_timeout=startup_timeout,
                rebuild=rebuild,
                preload=preload,
                config_file=config_path,
                n_threads=n_threads,
            )
            self.config_path = self.localai.config_file
            self._load_model_config()
            try:
                self.localai.ensure_running()
            except Exception as e:
                print(f"Warning: Could not start LocalAI container: {e}")
                print("You may need to start LocalAI manually")

        # Initialize the appropriate client
        self.client = self._initialize_client()

        # Preload models if specified
        if preload_models and not self._is_azure_configured():
            self._preload_models(preload_models)

        print(f"Initialized {'Azure OpenAI' if self.is_azure else 'LocalAI'} client")

    @property
    def is_azure(self) -> bool:
        """Check if using Azure OpenAI."""
        return self._is_azure_configured()

    @property
    def model_name(self) -> str:
        """Get the appropriate model name based on the client type."""
        if self.is_azure:
            return self.azure_deploy_name
        else:
            return self.local_ai_model

    @property
    def client_info(self) -> Dict[str, Any]:
        """Get information about the current client configuration."""
        if self.is_azure:
            return {
                "type": "Azure OpenAI",
                "endpoint": self.azure_endpoint,
                "deployment": self.azure_deploy_name,
                "model": self.azure_model_name,
                "api_version": "2024-02-01"
            }
        else:
            return {"type": "LocalAI", "url": self.local_ai_url, "model": self.local_ai_model}

    def _preload_models(self, models: List[str]):
        """Preload specified models on startup."""
        print(f"Preloading {len(models)} models...")
        # https://localai.io/models/#how-to-install-a-model-from-the-repositories
        for model_name in models:
            try:
                job_id = self.download_model(model_name)
                if job_id:
                    print(f"Started download for {model_name}: {job_id}")
                    if self.wait_for_download:
                        self.wait_for_model_download(job_id, model_name)
                else:
                    print(f"Model {model_name} already available or skipped")
            except Exception as e:
                print(f"Failed to preload {model_name}: {e}")

    def _is_azure_configured(self) -> bool:
        """Check if all required Azure OpenAI environment variables are set."""
        required_vars = [
            self.azure_api_key, self.azure_endpoint, self.azure_deploy_name, self.azure_model_name
        ]
        return all(var is not None and var.strip() != "" for var in required_vars)

    def _initialize_client(self):
        """Initialize either Azure OpenAI or LocalAI client based on configuration."""
        if self._is_azure_configured():
            print("Azure OpenAI configuration detected")
            return AzureOpenAI(
                api_key=self.azure_api_key,
                azure_endpoint=self.azure_endpoint,
                api_version="2024-02-01",
            )
        else:
            print("Using LocalAI (Azure OpenAI config not found)")
            return OpenAI(base_url=self.local_ai_url, api_key="sk-local")

    def _load_model_config(self):
        """Load model configuration from YAML file."""
        if not self.config_path.exists():
            print(f"Warning: Config file not found at {self.config_path}")
            return

        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)

            if 'models' in config:
                for model in config['models']:
                    model_name = model.get('name')
                    if model_name:
                        self.model_mapping[model_name] = model

            print(f"Loaded {len(self.model_mapping)} model mappings from config")

        except Exception as e:
            print(f"Error loading model config: {e}")

    def _preload_models(self, models: List[str]):
        """Preload specified models on startup."""
        print(f"Preloading {len(models)} models...")

        for model_name in models:
            try:
                job_id = self.download_model(model_name)
                if job_id:
                    print(f"Started download for {model_name}: {job_id}")
                    if self.wait_for_download:
                        self.wait_for_model_download(job_id, model_name)
                else:
                    print(f"Model {model_name} already available or skipped")
            except Exception as e:
                print(f"Failed to preload {model_name}: {e}")

    @property
    def is_azure(self) -> bool:
        """Check if using Azure OpenAI."""
        return self._is_azure_configured()

    @property
    def model_name(self) -> str:
        """Get the appropriate model name based on the client type."""
        if self.is_azure:
            return self.azure_deploy_name
        else:
            return self.local_ai_model

    @property
    def client_info(self) -> Dict[str, Any]:
        """Get information about the current client configuration."""
        if self.is_azure:
            return {
                "type": "Azure OpenAI",
                "endpoint": self.azure_endpoint,
                "deployment": self.azure_deploy_name,
                "model": self.azure_model_name,
                "api_version": "2024-02-01"
            }
        else:
            return {
                "type": "LocalAI",
                "url": self.local_ai_url,
                "model": self.local_ai_model,
                "available_models": list(self.model_mapping.keys())
            }

    def _is_azure_configured(self) -> bool:
        """Check if all required Azure OpenAI environment variables are set."""
        required_vars = [
            self.azure_api_key, self.azure_endpoint, self.azure_deploy_name, self.azure_model_name
        ]
        return all(var is not None and var.strip() != "" for var in required_vars)

    def _initialize_client(self):
        """Initialize either Azure OpenAI or LocalAI client based on configuration."""
        if self._is_azure_configured():
            print("Azure OpenAI configuration detected")
            return AzureOpenAI(
                api_key=self.azure_api_key,
                azure_endpoint=self.azure_endpoint,
                api_version="2024-02-01",
            )
        else:
            print("Using LocalAI (Azure OpenAI config not found)")
            return OpenAI(base_url=self.local_ai_url, api_key="sk-local")

    def get_model_config(self, openai_model_name: str) -> Optional[Dict[str, Any]]:
        """Get LocalAI model configuration for an OpenAI model name."""
        return self.model_mapping.get(openai_model_name)

    def download_model(self, openai_model_name: str) -> Optional[str]:
        """
        Download a model by OpenAI name.
        
        Args:
            openai_model_name: OpenAI model name (e.g., "gpt-4")
            
        Returns:
            Job ID for tracking download progress, or None if no download needed
        """
        if self._is_azure_configured():
            print(f"Azure OpenAI configured, no download needed for {openai_model_name}")
            return None

        model_config = self.get_model_config(openai_model_name)
        if not model_config:
            raise ValueError(f"No configuration found for model: {openai_model_name}")

        # Check if model already exists
        available_models = self.list_available_models()
        if openai_model_name in available_models:
            print(f"Model {openai_model_name} already available")
            return None

        # Prepare download request
        download_url = model_config.get('urls', [])
        if not download_url:
            raise ValueError(f"No download URL found for model: {openai_model_name}")

        # Use first URL for download
        url = download_url[0] if isinstance(download_url, list) else download_url
        file_name = model_config.get('file', f"{openai_model_name}.gguf")

        api_url = self.local_ai_url.replace('/v1', '/models/apply')

        payload = {"url": url, "name": openai_model_name, "file": file_name}

        try:
            response = requests.post(api_url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            if 'uuid' in result:
                job_id = result['uuid']
                print(f"Started download for {openai_model_name}: {job_id}")
                return job_id
            else:
                print(f"Unexpected response format: {result}")
                return None

        except requests.RequestException as e:
            print(f"Failed to start download for {openai_model_name}: {e}")
            raise

    def check_download_status(self, job_id: str) -> Dict[str, Any]:
        """
        Check the status of a model download.
        
        Args:
            job_id: Job ID returned from download_model
            
        Returns:
            Dictionary with download status information
        """
        if self._is_azure_configured():
            return {"status": "not_applicable", "message": "Using Azure OpenAI"}

        api_url = self.local_ai_url.replace('/v1', f'/models/jobs/{job_id}')

        try:
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            return {"status": "error", "message": str(e)}

    def wait_for_model_download(self, job_id: str, model_name: str, timeout: int = 1800) -> bool:
        """
        Wait for a model download to complete.
        
        Args:
            job_id: Job ID to monitor
            model_name: Model name for logging
            timeout: Maximum time to wait in seconds (default: 30 minutes)
            
        Returns:
            True if download completed successfully, False otherwise
        """
        start_time = time.time()
        last_progress = -1

        print(f"Waiting for {model_name} download to complete...")

        while time.time() - start_time < timeout:
            status = self.check_download_status(job_id)

            if status.get('error'):
                print(f"Download failed: {status['error']}")
                return False

            if status.get('processed', False):
                print(f"{model_name} download completed successfully")
                return True

            # Show progress if available
            progress = status.get('progress', 0)
            if progress != last_progress and progress > 0:
                downloaded_size = status.get('downloaded_size', 'Unknown')
                file_size = status.get('file_size', 'Unknown')
                print(f"Progress: {progress:.1f}% ({downloaded_size}/{file_size})")
                last_progress = progress

            time.sleep(5)  # Check every 5 seconds

        print(f"Timeout waiting for {model_name} download")
        return False

    def list_available_models(self) -> List[str]:
        """Get list of available models from LocalAI."""
        if self._is_azure_configured():
            return [self.azure_deploy_name]

        try:
            response = requests.get(f"{self.local_ai_url}/models", timeout=10)
            response.raise_for_status()
            result = response.json()

            return [model['id'] for model in result.get('data', [])]

        except requests.RequestException as e:
            print(f"Failed to get available models: {e}")
            return []

    def completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        content_only: bool = False,
        **kwargs,
    ) -> Any:
        """
        Create a chat completion using the configured client.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model name to use (defaults to configured model)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            content_only: If True, return only the content string
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Chat completion response or content string if content_only=True
        """
        model_to_use = model or self.model_name

        # Handle model download for LocalAI
        if not self._is_azure_configured():
            available_models = self.list_available_models()

            if model_to_use not in available_models:
                print(f"Model {model_to_use} not available, attempting download...")

                try:
                    job_id = self.download_model(model_to_use)

                    if job_id:
                        if self.wait_for_download:
                            success = self.wait_for_model_download(job_id, model_to_use)
                            if not success:
                                raise RuntimeError(f"Failed to download model {model_to_use}")
                        else:
                            raise RuntimeError(
                                f"Model {model_to_use} is downloading (Job ID: {job_id}). "
                                f"Please wait or set wait_for_download=True")

                except Exception as e:
                    raise RuntimeError(f"Cannot use model {model_to_use}: {e}")

        try:
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            if content_only:
                return response.choices[0].message.content
            return response

        except Exception as e:
            print(f"Error during chat completion: {e}")
            if not self.is_azure:
                print("Make sure LocalAI is running at", self.local_ai_url)
            raise

    def prompt(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a simple text prompt and get the response.
        
        Args:
            prompt: The text prompt to send
            model: Model name to use (defaults to configured model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            The response content as a string
        """
        messages = [{"role": "user", "content": prompt}]
        return self.completion(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            content_only=True,
        )


# Example usage
if __name__ == "__main__":
    # Initialize the adaptive client
    client = ChatClient()

    # Print client information
    print(f"\nClient Info:")
    for key, value in client.client_info.items():
        print(f"   {key}: {value}")

    # Example simple prompt
    try:
        response = client.prompt("Hello! Can you tell me what AI system you are?")
        print(f"\nResponse: {response}")
    except Exception as e:
        print(f"Failed to get response: {e}")

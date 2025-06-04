# RefLex LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://www.docker.com/)

**Intelligent OpenAI API fallback system with automatic local AI deployment**

RefLex LLMs provides seamless failover between OpenAI, Azure OpenAI, and local AI models when endpoints become unavailable. Perfect for development, testing, CI/CD pipelines, and production deployments that require high availability and cost optimization.

## Quick Start

```bash
pip install reflex-llms
```

```python
from reflex_llms import get_openai_client

# Drop-in replacement for OpenAI client with automatic failover
client = get_openai_client()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

That's it! RefLex automatically:
- Detects available providers (OpenAI, Azure, local)
- Starts local AI containers if needed
- Maps OpenAI models to local equivalents
- Handles failover transparently

## Key Features

### **Automatic Provider Selection**
Intelligently chooses between OpenAI, Azure OpenAI, and local Ollama based on availability and preferences

### **Zero-Config Docker Integration**  
Automatically manages local AI containers with Ollama - no manual setup required

### **Perfect OpenAI Compatibility**
Drop-in replacement for `openai` Python client - change one import, get automatic failover

### **Intelligent Model Mapping**
Automatically maps OpenAI models (`gpt-3.5-turbo`) to equivalent local models (`llama3.2:3b`)

### **Flexible Configuration**
File-based configuration with environment overrides for different deployment scenarios

### **Health Monitoring**
Continuous provider health checking with automatic recovery and performance optimization

## Installation

### Prerequisites
- Python 3.8 or higher
- Docker (for local AI capabilities)
- 4GB+ RAM (recommended for local models)

### Install RefLex
```bash
pip install reflex-llms
```

### Optional: Configure providers
```bash
# For OpenAI (if you have an API key)
export OPENAI_API_KEY="your-openai-key"

# For Azure OpenAI (if using Azure)
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-azure-key"
```

## Usage Examples

### Basic Chat Completion
```python
from reflex_llms import get_openai_client

client = get_openai_client()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    max_tokens=150
)

print(response.choices[0].message.content)
```

### Streaming Responses
```python
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a story about AI"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Embeddings
```python
response = client.embeddings.create(
    model="text-embedding-ada-002",
    input="RefLex provides intelligent AI failover capabilities"
)

embeddings = response.data[0].embedding
print(f"Generated {len(embeddings)}-dimensional embedding")
```

### Custom Provider Preferences
```python
# Prefer local AI, fallback to OpenAI
client = get_openai_client(
    preference_order=["reflex", "openai"]
)

# Custom timeouts and endpoints
client = get_openai_client(
    preference_order=["openai", "reflex"],
    timeout=10.0,
    openai_base_url="https://custom-endpoint.com/v1"
)
```

## Configuration

### Configuration File (reflex.json)
Create a `reflex.json` file for persistent configuration:

```json
{
    "preference_order": ["openai", "azure", "reflex"],
    "timeout": 120.0,
    "openai_base_url": "https://api.openai.com/v1",
    "azure_api_version": "2024-02-15-preview",
    "reflex_server": {
        "host": "127.0.0.1",
        "port": 11434,
        "auto_setup": true,
        "model_mappings": {
            "minimal_setup": true,
            "model_mapping": {
                "gpt-3.5-turbo": "llama3.2:3b",
                "gpt-4": "llama3.1:8b",
                "gpt-4o": "gemma3:4b",
                "gpt-4o-mini": "gemma3:1b",
                "text-embedding-ada-002": "nomic-embed-text"
            }
        }
    }
}
```

### Load from Configuration File
```python
client = get_openai_client(from_file=True)
```

## Provider Management

### Check Current Provider
```python
from reflex_llms import get_selected_provider, is_using_reflex

print(f"Using provider: {get_selected_provider()}")
print(f"Using local AI: {is_using_reflex()}")
```

### System Status
```python
from reflex_llms import get_module_status

status = get_module_status()
print(f"Selected provider: {status['selected_provider']}")
print(f"RefLex server running: {status['reflex_server_running']}")
print(f"Configuration cached: {status['has_cached_config']}")
```

### Server Management (RefLex Provider)
```python
from reflex_llms import get_reflex_server

server = get_reflex_server()
if server:
    print(f"API URL: {server.openai_compatible_url}")
    print(f"Health status: {server.is_healthy}")
    
    # Get detailed status
    status = server.get_status()
    print(f"Available models: {len(status['openai_compatible_models'])}")
```

### Cache Management
```python
from reflex_llms import clear_cache, stop_reflex_server

# Force provider re-detection
clear_cache()

# Clean shutdown of local server
stop_reflex_server()
```

## Advanced Usage

### Custom Model Mappings
```python
from reflex_llms.server import ReflexServer, ModelMapping

# Configure custom model mappings
custom_mappings = ModelMapping(
    minimal_setup=False,
    model_mapping={
        "gpt-3.5-turbo": "llama3.2:3b",
        "gpt-4": "llama3.1:8b",
        "custom-model": "phi3:mini"
    }
)

server = ReflexServer(
    port=8080,
    model_mappings=custom_mappings
)
```

### Environment-Specific Configuration
```python
import os

# Production: prefer cloud providers
if os.getenv('ENVIRONMENT') == 'production':
    client = get_openai_client(["openai", "azure"])
    
# Development: prefer local AI
elif os.getenv('ENVIRONMENT') == 'development':
    client = get_openai_client(["reflex", "openai"])
    
# CI/CD: local only
else:
    client = get_openai_client(["reflex"])
```

## Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

# Install Docker in container for RefLex
RUN apt-get update && apt-get install -y docker.io

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "app.py"]
```

### Kubernetes Configuration
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: reflex-config
data:
  reflex.json: |
    {
      "preference_order": ["openai", "reflex"],
      "timeout": 30.0,
      "reflex_server": {
        "minimal_setup": true
      }
    }
```

### Health Checks
```python
from reflex_llms import get_module_status

def health_check():
    status = get_module_status()
    return {
        "healthy": status["selected_provider"] is not None,
        "provider": status["selected_provider"],
        "reflex_running": status["reflex_server_running"]
    }
```

## Model Mapping Reference

| OpenAI Model | Local Equivalent | Use Case |
|--------------|------------------|----------|
| `gpt-3.5-turbo` | `llama3.2:3b` | General chat, fast responses |
| `gpt-4` | `llama3.1:8b` | Complex reasoning, high quality |
| `gpt-4o` | `gemma3:4b` | Latest capabilities |
| `gpt-4o-mini` | `gemma3:1b` | Lightweight, fast |
| `o1-preview` | `phi3:reasoning` | Mathematical reasoning |
| `text-embedding-ada-002` | `nomic-embed-text` | Text embeddings |

## Testing

### Run Tests
```bash
# Install development dependencies
pip install -e .[dev]

# Run all tests
pytest

# Run integration tests (requires Docker)
pytest -m integration

# Run with coverage
pytest --cov=reflex_llms
```

### Test Failover Behavior
```python
from reflex_llms import get_openai_client, clear_cache

# Test with invalid OpenAI endpoint
client = get_openai_client(
    openai_base_url="https://invalid-endpoint.com/v1",
    preference_order=["openai", "reflex"]
)

# Should automatically fallback to RefLex
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Test failover"}]
)
```

## Troubleshooting

### Common Issues

**Docker not running**
```bash
# Start Docker daemon
sudo systemctl start docker

# Or on macOS/Windows, start Docker Desktop
```

**Port conflicts**
```python
# Use custom port
client = get_openai_client(
    reflex_server_config={"port": 8080}
)
```

**Model download issues**
```python
# Check server status
from reflex_llms import get_reflex_server

server = get_reflex_server()
if server:
    status = server.get_status()
    print(f"Setup complete: {status['setup_complete']}")
    print(f"Available models: {status['total_models']}")
```

**Clear stuck state**
```python
from reflex_llms import clear_cache, stop_reflex_server

# Reset everything
stop_reflex_server()
clear_cache()

# Restart
client = get_openai_client()
```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose output
client = get_openai_client(force_recheck=True)
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/amadou-6e/reflex-llms.git
cd reflex-llms
pip install -e .[dev]
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on [Ollama](https://ollama.ai/) for local AI capabilities
- Compatible with [OpenAI API](https://openai.com/api/)
- Uses [Docker](https://docker.com/) for containerization
- Supports [Azure OpenAI Service](https://azure.microsoft.com/en-us/products/cognitive-services/openai-service/)

## Support

- Documentation: [docs/](docs/)
- Issue Tracker: [GitHub Issues](https://github.com/amadou-6e/reflex-llms/issues)
- Discussions: [GitHub Discussions](https://github.com/amadou-6e/reflex-llms/discussions)

---

**RefLex LLMs**: Making AI applications resilient, cost-effective, and deployment-ready!
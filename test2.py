#!/usr/bin/env python3
"""
Minimal script to debug Ollama pull JSON parsing issue.
"""

import requests
import json
import sys
from pathlib import Path

# Add your project to path if needed
# sys.path.append(str(Path(__file__).parent.parent))

from reflex_llms.containers import ContainerHandler
from reflex_llms.models import OllamaModelManager


def debug_raw_request(ollama_url: str, model_name: str):
    """Debug the raw HTTP request to Ollama pull API."""
    print("=" * 60)
    print("DEBUGGING RAW HTTP REQUEST")
    print("=" * 60)

    url = f"{ollama_url}/api/pull"
    data = {"name": model_name}

    print(f"URL: {url}")
    print(f"Data: {data}")

    try:
        print("\n--- Making Request ---")
        response = requests.post(url, json=data, timeout=60, stream=False)

        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Raw Content Length: {len(response.content)}")

        print("\n--- Raw Response Content ---")
        raw_content = response.content.decode('utf-8', errors='replace')
        print(repr(raw_content))  # Show exact bytes including hidden characters

        print("\n--- Attempting JSON Parse ---")
        try:
            json_response = response.json()
            print("JSON parsed successfully:")
            print(json.dumps(json_response, indent=2))
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            print(f"Error at position: {e.pos}")
            print(f"Error around: {repr(raw_content[max(0, e.pos-10):e.pos+10])}")

            # Try to find where the issue is
            lines = raw_content.split('\n')
            print(f"\nContent split by lines ({len(lines)} lines):")
            for i, line in enumerate(lines[:5]):  # Show first 5 lines
                print(f"Line {i+1}: {repr(line)}")

            return False

    except Exception as e:
        print(f"Request failed: {e}")
        return False

    return True


def debug_streaming_request(ollama_url: str, model_name: str):
    """Debug streaming request which might be the expected format."""
    print("=" * 60)
    print("DEBUGGING STREAMING REQUEST")
    print("=" * 60)

    url = f"{ollama_url}/api/pull"
    data = {"name": model_name}

    try:
        print(f"Making streaming request to: {url}")
        response = requests.post(url, json=data, timeout=60, stream=True)

        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")

        print("\n--- Streaming Response ---")
        line_count = 0
        for line in response.iter_lines(decode_unicode=True):
            line_count += 1
            if line:
                print(f"Line {line_count}: {repr(line)}")
                try:
                    json_data = json.loads(line)
                    print(f"  Parsed JSON: {json_data}")
                except json.JSONDecodeError as e:
                    print(f"  JSON Error: {e}")

                if line_count > 10:  # Limit output
                    print("  ... (truncated)")
                    break

        return True

    except Exception as e:
        print(f"Streaming request failed: {e}")
        return False


def debug_model_manager_implementation():
    """Debug the current model manager implementation."""
    print("=" * 60)
    print("DEBUGGING MODEL MANAGER IMPLEMENTATION")
    print("=" * 60)

    # Start container
    container = ContainerHandler(port=11435, container_name="debug-ollama-pull")

    try:
        print("Starting Ollama container...")
        container.ensure_running()

        manager = OllamaModelManager(ollama_url=container.api_url)

        # Test basic connectivity
        print(f"Testing connectivity to: {container.api_url}")
        models = manager.list_models()
        print(f"Current models: {len(models)}")

        # Debug the problematic method
        test_model = "smollm:135m"
        print(f"\nDebugging pull_model() for: {test_model}")

        # Let's look at what _make_request actually does
        print("\n--- Inspecting _make_request method ---")
        try:
            # First check what the raw response looks like
            debug_raw_request(container.api_url, test_model)
            debug_streaming_request(container.api_url, test_model)

        except Exception as e:
            print(f"Debug failed: {e}")

    finally:
        print("\nCleaning up container...")
        container.stop()


def test_minimal_reproduction():
    """Minimal reproduction of the exact error."""
    print("=" * 60)
    print("MINIMAL REPRODUCTION TEST")
    print("=" * 60)

    container = ContainerHandler(port=11436, container_name="minimal-debug-ollama")

    try:
        container.ensure_running()

        # Exactly replicate what's failing
        manager = OllamaModelManager(ollama_url=container.api_url)

        print("Testing the exact failing scenario...")
        result = manager.pull_model("smollm:135m")
        print(f"Result: {result}")

    except Exception as e:
        print(f"Error reproduced: {e}")
        import traceback
        traceback.print_exc()
    finally:
        container.stop()


if __name__ == "__main__":
    print("Ollama Pull Debug Script")
    print("This will help diagnose the JSON parsing issue")

    if len(sys.argv) > 1:
        if sys.argv[1] == "raw":
            # Test with existing Ollama instance
            debug_raw_request("http://127.0.0.1:11434", "smollm:135m")
        elif sys.argv[1] == "stream":
            debug_streaming_request("http://127.0.0.1:11434", "smollm:135m")
        elif sys.argv[1] == "minimal":
            test_minimal_reproduction()
        else:
            print("Usage: python debug_script.py [raw|stream|minimal]")
    else:
        # Full debug session
        debug_model_manager_implementation()

#!/usr/bin/env python3
"""
Minimal LocalAI test script - no classes, just sequential steps.
Designed for debugging CPU underutilization issues.
"""

import docker
import requests
import time
import json
import psutil
import os
from pathlib import Path

# Configuration
HOST = "localhost"
PORT = 8080
IMAGE = "localai/localai:latest-cpu"  # or try latest-aio-cpu for preloaded models
CONTAINER_NAME = "test-localai"

# Get optimal thread count (physical cores)
PHYSICAL_CORES = psutil.cpu_count(logical=False)
LOGICAL_CORES = psutil.cpu_count(logical=True)
print(f"System info: {PHYSICAL_CORES} physical cores, {LOGICAL_CORES} logical cores")

# Use physical cores for better performance
THREADS = PHYSICAL_CORES or 4
MEMORY_LIMIT = "4g"

# Model to test with (small and fast)
MODEL_ID = "localai@gemma-3-1b-it"
MODEL_NAME = "gemma-3-1b-it"

# Create directories
MODELS_DIR = Path("./test_models")
CACHE_DIR = Path("./test_cache")
MODELS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

print(f"Using {THREADS} threads, {MEMORY_LIMIT} memory limit")
print(f"Models dir: {MODELS_DIR.absolute()}")
print(f"Cache dir: {CACHE_DIR.absolute()}")


def check_docker():
    """Step 1: Check if Docker is running"""
    print("\n=== STEP 1: Checking Docker ===")
    try:
        client = docker.from_env()
        client.ping()
        print(" Docker is running")
        return client
    except Exception as e:
        print(f"Docker error: {e}")
        print("Make sure Docker Desktop is running")
        exit(1)


def cleanup_container(client):
    """Step 2: Clean up any existing container"""
    print("\n=== STEP 2: Cleaning up existing container ===")
    try:
        existing = client.containers.get(CONTAINER_NAME)
        print(f"Found existing container: {existing.status}")
        existing.remove(force=True)
        print(" Removed existing container")
    except docker.errors.NotFound:
        print(" No existing container to remove")


def pull_image(client):
    """Step 3: Ensure image is available"""
    print("\n=== STEP 3: Checking/pulling image ===")
    try:
        image = client.images.get(IMAGE)
        print(f"Image {IMAGE} already available")
        return image
    except docker.errors.NotFound:
        print(f"Pulling {IMAGE}...")
        image = client.images.pull(IMAGE)
        print(" Image pulled successfully")
        return image


def create_container(client):
    """Step 4: Create optimized container"""
    print("\n=== STEP 4: Creating container ===")

    # Mount points
    mounts = [
        docker.types.Mount(target="/build/models", source=str(MODELS_DIR.absolute()), type="bind"),
        docker.types.Mount(target="/tmp/generated", source=str(CACHE_DIR.absolute()), type="bind")
    ]

    # Optimized environment variables for CPU performance
    env_vars = {
        # Core settings
        "LOCALAI_THREADS": str(THREADS),
        "LOCALAI_CONTEXT_SIZE": "2048",

        # Performance optimizations
        "LOCALAI_PARALLEL_REQUESTS": "true",
        "LOCALAI_F16": "false",  # Enable F16 for better performance
        "LOCALAI_SINGLE_ACTIVE_BACKEND": "false",  # Allow multiple backends

        # CPU-specific optimizations
        "OMP_NUM_THREADS": str(THREADS),  # OpenMP threads
        "GOMAXPROCS": str(THREADS),  # Go runtime threads
        "MKL_NUM_THREADS": str(THREADS),  # Intel MKL threads

        # Disable debug for performance
        "DEBUG": "true",
        "LOCALAI_LOG_LEVEL": "info",

        # Memory optimizations
        "LOCALAI_WATCHDOG_IDLE": "true",
        "LOCALAI_WATCHDOG_IDLE_TIMEOUT": "10m",

        # Backend-specific optimizations
        "REBUILD": "true",  # Don't rebuild, use precompiled
        "LLAMACPP_PARALLEL": str(max(1, THREADS // 2)),  # Parallel llama.cpp workers
    }

    ports = {"8080/tcp": PORT}

    container = client.containers.create(
        image=IMAGE,
        name=CONTAINER_NAME,
        mounts=mounts,
        ports=ports,
        environment=env_vars,
        detach=True,
        # Resource limits for Windows
        mem_limit=MEMORY_LIMIT,
        # Allow container to use all CPU cores but don't overwhelm system
        cpu_period=100000,
        cpu_quota=int(100000 * THREADS * 0.9),  # Use 90% of available CPU
        # Remove default CPU shares limit
        cpu_shares=1024 * THREADS,
    )

    print(f"Container created with optimizations:")
    print(f" - {THREADS} threads")
    print(f" - {MEMORY_LIMIT} memory limit")
    print(f" - CPU quota: {int(100000 * THREADS * 0.9)} (90% of {THREADS} cores)")
    print(f" - Parallel requests enabled")

    return container


def start_container(container):
    """Step 5: Start container"""
    print("\n=== STEP 5: Starting container ===")
    container.start()
    print(" Container started")

    # Show initial logs
    time.sleep(2)
    logs = container.logs(tail=10).decode('utf-8', errors='ignore')
    print("Initial container logs:")
    print("---")
    print(logs)
    print("---")


def wait_for_ready():
    """Step 6: Wait for LocalAI to be ready"""
    print("\n=== STEP 6: Waiting for LocalAI to be ready ===")
    start_time = time.time()
    timeout = 600  # 5 minutes

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://{HOST}:{PORT}/readyz", timeout=10)
            if response.status_code == 200:
                elapsed = time.time() - start_time
                print(f"LocalAI ready in {elapsed:.1f} seconds")
                return True
        except requests.RequestException:
            pass

        elapsed = time.time() - start_time
        print(f"Waiting... ({elapsed:.1f}s)")
        time.sleep(5)

    print(" LocalAI not ready within timeout")
    return False


def show_system_status():
    """Show system resource usage"""
    print("\n=== SYSTEM STATUS ===")
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    memory = psutil.virtual_memory()

    print(f"CPU usage per core: {[f'{x:.1f}%' for x in cpu_percent]}")
    print(f"Average CPU: {sum(cpu_percent)/len(cpu_percent):.1f}%")
    print(
        f"Memory: {memory.percent:.1f}% ({memory.used//1024//1024}MB/{memory.total//1024//1024}MB)")


def check_models():
    """Step 7: Check what models are available"""
    print("\n=== STEP 7: Checking available models ===")

    # Check loaded models
    try:
        response = requests.get(f"http://{HOST}:{PORT}/v1/models", timeout=10)
        if response.status_code == 200:
            models = response.json().get('data', [])
            print(f"Currently loaded models: {len(models)}")
            for model in models[:5]:  # Show first 5
                print(f" - {model.get('id', 'unknown')}")
        else:
            print(f"Failed to get models: {response.status_code}")
    except Exception as e:
        print(f"Error checking models: {e}")

    # Check available models for download
    try:
        response = requests.get(f"http://{HOST}:{PORT}/models/available", timeout=30)
        if response.status_code == 200:
            available = response.json()
            print(f"Available for download: {len(available)} models")
            # Find our target model
            target_found = any(MODEL_ID.split('@')[1] in str(model) for model in available)
            print(f"Target model '{MODEL_ID}' available: {target_found}")
        else:
            print(f"Failed to get available models: {response.status_code}")
    except Exception as e:
        print(f"Error checking available models: {e}")


def install_model():
    """Step 8: Install model"""
    print(f"\n=== STEP 8: Installing model {MODEL_ID} ===")

    # Check if already loaded
    try:
        response = requests.get(f"http://{HOST}:{PORT}/v1/models", timeout=10)
        if response.status_code == 200:
            models = response.json().get('data', [])
            if any(model.get('id') == MODEL_NAME for model in models):
                print(f"Model {MODEL_NAME} already loaded")
                return True
    except Exception as e:
        print(f"Warning: Could not check existing models: {e}")

    # Install model
    install_data = {"id": MODEL_ID, "name": MODEL_NAME}

    print(f"Installing model: {install_data}")

    try:
        response = requests.post(f"http://{HOST}:{PORT}/models/apply",
                                 json=install_data,
                                 timeout=30)

        if response.status_code != 200:
            print(f"Install request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False

        result = response.json()
        job_id = result.get('uuid')

        if not job_id:
            print(f"No job ID returned: {result}")
            return False

        print(f"Installation started, job ID: {job_id}")

        # Monitor job progress
        start_time = time.time()
        timeout = 600  # 10 minutes for model download

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://{HOST}:{PORT}/models/jobs/{job_id}", timeout=10)
                if response.status_code == 200:
                    status = response.json()

                    if status.get('processed', False):
                        if status.get('error'):
                            print(f"Installation failed: {status['error']}")
                            return False
                        else:
                            elapsed = time.time() - start_time
                            print(f"Model installed successfully in {elapsed:.1f} seconds")
                            return True
                    else:
                        elapsed = time.time() - start_time
                        message = status.get('message', 'processing')
                        print(f"Installing... ({elapsed:.1f}s) - {message}")

                        # Show system status during download
                        if int(elapsed) % 30 == 0:  # Every 30 seconds
                            show_system_status()

                        time.sleep(10)
                else:
                    print(f"Warning: Job status check failed: {response.status_code}")
                    time.sleep(5)

            except Exception as e:
                print(f"Warning: Error checking job status: {e}")
                time.sleep(5)

        print(f"Model installation timed out after {timeout} seconds")
        return False

    except Exception as e:
        print(f"Installation error: {e}")
        return False


def test_query():
    """Step 9: Test model query"""
    print(f"\n=== STEP 9: Testing model query ===")

    query_data = {
        "model": MODEL_NAME,
        "prompt": "What is the capital of France? Answer briefly.",
        "max_tokens": 50,
        "temperature": 0.7
    }

    print(f"Query: {query_data}")
    print("Measuring CPU usage during inference...")

    # Measure CPU before query
    cpu_before = psutil.cpu_percent(interval=1)

    try:
        start_time = time.time()
        response = requests.post(f"http://{HOST}:{PORT}/v1/completions", json=query_data)
        # timeout=120)
        elapsed = time.time() - start_time

        # Measure CPU during/after query
        cpu_after = psutil.cpu_percent(interval=1)

        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                answer = result['choices'][0]['text'].strip()
                usage = result.get('usage', {})

                print(f"Query successful in {elapsed:.1f} seconds")
                print(f"Answer: {answer}")
                print(f"Usage: {usage}")
                print(f"CPU before query: {cpu_before:.1f}%")
                print(f"CPU during query: {cpu_after:.1f}%")

                return True
            else:
                print(f"Invalid response format: {result}")
        else:
            print(f"Query failed: {response.status_code}")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"Query error: {e}")

    return False


def show_final_status(client):
    """Step 10: Show final status and cleanup info"""
    print("\n=== STEP 10: Final Status ===")

    try:
        container = client.containers.get(CONTAINER_NAME)
        container.reload()
        print(f"Container status: {container.status}")

        # Show final logs
        logs = container.logs(tail=20).decode('utf-8', errors='ignore')
        print("\nFinal container logs:")
        print("---")
        print(logs)
        print("---")

    except Exception as e:
        print(f"Error getting container status: {e}")

    show_system_status()

    print(f"\nTo stop container: docker stop {CONTAINER_NAME}")
    print(f"To remove container: docker rm {CONTAINER_NAME}")
    print(f"To view logs: docker logs {CONTAINER_NAME}")


def main():
    """Main execution"""
    print("=== MINIMAL LOCALAI TEST SCRIPT ===")
    print(f"Target: Install {MODEL_ID} and run a test query")

    try:
        # Sequential steps
        client = check_docker()
        cleanup_container(client)
        pull_image(client)
        container = create_container(client)
        start_container(container)

        if not wait_for_ready():
            print("\n LocalAI failed to start properly")
            # Show logs for debugging
            logs = container.logs().decode('utf-8', errors='ignore')
            print("Container logs for debugging:")
            print("=" * 50)
            print(logs)
            print("=" * 50)
            return False

        check_models()

        if not install_model():
            print("\n Model installation failed")
            return False

        if not test_query():
            print("\n Model query failed")
            return False

        show_final_status(client)
        print("\n ALL TESTS PASSED!")
        return True

    except KeyboardInterrupt:
        print("\n  Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

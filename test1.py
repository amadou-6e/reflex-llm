#!/usr/bin/env python3
"""
Ollama Docker Container Performance Test Script
Compares Ollama native vs Docker vs LocalAI performance
"""

import docker
import requests
import time
import json
import psutil
import subprocess
from pathlib import Path
import threading

# Configuration
OLLAMA_DOCKER_IMAGE = "ollama/ollama:latest"
OLLAMA_DOCKER_PORT = 11435  # Different from native Ollama
OLLAMA_NATIVE_PORT = 11434  # Native Ollama port
LOCALAI_PORT = 8080  # Your existing LocalAI

CONTAINER_NAME = "ollama-docker-test"
TEST_MODEL = "gemma:2b"  # Small, fast model for testing

# Test prompts
TEST_PROMPTS = [
    "What is the capital of France?", "Write a Python function to add two numbers.",
    "Explain quantum computing in one sentence.", "What is 2+2?", "Hello, how are you?"
]


def check_docker():
    """Check if Docker is available"""
    print("=== Checking Docker ===")
    try:
        client = docker.from_env()
        client.ping()
        print("✓ Docker is running")
        return client
    except Exception as e:
        print(f"✗ Docker error: {e}")
        return None


def cleanup_ollama_container(client):
    """Clean up existing Ollama container"""
    print("=== Cleaning up existing Ollama container ===")
    try:
        container = client.containers.get(CONTAINER_NAME)
        print(f"Found existing container: {container.status}")
        container.remove(force=True)
        print("✓ Removed existing container")
    except docker.errors.NotFound:
        print("✓ No existing container found")


def pull_ollama_image(client):
    """Pull Ollama Docker image"""
    print("=== Pulling Ollama Docker image ===")
    try:
        image = client.images.get(OLLAMA_DOCKER_IMAGE)
        print("✓ Ollama image already available")
        return image
    except docker.errors.NotFound:
        print(f"Pulling {OLLAMA_DOCKER_IMAGE}...")
        image = client.images.pull(OLLAMA_DOCKER_IMAGE)
        print("✓ Ollama image pulled successfully")
        return image


def start_ollama_container(client):
    """Start optimized Ollama container"""
    print("=== Starting Ollama Docker container ===")

    # Get system resources
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total // (1024**3)

    print(f"System: {cpu_count} CPUs, {memory_gb}GB RAM")

    # Create container with optimizations
    container = client.containers.run(
        OLLAMA_DOCKER_IMAGE,
        name=CONTAINER_NAME,
        ports={f"11434/tcp": OLLAMA_DOCKER_PORT},
        detach=True,
        # Resource optimizations
        mem_limit=f"{min(8, memory_gb-2)}g",  # Leave 2GB for system
        # Environment optimizations
        environment={
            "OLLAMA_NUM_PARALLEL": str(max(1, cpu_count // 4)),  # Parallel requests
            "OLLAMA_MAX_LOADED_MODELS": "2",  # Keep models in memory
        },
        # Volume for model persistence (optional)
        volumes={str(Path.cwd() / "ollama_data"): {
                     "bind": "/root/.ollama",
                     "mode": "rw"
                 }},
        # Remove container on exit for cleanup
        remove=False)

    print(f"✓ Container started: {container.id[:12]}")
    print(f"  - Memory limit: {min(8, memory_gb-2)}GB")
    print(f"  - Port: {OLLAMA_DOCKER_PORT}")

    return container


def wait_for_ollama_docker():
    """Wait for Ollama Docker to be ready"""
    print("=== Waiting for Ollama Docker to be ready ===")
    start_time = time.time()
    timeout = 60

    while time.time() - start_time < timeout:
        try:
            # Check if Ollama API is responding
            response = requests.get(f"http://localhost:{OLLAMA_DOCKER_PORT}/api/version", timeout=5)
            if response.status_code == 200:
                elapsed = time.time() - start_time
                print(f"✓ Ollama Docker ready in {elapsed:.1f} seconds")
                return True
        except requests.RequestException:
            pass

        elapsed = time.time() - start_time
        print(f"Waiting... ({elapsed:.1f}s)")
        time.sleep(3)

    print("✗ Ollama Docker not ready within timeout")
    return False


def download_model_to_docker():
    """Download test model to Docker Ollama"""
    print(f"=== Downloading {TEST_MODEL} to Docker Ollama ===")

    try:
        # Pull model via API
        response = requests.post(
            f"http://localhost:{OLLAMA_DOCKER_PORT}/api/pull",
            json={"name": TEST_MODEL},
            timeout=300  # 5 minutes for download
        )

        if response.status_code == 200:
            print(f"✓ Model {TEST_MODEL} downloaded to Docker Ollama")
            return True
        else:
            print(f"✗ Failed to download model: {response.status_code}")
            print(response.text)
            return False

    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        return False


def check_services():
    """Check which services are available"""
    print("\n=== Checking Available Services ===")

    services = {}

    # Check native Ollama
    try:
        response = requests.get(f"http://localhost:{OLLAMA_NATIVE_PORT}/api/version", timeout=5)
        if response.status_code == 200:
            services["ollama_native"] = f"http://localhost:{OLLAMA_NATIVE_PORT}"
            print("✓ Native Ollama available")
        else:
            print("✗ Native Ollama not responding")
    except:
        print("✗ Native Ollama not available")

    # Check Docker Ollama
    try:
        response = requests.get(f"http://localhost:{OLLAMA_DOCKER_PORT}/api/version", timeout=5)
        if response.status_code == 200:
            services["ollama_docker"] = f"http://localhost:{OLLAMA_DOCKER_PORT}"
            print("✓ Docker Ollama available")
        else:
            print("✗ Docker Ollama not responding")
    except:
        print("✗ Docker Ollama not available")

    # Check LocalAI
    try:
        response = requests.get(f"http://localhost:{LOCALAI_PORT}/readyz", timeout=5)
        if response.status_code == 200:
            services["localai"] = f"http://localhost:{LOCALAI_PORT}"
            print("✓ LocalAI available")
        else:
            print("✗ LocalAI not responding")
    except:
        print("✗ LocalAI not available")

    return services


def benchmark_service(service_name, base_url, model_name, prompt, timeout=60):
    """Benchmark a single service"""

    # Determine API endpoint based on service
    if "ollama" in service_name:
        api_url = f"{base_url}/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 50
            }
        }
    else:  # LocalAI
        api_url = f"{base_url}/v1/completions"
        payload = {"model": model_name, "prompt": prompt, "max_tokens": 50, "temperature": 0.7}

    # Monitor CPU before request
    cpu_before = psutil.cpu_percent(interval=0.1)

    try:
        start_time = time.time()
        response = requests.post(api_url, json=payload, timeout=timeout)
        end_time = time.time()

        # Monitor CPU after request
        cpu_after = psutil.cpu_percent(interval=0.1)

        elapsed = end_time - start_time

        if response.status_code == 200:
            result = response.json()

            # Extract response text based on service
            if "ollama" in service_name:
                response_text = result.get("response", "")
            else:  # LocalAI
                choices = result.get("choices", [])
                response_text = choices[0].get("text", "") if choices else ""

            return {
                "success": True,
                "elapsed": elapsed,
                "response_length": len(response_text),
                "response_preview": response_text[:100],
                "cpu_before": cpu_before,
                "cpu_after": cpu_after,
                "status_code": response.status_code
            }
        else:
            return {
                "success": False,
                "elapsed": elapsed,
                "error": f"HTTP {response.status_code}: {response.text[:200]}",
                "cpu_before": cpu_before,
                "cpu_after": cpu_after,
                "status_code": response.status_code
            }

    except Exception as e:
        return {
            "success": False,
            "elapsed": float('inf'),
            "error": str(e),
            "cpu_before": cpu_before,
            "cpu_after": cpu_after,
            "status_code": None
        }


def run_comprehensive_benchmark(services):
    """Run comprehensive benchmark across all services"""
    print("\n=== Running Comprehensive Benchmark ===")

    # Service configurations
    service_configs = {
        "ollama_native": {
            "model": TEST_MODEL,
            "timeout": 30
        },
        "ollama_docker": {
            "model": TEST_MODEL,
            "timeout": 30
        },
        # "localai": {
        #     "model": "gemma-3-1b-it",
        #     "timeout": 520
        # }  # LocalAI often slower
    }

    results = {}

    for service_name, base_url in services.items():
        if service_name not in service_configs:
            continue

        config = service_configs[service_name]
        print(f"\n--- Testing {service_name.upper()} ---")
        print(f"URL: {base_url}")
        print(f"Model: {config['model']}")

        service_results = []

        for i, prompt in enumerate(TEST_PROMPTS):
            print(f"\nTest {i+1}/5: {prompt[:50]}...")

            result = benchmark_service(service_name, base_url, config['model'], prompt,
                                       config['timeout'])

            if result["success"]:
                print(f"  ✓ {result['elapsed']:.2f}s - {result['response_preview'][:50]}...")
                print(f"  CPU: {result['cpu_before']:.1f}% → {result['cpu_after']:.1f}%")
            else:
                print(f"  ✗ Failed: {result['error']}")

            service_results.append(result)
            time.sleep(1)  # Brief pause between requests

        results[service_name] = service_results

        # Calculate service summary
        successful_results = [r for r in service_results if r["success"]]
        if successful_results:
            avg_time = sum(r["elapsed"] for r in successful_results) / len(successful_results)
            avg_cpu = sum(r["cpu_after"] for r in successful_results) / len(successful_results)
            success_rate = len(successful_results) / len(service_results) * 100

            print(f"\n📊 {service_name.upper()} SUMMARY:")
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Average time: {avg_time:.2f}s")
            print(f"  Average CPU: {avg_cpu:.1f}%")
        else:
            print(f"\n❌ {service_name.upper()}: All tests failed")

    return results


def generate_performance_report(results):
    """Generate detailed performance comparison report"""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON REPORT")
    print("=" * 60)

    # Service rankings
    service_scores = {}

    for service_name, service_results in results.items():
        successful_results = [r for r in service_results if r["success"]]

        if successful_results:
            avg_time = sum(r["elapsed"] for r in successful_results) / len(successful_results)
            success_rate = len(successful_results) / len(service_results)
            avg_cpu = sum(r["cpu_after"] for r in successful_results) / len(successful_results)

            # Score: lower time is better, higher success rate is better
            score = (1 / max(avg_time, 0.1)) * success_rate * 100

            service_scores[service_name] = {
                "avg_time": avg_time,
                "success_rate": success_rate,
                "avg_cpu": avg_cpu,
                "score": score
            }
        else:
            service_scores[service_name] = {
                "avg_time": float('inf'),
                "success_rate": 0,
                "avg_cpu": 0,
                "score": 0
            }

    # Sort by score (higher is better)
    ranked_services = sorted(service_scores.items(), key=lambda x: x[1]["score"], reverse=True)

    print("\n🏆 PERFORMANCE RANKING:")
    for i, (service_name, stats) in enumerate(ranked_services, 1):
        if stats["success_rate"] > 0:
            print(f"{i}. {service_name.upper()}")
            print(f"   Time: {stats['avg_time']:.2f}s")
            print(f"   Success: {stats['success_rate']*100:.1f}%")
            print(f"   CPU: {stats['avg_cpu']:.1f}%")
            print(f"   Score: {stats['score']:.1f}")
        else:
            print(f"{i}. {service_name.upper()} - FAILED")
        print()

    # Recommendations
    print("🎯 RECOMMENDATIONS:")

    best_service = ranked_services[0]
    if best_service[1]["success_rate"] > 0:
        print(f"✓ Use {best_service[0].upper()} for best performance")

        if best_service[1]["avg_time"] < 5:
            print("✓ Excellent response times achieved")
        elif best_service[1]["avg_time"] < 30:
            print("⚠ Acceptable response times")
        else:
            print("❌ Response times need improvement")

        if best_service[1]["avg_cpu"] > 50:
            print("✓ Good CPU utilization")
        else:
            print("⚠ CPU underutilized - consider optimizations")
    else:
        print("❌ All services failed - check configurations")


def cleanup_and_summary(client, container):
    """Cleanup and show final summary"""
    print("\n=== Cleanup ===")

    try:
        if container:
            print("Stopping Ollama Docker container...")
            container.stop()
            container.remove()
            print("✓ Container cleaned up")
    except Exception as e:
        print(f"Warning: Cleanup error: {e}")

    print("\n=== SUMMARY ===")
    print("This test compared:")
    print("1. Native Ollama (Windows installation)")
    print("2. Ollama in Docker container")
    print("3. LocalAI (your existing setup)")
    print()
    print("For production use, consider:")
    print("- Native Ollama for best Windows performance")
    print("- Docker Ollama for containerized deployments")
    print("- LocalAI for advanced features and model variety")


def main():
    """Main test execution"""
    print("=== OLLAMA DOCKER VS NATIVE VS LOCALAI BENCHMARK ===")
    print("This will test performance across all available services\n")

    # Check Docker
    client = check_docker()
    if not client:
        print("Docker not available - will test native services only")
        services = check_services()
    else:
        # Setup Docker Ollama
        cleanup_ollama_container(client)
        pull_ollama_image(client)
        container = start_ollama_container(client)

        if wait_for_ollama_docker():
            if download_model_to_docker():
                services = check_services()
            else:
                print("Failed to download model to Docker - testing available services")
                services = check_services()
        else:
            print("Docker Ollama failed to start - testing available services")
            services = check_services()
            container = None

    if not services:
        print("❌ No services available for testing")
        return

    # Run benchmarks
    try:
        results = run_comprehensive_benchmark(services)
        generate_performance_report(results)
    except KeyboardInterrupt:
        print("\n⚠ Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if client and 'container' in locals():
            cleanup_and_summary(client, container)


if __name__ == "__main__":
    main()

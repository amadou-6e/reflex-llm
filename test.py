import requests
import time

# Endpoint config
LOCALAI_HOST = "http://127.0.0.1:9045"
# GGUF_URL = "https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q4_K_M.gguf"
GGUF_URL = "https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct.Q4_K_M.gguf"

MODEL_NAME = "phi-4-mini"
FILENAME = "model.gguf"

# Use a base config YAML to trigger GGUF registration
CONFIG_URL = "https://raw.githubusercontent.com/mudler/LocalAI/main/gallery/base.yaml"

# 1. Submit model apply request
payload = {
    "name": MODEL_NAME,
    "url": CONFIG_URL,
    "files": [{
        "uri": GGUF_URL,
        "filename": FILENAME
    }],
    "overrides": {
        "backend": "llama.cpp"
    }
}

print("[*] Sending model apply request...")
resp = requests.post(f"{LOCALAI_HOST}/models/apply", json=payload)

if resp.status_code != 200:
    print(f"[!] Failed to apply model: {resp.status_code}")
    print(resp.text)
    exit(1)

response_data = resp.json()
job_id = response_data.get("uuid")
print(f"[+] Model apply job submitted. Job ID: {job_id}")

# 2. Poll job status
print("[*] Checking model job status...")

status_url = f"{LOCALAI_HOST}/models/jobs/{job_id}"

while True:
    status_resp = requests.get(status_url)
    data = status_resp.json()

    print(f"  - Status: {data.get('message')}")
    if data.get("processed", False):
        error = data.get("error")
        message = data.get("message", "")
        if error or "error" in message.lower():
            print("[!] Model load failed:")
            print(f"  ↳ {message}")
        else:
            print("[✅] Model loaded successfully.")
        break

    time.sleep(2)

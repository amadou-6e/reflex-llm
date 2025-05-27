- list models:
    curl http://localhost:8541/v1/models/   
- Download bert-embeddings
    curl -X POST http://localhost:8299/models/apply -H "Content-Type: application/json" -d "{\"id\": \"localai@microsoft_phi-4-mini-instruct\", \"name\": \"gpt4o\"}"
    curl -X POST http://localhost:8696/models/apply -H "Content-Type: application/json" -d "{\"id\": \"localai@gemma-3-1b-it\", \"name\": \"gemma-3-1b-it\"}"
- Curl all jobs:
    curl http://localhost:8541/v1/models
- Prompt the model:
    curl -X POST http://localhost:9045/v1/completions  -H "Content-Type: application/json"  -d "{\"model\": \"llama-3.2-1b-instruct-q4_k_m.gguf\", \"prompt\
": \"What is the capital of France?\", \"max_tokens\": 20}" 
    curl -X POST http://localhost:8299/v1/completions  -H "Content-Type: application/json"  -d "{\"model\": \"microsoft_phi-4-mini-instruct\", \"prompt\": \"What is the capital of France?\", \"max_tokens\": 20}"  
    curl -X POST http://localhost:8696/v1/completions  -H "Content-Type: application/json"  -d "{\"name\": \"gemma-3-1b-it\", \"prompt\": \"What is the capital of France?\", \"max_tokens\": 20, \"stream\": true}"   
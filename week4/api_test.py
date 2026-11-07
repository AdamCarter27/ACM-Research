import os
import requests
from pathlib import Path
from dotenv import load_dotenv

# load api key from env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is missing from .env")

url = "https://openrouter.ai/api/v1/chat/completions"

models = [
    "meta-llama/llama-3.3-70b-instruct",
    "openai/gpt-4o-mini",
]

prompt = "Explain what AI alignment means in 3 sentences."

for model in models:
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 300,
        },
        timeout=60,
    )

    if response.status_code != 200:
        print(f"Model: {model}")
        print(f"Status: {response.status_code}")
        print(f"Error: {response.text}")
        continue
    result = response.json()

    print(f"\nMODEL: {model}")
    print(result["choices"][0]["message"]["content"])


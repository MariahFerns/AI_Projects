import json
from pathlib import Path

import requests

# Load the API key from config.json (in the ADVANCED folder, one level above day2_fastapi)
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.json"
with open(CONFIG_PATH) as f:
    config = json.load(f)
api_key = config["OLLAMA_API_KEY"]

URL = "https://ollama.com/api/chat"
MODEL = "gpt-oss:20b-cloud"
HEADERS = {"Authorization": f"Bearer {api_key}"}


def ask(question: str) -> str:
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": question}],
        "stream": False,  # get the full answer in one response
    }
    response = requests.post(URL, headers=HEADERS, json=data)
    response.raise_for_status()
    return response.json()["message"]["content"]

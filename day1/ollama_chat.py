import json
import requests

# 1. Load the API key from config.json
with open("config.json") as f:
    config = json.load(f)
api_key = config["OLLAMA_API_KEY"]

# 2. API details
url = "https://ollama.com/api/chat"
headers = {"Authorization": f"Bearer {api_key}"}

# 3. Ask the user for a question
question = input("Ask a question: ")

# 4. Build the request body
data = {
    "model": "gpt-oss:20b-cloud",
    "messages": [{"role": "user", "content": question}],
    "stream": False,  # get the full answer in one response
}

# 5. Send the request and print the answer
response = requests.post(url, headers=headers, json=data)
response.raise_for_status()
answer = response.json()["message"]["content"]
print("\nAnswer:", answer)

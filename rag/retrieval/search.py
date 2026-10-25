"""Step 5 of the RAG PoC: embed a question with Nomic, fetch the top 3 chunks from Pinecone,
and ask GPT-OSS 120B (on Groq) for an answer grounded in those chunks.

How it works:
ask_llm() joins the chunks into one context block, each labelled [Section: ...], and sends it with the question.
The system prompt tells the model to answer only from that context, to say it doesn't know when the answer isn't there, and to name the sections it used.
temperature is 0.2 to keep answers close to the source text. max_tokens is 2000 because GPT-OSS reasons before it answers.
If no chunk passes the 0.6 threshold, the LLM isn't called at all, which saves a request.

to run: py -m search "What is Rotunda"  OR py -m search
"""

import os
import sys
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from fastembed import TextEmbedding
from pinecone import Pinecone

# Reads PINECONE_API_KEY, PINECONE_HOST, PINECONE_NAMESPACE and GROQ_API_KEY from ADVANCED/.env
load_dotenv(Path(__file__).parents[2] / ".env")

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
TOP_K = 3
MIN_SCORE = 0.6  # ignore chunks less similar than this

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = (
    "Answer the question using ONLY the context below. "
    "If the context does not contain the answer, say you don't know based on the documents. "
    "Mention the section names you used."
)

model = TextEmbedding(MODEL_NAME)
index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(host=os.environ["PINECONE_HOST"])


def ask_llm(question, matches):
    context = "\n\n".join(f"[Section: {m.metadata['section']}]\n{m.metadata['text']}" for m in matches)
    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            "temperature": 0.2,
            # GPT-OSS reasons before answering, so leave plenty of room.
            "max_tokens": 2000,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def search(question):
    # Questions get "search_query: " (chunks were stored with "search_document: ").
    vector = next(model.embed(["search_query: " + question]))
    vector = vector / np.linalg.norm(vector)

    result = index.query(
        vector=vector.tolist(),
        top_k=TOP_K,
        namespace=os.environ["PINECONE_NAMESPACE"],
        include_metadata=True,
    )

    matches = [m for m in result.matches if m.score >= MIN_SCORE]
    if not matches:
        print(f"\nNo chunks with similarity >= {MIN_SCORE}, so there is nothing to answer from.")
        return

    print("\nANSWER")
    print("=" * 80)
    print(ask_llm(question, matches))

    print("\n\nSOURCES")
    print("=" * 80)
    for rank, match in enumerate(matches, start=1):
        print(f"\n#{rank}  score={match.score:.3f}  section={match.metadata['section']}  source={match.metadata['source']}")
        print("-" * 80)
        print(match.metadata["text"])


def main():
    # Question from the command line, or ask interactively until an empty line.
    if len(sys.argv) > 1:
        search(" ".join(sys.argv[1:]))
        return
    while question := input("\nQuestion (Enter to quit): ").strip():
        search(question)


if __name__ == "__main__":
    main()

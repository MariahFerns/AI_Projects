"""Step 4 of the RAG PoC: upload the embeddings and their metadata to Pinecone.

How the script works:
It reads the Pinecone settings from ADVANCED/.env, so your key isn't in the code.
It uploads in batches of 100. That's more than one file needs now, but it means larger documents won't hit Pinecone's request size limit.
It uses upsert, which replaces a vector that already has the same id. Running it again updates the 26 chunks rather than adding duplicates.

"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

BASE_DIR = Path(__file__).parents[1]
EMBEDDINGS_DIR = BASE_DIR / "source" / "embeddings"

# Reads PINECONE_API_KEY, PINECONE_HOST and PINECONE_NAMESPACE from ADVANCED/.env
load_dotenv(BASE_DIR.parent / ".env")

BATCH_SIZE = 100


def main():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(host=os.environ["PINECONE_HOST"])
    namespace = os.environ["PINECONE_NAMESPACE"]

    for path in sorted(EMBEDDINGS_DIR.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]

        vectors = [
            {
                "id": row["id"],
                "values": row["embedding"],
                "metadata": {
                    "source": row["source"],
                    "section": row["section"],
                    "text": row["text"],
                    "model": row["model"],
                },
            }
            for row in rows
        ]

        for i in range(0, len(vectors), BATCH_SIZE):
            index.upsert(vectors=vectors[i:i + BATCH_SIZE], namespace=namespace)

        print(f"{path.name}: {len(vectors)} vectors -> namespace '{namespace}'")


if __name__ == "__main__":
    main()

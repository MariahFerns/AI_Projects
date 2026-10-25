"""Step 3 of the RAG PoC: embed chunks with Nomic and store them as JSONL.

embed_chunks.py embedded all 26 chunks with nomic-ai/nomic-embed-text-v1.5 and saved them to source/embeddings/RAG demo.jsonl.

What changed from my first version: I first wrote it with sentence-transformers, but Nomic's model code doesn't work with the transformers 5.x in your venv and crashed. As you chose, I switched to fastembed, which runs the same Nomic model through ONNX. The script is still about 35 lines. I added fastembed to requirements.txt. sentence-transformers is no longer used, so you can remove it if you like.

What each line of the output file contains: the original chunk fields (id, source, section, text), plus model (the model name) and embedding (768 numbers).

How the script works:
It adds search_document: in front of every chunk before embedding, as Nomic expects. At query time you'll add search_query: instead.
It normalizes each vector to length 1, so dot product and cosine similarity give the same score. I checked: every vector has length 1.0.

jsonl file is stored just for knowledge, no need to do..can send directly to pincecone
"""

import json
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

SOURCE_DIR = Path(__file__).parents[1] / "source"
CHUNKS_DIR = SOURCE_DIR / "chunks"
EMBEDDINGS_DIR = SOURCE_DIR / "embeddings"

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"


def main():
    EMBEDDINGS_DIR.mkdir(exist_ok=True)
    model = TextEmbedding(MODEL_NAME)

    for chunks_path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        with chunks_path.open(encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]

        # Nomic expects "search_document: " on stored text ("search_query: " on questions).
        texts = ["search_document: " + chunk["text"] for chunk in chunks]
        vectors = list(model.embed(texts))

        out_path = EMBEDDINGS_DIR / chunks_path.name
        with out_path.open("w", encoding="utf-8") as f:
            for chunk, vector in zip(chunks, vectors):
                chunk["model"] = MODEL_NAME
                # Normalize to length 1 so dot product equals cosine similarity.
                chunk["embedding"] = (vector / np.linalg.norm(vector)).tolist()
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        print(f"{chunks_path.name}: {len(chunks)} embeddings ({len(vectors[0])} dims) -> {out_path}")


if __name__ == "__main__":
    main()

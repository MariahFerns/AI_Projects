Python files

1. pdf -> markdown (md) -> can separate sections easily using delimiter ## -> using Dockling
markdown can also convert table. pdf -> markdown, word -> markdown, txt -> markdown, csv -> txt
2. md -> chunks
3. chunk -> contextual vector CV (using nomic embedding model) # can finetune it if not working best
4. CV -> VectorDB (Pinecone)
5. Query -> CV -> similarity search -> chunk -> LLM

## Setup

Uses the venv at `ADVANCED\.venv` (Python 3.12).

```powershell
ADVANCED\.venv\Scripts\pip install -r ADVANCED\day5_rag\requirements.txt
```

Pinecone settings are read from `ADVANCED\.env`:

```
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=rag-demo
PINECONE_NAMESPACE=testdata
PINECONE_HOST=https://...pinecone.io
```

The index must have `dimension=768` and `metric=cosine` (to match Nomic's embeddings).

## Folder layout

```
day5_rag/
├── ingestion/
│   ├── pdf_to_markdown.py     # step 1
│   ├── chunk_markdown.py      # step 2
│   ├── embed_chunks.py        # step 3
│   └── upload_to_pinecone.py  # step 4
├── retrieval/
│   └── search.py              # step 5
├── requirements.txt
└── source/
    ├── pdf/          # input PDFs
    ├── markdown/     # step 1 output (.md)
    ├── chunks/       # step 2 output (.jsonl)
    └── embeddings/   # step 3 output (.jsonl)
```

## Ingestion

Run the steps in order.

### Step 1: PDF -> Markdown (`pdf_to_markdown.py`)

Converts every `*.pdf` in `source/pdf/` to Markdown with Docling and saves `source/markdown/<same name>.md`.

```powershell
ADVANCED\.venv\Scripts\python ADVANCED\day5_rag\ingestion\pdf_to_markdown.py
```

- The first run downloads Docling's layout and table models (a few hundred MB). After that they're cached.
- If one PDF fails, the script reports the error and carries on with the rest.

### Step 2: Markdown -> chunks (`chunk_markdown.py`)

Splits each `.md` file at its `## ` headings (one chunk per section) and saves `source/chunks/<same name>.jsonl`. Each chunk keeps its heading at the top of its text, so its topic stays with it.

```powershell
ADVANCED\.venv\Scripts\python ADVANCED\day5_rag\ingestion\chunk_markdown.py
```

Each line: `{"id", "source", "section", "text"}`.

### Step 3: Chunks -> embeddings (`embed_chunks.py`)

Embeds each chunk with `nomic-ai/nomic-embed-text-v1.5` (via `fastembed`) and saves `source/embeddings/<same name>.jsonl`: the chunk fields plus `model` and `embedding` (768 numbers).

```powershell
ADVANCED\.venv\Scripts\python ADVANCED\day5_rag\ingestion\embed_chunks.py
```

- Each chunk is prefixed with `search_document: ` before embedding, as Nomic expects.
- Vectors are normalized to length 1.
- `fastembed` is used because Nomic's model code crashes with `sentence-transformers` on `transformers` 5.x.

### Step 4: Embeddings -> Pinecone (`upload_to_pinecone.py`)

Uploads every vector to the index in `PINECONE_NAMESPACE`, with `source`, `section`, `text` and `model` as metadata.

```powershell
ADVANCED\.venv\Scripts\python ADVANCED\day5_rag\ingestion\upload_to_pinecone.py
```

Uses `upsert`, so re-running replaces existing chunks with the same id instead of adding duplicates.

## Retrieval

### Step 5: Question -> top 3 chunks (`search.py`)

Embeds the question with Nomic (prefix `search_query: `), finds the 3 most similar chunks in Pinecone, and prints each with its similarity score, section, source and text.

```powershell
# one question
ADVANCED\.venv\Scripts\python ADVANCED\day5_rag\retrieval\search.py "Where do first-year students live?"

# keep asking questions; press Enter on an empty line to quit
ADVANCED\.venv\Scripts\python ADVANCED\day5_rag\retrieval\search.py
```

Next: pass the retrieved chunks to an LLM to write the answer.

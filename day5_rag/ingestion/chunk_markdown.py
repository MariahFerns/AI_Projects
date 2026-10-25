"""Step 2 of the RAG PoC: split Markdown files into section-wise chunks (JSONL)."""

import json
from pathlib import Path

SOURCE_DIR = Path(__file__).parents[1] / "source"
MARKDOWN_DIR = SOURCE_DIR / "markdown"
CHUNKS_DIR = SOURCE_DIR / "chunks"


def main():
    CHUNKS_DIR.mkdir(exist_ok=True)

    for md_path in sorted(MARKDOWN_DIR.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        sections = text.split("\n## ")

        out_path = CHUNKS_DIR / f"{md_path.stem}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for i, section in enumerate(sections):
                heading, _, body = section.partition("\n")
                heading = heading.lstrip("# ").strip()
                chunk = {
                    "id": f"{md_path.stem}-{i}",
                    "source": md_path.name,
                    "section": heading,
                    # Section heading is kept with the text so each chunk carries its own context.
                    "text": f"{heading}\n{body.strip()}",
                }
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        print(f"{md_path.name}: {len(sections)} chunks -> {out_path}")


if __name__ == "__main__":
    main()

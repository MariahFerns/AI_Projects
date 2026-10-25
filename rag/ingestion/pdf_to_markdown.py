"""Step 1 of the RAG PoC: convert source PDFs to Markdown using Docling."""

from pathlib import Path

from docling.document_converter import DocumentConverter

SOURCE_DIR = Path(__file__).parents[1] / "source"
PDF_DIR = SOURCE_DIR / "pdf"
MARKDOWN_DIR = SOURCE_DIR / "markdown"


def main():
    MARKDOWN_DIR.mkdir(exist_ok=True)
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}")
        return

    converter = DocumentConverter()
    for pdf_path in pdf_files:
        print(f"Converting {pdf_path.name} ...")
        try:
            result = converter.convert(pdf_path)
            markdown = result.document.export_to_markdown()
            out_path = MARKDOWN_DIR / f"{pdf_path.stem}.md"
            out_path.write_text(markdown, encoding="utf-8")
            print(f"  -> {out_path}")
        except Exception as e:
            print(f"  Failed: {e}")


if __name__ == "__main__":
    main()

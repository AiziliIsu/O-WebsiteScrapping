# Heuristic Bilingual PDF Sectioner

This tool extracts text from two-column (bilingual) PDFs and uses visual heuristics to identify section headers. It is ideal for documents where a Table of Contents is missing or unreliable.

## 🛠️ Logic & Heuristics
- **Visual Centering:** Detects headers by calculating the `bbox` (bounding box) relative to the page's center gutter.
- **Font Analysis:** Automatically determines the document's base font size and identifies headers by font-size deviation.
- **Deduplication:** Uses SHA-1 content hashing to merge accidental duplicate extractions.
- **Language Pairing:** Attempts to align Left (KY) and Right (RU) sections by matching numerical section keys (e.g., 1.1, Article 5).

## 🚀 Usage

### 1. Extract and Split Sections (Auto-Detection)
The `--split` flag activates the heuristic engine to find titles:
```bash
python parse_bilingual_pdf_to_json_files.py "legal_doc.pdf" --split --out-dir "./output"
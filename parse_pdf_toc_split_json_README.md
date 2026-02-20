# Bilingual PDF TOC Splitter

A robust tool for converting complex PDFs (especially official bilingual documents) into structured JSON sections based on the document's own Table of Contents.

## 🌟 Features
- **Layout Awareness:** Handles standard single-column PDFs or bilingual side-by-side (Left/Right) layouts.
- **Smart TOC Parsing:** Extracts section titles and page numbers from user-specified TOC pages.
- **Automatic Language Routing:** Detects Kyrgyz, Russian, or English and organizes output into subfolders.
- **Deduplication:** Uses SHA-1 hashing to ensure identical sections aren't processed twice.

## 🛠️ Installation
```bash
pip install pymupdf langdetect
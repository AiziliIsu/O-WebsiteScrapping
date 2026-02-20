# Scanned PDF OCR & Segmenter

A Python-based utility to convert scanned (non-selectable) PDFs into structured JSON data. It features advanced Table of Contents (TOC) parsing to split documents into logical sections.

## 🛠️ Prerequisites

1.  **Tesseract OCR:** You must have Tesseract installed on your system.
    * The script currently points to: `C:\Users\1000001392\AppData\Local\Programs\Tesseract-OCR\tesseract.exe`
2.  **Tessdata:** Ensure you have the language data files for `kir` (Kyrgyz), `rus` (Russian), and `eng` (English).

## 🚀 Key Features
* **OCR Engine:** Powered by `pytesseract` and `PyMuPDF` (fitz).
* **Section Splitting:** Automatically parses TOC page text to identify chapter boundaries.
* **Multi-language Support:** Specialized regex for Cyrillic characters (Ң, ң, Ү, ү, Ө, ө).
* **Automatic Sorting:** Detects language and routes files to `/russian`, `/kyrgyz`, or `/english` directories.

## 📖 Usage

### Mode 1: Full Document Extraction
Extracts the entire PDF into a single JSON file:
```bash
python scanned_pdf_ocr_to_json.py "path/to/document.pdf" --mode full
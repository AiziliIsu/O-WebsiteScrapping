# file: parse_pdf_toc_split_json.py
#
# Split PDF into section JSON files strictly according to Table of Contents
# where TOC page number(s) are provided via terminal.
#
# Supports:
# - single-layout PDFs
# - bilingual PDFs (left/right halves) if you pass --layout bilingual
#
# Usage (single):
#   python .\parse_pdf_toc_split_json.py "C:\path\doc.pdf" --out-dir "C:\out" --mode split --toc-page 2
#
# Usage (TOC spans 2 pages):
#   python .\parse_pdf_toc_split_json.py "C:\path\doc.pdf" --out-dir "C:\out" --mode split --toc-page 2 --toc-page-2 3
#
# Usage (bilingual):
#   python .\parse_pdf_toc_split_json.py "C:\path\bilingual.pdf" --layout bilingual --take both --mode split --toc-page 1 --out-dir "C:\out"
#
# Dependencies:
#   pip install pymupdf langdetect

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Sequence, Set, Tuple

import fitz  # PyMuPDF
from langdetect import DetectorFactory, detect

DetectorFactory.seed = 42
LOGGER = logging.getLogger("pdf_toc_splitter")

KYRGYZ_LETTERS_RE = re.compile(r"[ҢңҮүӨө]")

# "TITLE .... 12" OR "TITLE 12"
TOC_LINE_RE = re.compile(r"^(?P<title>.+?)\s+(?P<page>\d{1,4})\s*$", re.UNICODE)

# skip noise lines
TOC_NOISE_RE = re.compile(r"^(содержание|оглавление|мазмуну)\b", re.IGNORECASE)

NUM_HEADER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s*[.)]?\s*(.+?)\s*$")


@dataclass
class Line:
    text: str
    page: int
    is_bold: bool
    font_size: float
    bbox: Tuple[float, float, float, float]
    clip: Tuple[float, float, float, float]


@dataclass
class Section:
    name: str
    section_key: Optional[str]
    page_numbers: Set[int]
    content: str


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def normalize_multiline(text: str) -> str:
    if not text:
        return ""
    out: List[str] = []
    for ln in text.splitlines():
        ln = normalize_text(ln)
        if ln:
            out.append(ln)
    return "\n".join(out).strip()


def normalize_for_dedupe(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def sha1_norm(text: str) -> str:
    return hashlib.sha1(normalize_for_dedupe(text).encode("utf-8")).hexdigest()


def sanitize_filename(name: str, max_len: int = 160) -> str:
    name = normalize_text(name)
    name = re.sub(r'[\\/*?:"<>|]+', "_", name)
    name = name.strip(" .")
    if not name:
        name = "untitled"
    return name[:max_len]


def detect_language(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "ru"
    if KYRGYZ_LETTERS_RE.search(t):
        return "ky"
    try:
        lang = detect(t)
    except Exception:
        lang = "ru"
    if lang.startswith("ru"):
        return "ru"
    if lang.startswith("en"):
        return "en"
    if lang.startswith("ky"):
        return "ky"
    if re.search(r"[А-Яа-яЁё]", t):
        return "ru"
    return "en"


def lang_to_folder(lang: str) -> str:
    lang = (lang or "").lower()
    if lang == "ru":
        return "russian"
    if lang == "ky":
        return "kyrgyz"
    if lang == "en":
        return "english"
    return "other"


def validate_page_range(page_from: Optional[int], page_to: Optional[int], total_pages: int) -> Tuple[int, int]:
    if total_pages <= 0:
        return 1, 0
    pf = 1 if page_from is None else int(page_from)
    pt = total_pages if page_to is None else int(page_to)
    pf = max(1, pf)
    pt = min(total_pages, pt)
    if pf > pt:
        raise ValueError(f"Invalid page range: page_from={pf} > page_to={pt} (total_pages={total_pages})")
    return pf, pt


def _sort_key_line(y: float, x: float) -> Tuple[int, int]:
    return (int(y * 10), int(x * 10))


def extract_lines_page(page: fitz.Page, clip: Optional[fitz.Rect]) -> List[Line]:
    data = page.get_text("dict", clip=clip)
    out: List[Line] = []

    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for bline in block.get("lines", []):
            spans = bline.get("spans", [])
            if not spans:
                continue

            parts: List[str] = []
            sizes: List[float] = []
            bold_votes = 0

            bbox = tuple(bline.get("bbox", (0.0, 0.0, 0.0, 0.0)))
            if len(bbox) != 4:
                bbox = (0.0, 0.0, 0.0, 0.0)

            for sp in spans:
                s = sp.get("text", "")
                if s and s.strip():
                    parts.append(s.strip())
                sizes.append(float(sp.get("size", 0.0)))

                flags = int(sp.get("flags", 0))
                font = str(sp.get("font", "")).lower()
                if (flags & 16) or ("bold" in font) or ("bd" in font):
                    bold_votes += 1

            text = normalize_text(" ".join(parts))
            if not text:
                continue

            is_bold = bold_votes >= max(1, len(spans) // 2)
            font_size = float(median(sizes)) if sizes else 0.0

            c = clip if clip is not None else page.rect
            out.append(
                Line(
                    text=text,
                    page=page.number + 1,
                    is_bold=is_bold,
                    font_size=font_size,
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    clip=(c.x0, c.y0, c.x1, c.y1),
                )
            )

    out.sort(key=lambda ln: _sort_key_line(ln.bbox[1], ln.bbox[0]))
    return out


def extract_lines_single(pdf: fitz.Document, pf: int, pt: int) -> List[Line]:
    lines: List[Line] = []
    for pno in range(pf - 1, pt):
        page = pdf.load_page(pno)
        lines.extend(extract_lines_page(page, clip=None))
    return lines


def extract_lines_bilingual_side(pdf: fitz.Document, pf: int, pt: int, side: str, gutter_ratio: float) -> List[Line]:
    assert side in {"left", "right"}
    lines: List[Line] = []
    for pno in range(pf - 1, pt):
        page = pdf.load_page(pno)
        rect = page.rect
        w = rect.width
        gutter = w * gutter_ratio
        mid = rect.x0 + w / 2

        if side == "left":
            clip = fitz.Rect(rect.x0, rect.y0, mid - gutter, rect.y1)
        else:
            clip = fitz.Rect(mid + gutter, rect.y0, rect.x1, rect.y1)

        lines.extend(extract_lines_page(page, clip=clip))
    return lines


def extract_section_key(title: str) -> Optional[str]:
    m = NUM_HEADER_RE.match((title or "").strip())
    return m.group(1) if m else None


def parse_toc_entries_from_specific_pages(pdf: fitz.Document, toc_pages: Sequence[int]) -> List[Tuple[str, int]]:
    """
    Parse TOC entries from user-specified TOC pages.
    Handles wrapped titles across lines.
    Returns list[(title, start_page)] sorted by start_page.
    """
    entries: List[Tuple[str, int]] = []

    for p in toc_pages:
        if p < 1 or p > pdf.page_count:
            raise ValueError(f"TOC page {p} is out of range. PDF has {pdf.page_count} pages.")
        page = pdf.load_page(p - 1)
        txt = page.get_text("text") or ""
        raw_lines = [normalize_text(x) for x in txt.splitlines() if normalize_text(x)]

        buf: List[str] = []
        for ln in raw_lines:
            if TOC_NOISE_RE.search(ln):
                continue
            # ignore standalone numbers / decorative bits
            if re.fullmatch(r"\d{1,4}", ln):
                continue
            if len(ln) <= 1:
                continue

            buf.append(ln)
            joined = " ".join(buf).strip()

            m = TOC_LINE_RE.match(joined)
            if m:
                title = normalize_text(m.group("title"))
                page_num = int(m.group("page"))
                title = re.sub(r"^[•\-\–—\s]+", "", title).strip()

                if title and 1 <= page_num <= pdf.page_count:
                    entries.append((title, page_num))
                buf = []

    # de-dupe
    seen = set()
    uniq: List[Tuple[str, int]] = []
    for t, p in entries:
        k = (t.lower(), p)
        if k not in seen:
            seen.add(k)
            uniq.append((t, p))

    uniq.sort(key=lambda x: (x[1], x[0]))
    return uniq


def build_sections_from_toc(lines: List[Line], toc: List[Tuple[str, int]], pf: int, pt: int) -> List[Section]:
    """
    Create sections by page ranges defined by TOC.
    start = toc[i].page
    end = toc[i+1].page - 1
    last end = pt
    """
    toc2 = [(t, p) for (t, p) in toc if pf <= p <= pt]
    if not toc2:
        return []

    toc2.sort(key=lambda x: x[1])

    sections: List[Section] = []
    for i, (title, start) in enumerate(toc2):
        end = (toc2[i + 1][1] - 1) if i + 1 < len(toc2) else pt
        if end < start:
            continue

        chunk = [ln for ln in lines if start <= ln.page <= end]
        content = normalize_multiline("\n".join(ln.text for ln in chunk))
        pages = {ln.page for ln in chunk}
        sections.append(
            Section(
                name=title,
                section_key=extract_section_key(title),
                page_numbers=pages,
                content=content,
            )
        )
    return sections


def dedupe_sections(sections: List[Section]) -> List[Section]:
    if not sections:
        return sections

    best: Dict[str, Section] = {}
    for s in sections:
        h = sha1_norm(s.content)
        if h not in best or len(s.content) > len(best[h].content):
            best[h] = s
    return list(best.values())


def build_json(source_name: str, section: Section, forced_lang: Optional[str]) -> Dict:
    lang = forced_lang or detect_language(section.content)
    return {
        "source_name": source_name,
        "content": section.content,
        "metadata": {
            "section_name": section.name,
            "page_numbers": sorted(section.page_numbers),
            "language": lang,
            "rag_header": f"[Document: {source_name}]\n[Section: {section.name}]",
        },
    }


def write_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split PDF into JSON files by TOC page you provide.")
    parser.add_argument("file_path", help="Path to PDF.")
    parser.add_argument("--out-dir", default="processed_data", help="Base output directory.")
    parser.add_argument("--mode", choices=["full", "split"], default="split", help="full=one JSON, split=TOC sections.")
    parser.add_argument("--layout", choices=["single", "bilingual"], default="single", help="single or bilingual PDF.")
    parser.add_argument("--take", choices=["left", "right", "both"], default="both", help="For bilingual: which side(s).")
    parser.add_argument("--left-lang", default="ky", help="Left side language label (default ky).")
    parser.add_argument("--right-lang", default="ru", help="Right side language label (default ru).")
    parser.add_argument("--page-from", type=int, default=None, help="Start page (1-based inclusive).")
    parser.add_argument("--page-to", type=int, default=None, help="End page (1-based inclusive).")
    parser.add_argument("--gutter-ratio", type=float, default=0.02, help="Ignored center gutter ratio (default 0.02).")
    parser.add_argument("--toc-page", type=int, required=True, help="1-based page number where TOC is located.")
    parser.add_argument("--toc-page-2", type=int, default=None, help="Optional second TOC page if TOC spans multiple pages.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    file_path = Path(args.file_path)
    if file_path.suffix.lower() != ".pdf":
        raise ValueError("This script expects a PDF file.")

    base_out_dir = Path(args.out_dir)
    source_name = file_path.name
    stem = file_path.stem

    with fitz.open(file_path) as pdf:
        pf, pt = validate_page_range(args.page_from, args.page_to, pdf.page_count)

        toc_pages = [args.toc_page] + ([args.toc_page_2] if args.toc_page_2 else [])
        toc = parse_toc_entries_from_specific_pages(pdf, toc_pages)
        if not toc:
            raise RuntimeError(f"No TOC entries parsed from toc page(s): {toc_pages}. Try adding --toc-page-2 or check the page number.")

        written: List[Path] = []

        def out_folder(lang_code: str) -> Path:
            return base_out_dir / lang_to_folder(lang_code)

        if args.layout == "single":
            lines = extract_lines_single(pdf, pf, pt)

            if args.mode == "full":
                # full doc (ignore TOC)
                content = normalize_multiline("\n".join(ln.text for ln in lines))
                sec = Section("Full Document", None, {ln.page for ln in lines}, content)
                obj = build_json(source_name, sec, forced_lang=None)
                out = out_folder(obj["metadata"]["language"]) / f"{sanitize_filename(stem)}.json"
                write_json(obj, out)
                written.append(out)
            else:
                # split by TOC
                sections = dedupe_sections(build_sections_from_toc(lines, toc, pf=pf, pt=pt))
                # majority language folder
                langs = [detect_language(s.content) for s in sections] or ["other"]
                majority = max(set(langs), key=langs.count)
                folder = out_folder(majority)

                used: Set[str] = set()
                for s in sections:
                    key = s.section_key or sanitize_filename(s.name)[:24]
                    base = sanitize_filename(f"{stem}__{key}__{s.name}")
                    cand = base
                    k = 2
                    while cand.lower() in used:
                        cand = sanitize_filename(f"{base}__{k}")
                        k += 1
                    used.add(cand.lower())

                    out = folder / f"{cand}.json"
                    write_json(build_json(source_name, s, forced_lang=None), out)
                    written.append(out)

        else:
            # bilingual
            if args.take in {"left", "both"}:
                left_lines = extract_lines_bilingual_side(pdf, pf, pt, side="left", gutter_ratio=args.gutter_ratio)
                if args.mode == "full":
                    content = normalize_multiline("\n".join(ln.text for ln in left_lines))
                    sec = Section("Full Document", None, {ln.page for ln in left_lines}, content)
                    out = out_folder(args.left_lang) / f"{sanitize_filename(stem)}__{args.left_lang}.json"
                    write_json(build_json(source_name, sec, forced_lang=args.left_lang), out)
                    written.append(out)
                else:
                    left_sections = dedupe_sections(build_sections_from_toc(left_lines, toc, pf=pf, pt=pt))
                    for s in left_sections:
                        key = s.section_key or sanitize_filename(s.name)[:24]
                        out = out_folder(args.left_lang) / f"{sanitize_filename(stem)}__{key}__{args.left_lang}__{sanitize_filename(s.name)}.json"
                        write_json(build_json(source_name, s, forced_lang=args.left_lang), out)
                        written.append(out)

            if args.take in {"right", "both"}:
                right_lines = extract_lines_bilingual_side(pdf, pf, pt, side="right", gutter_ratio=args.gutter_ratio)
                if args.mode == "full":
                    content = normalize_multiline("\n".join(ln.text for ln in right_lines))
                    sec = Section("Full Document", None, {ln.page for ln in right_lines}, content)
                    out = out_folder(args.right_lang) / f"{sanitize_filename(stem)}__{args.right_lang}.json"
                    write_json(build_json(source_name, sec, forced_lang=args.right_lang), out)
                    written.append(out)
                else:
                    right_sections = dedupe_sections(build_sections_from_toc(right_lines, toc, pf=pf, pt=pt))
                    for s in right_sections:
                        key = s.section_key or sanitize_filename(s.name)[:24]
                        out = out_folder(args.right_lang) / f"{sanitize_filename(stem)}__{key}__{args.right_lang}__{sanitize_filename(s.name)}.json"
                        write_json(build_json(source_name, s, forced_lang=args.right_lang), out)
                        written.append(out)

    LOGGER.info("Parsed %d TOC entries from page(s) %s", len(toc), toc_pages)
    for t, p in toc[:15]:
        LOGGER.info("  TOC: %s -> %d", t, p)

    LOGGER.info("Wrote %d file(s) under %s", len(written), base_out_dir)
    for p in written:
        LOGGER.info("  %s", p)


if __name__ == "__main__":
    main()

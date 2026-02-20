# file: parse_bilingual_pdf_to_json_files.py
#
# Two-column bilingual PDF parser (left/right language) -> JSON files per doc or per section
# - Uses PyMuPDF (fitz) for PDF
# - Uses langdetect for language id (plus Kyrgyz-letter heuristic)
# - Can extract: left, right, or both
# - If splitting by sections, produces matched KY and RU JSON files per section number.

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Set, Tuple

import fitz  # PyMuPDF
from langdetect import DetectorFactory, detect

DetectorFactory.seed = 42
LOGGER = logging.getLogger("bilingual_pdf_parser")

KYRGYZ_LETTERS_RE = re.compile(r"[ҢңҮүӨө]")

# Strong RU legal headers
RU_HEADER_RE = re.compile(
    r"^\s*(?:"
    r"(?:статья|ст\.)\s*\d+(?:\.\d+)*"
    r"|(?:раздел)\s*\d+(?:\.\d+)*\.?"
    r"|(?:глава)\s*(?:[IVXLCDM]+|\d+)\.?"
    r"|(?:пункт|п\.)\s*\d+(?:\.\d+)*\.?"
    r"|(?:приложение)\s*(?:[A-ZА-Я]|\d+)\.?"
    r")\s*(?:[-–—:]?\s*)?.*$",
    re.IGNORECASE,
)

# Common "numbered section title" headers: "1. ОБЩИЕ ПОЛОЖЕНИЯ", "1.ЖАЛПЫ ЖОБОЛОР"
NUM_HEADER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s*[.)]?\s*([^\n]{2,220})\s*$")

# Glossary-style definitions should NOT become headers: "TERM — definition..."
DEF_DASH_RE = re.compile(r"\s[–—-]\s")
DEF_AFTER_DASH_RE = re.compile(r"(?:\s[–—-]\s)\s*([a-zа-яёҥүө0-9])", re.IGNORECASE)

TITLEISH_RE = re.compile(r"^\s*[A-ZА-ЯЁҢҮӨ][\w\s\-,–—:()«»\"'’ЁёҢҮӨ]+$")


@dataclass
class Line:
    text: str
    page: int
    is_bold: bool
    font_size: float


@dataclass
class Section:
    name: str
    section_key: Optional[str]  # usually "1", "2.1", etc.
    page_numbers: Set[int]
    content: str


# =========================
# Normalization / utils
# =========================
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def normalize_multiline(text: str) -> str:
    if not text:
        return ""
    lines = []
    for ln in text.splitlines():
        ln = normalize_text(ln)
        if ln:
            lines.append(ln)
    return "\n".join(lines).strip()


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


# =========================
# Bilingual extraction (left/right clipping)
# =========================
def extract_lines_side(
    pdf: fitz.Document,
    page_from: int,
    page_to: int,
    side: str,
    gutter_ratio: float = 0.02,
) -> List[Line]:
    """
    Extract lines from left or right half of each page using clip rectangles.
    side: 'left' | 'right'
    gutter_ratio: ignored center gutter to avoid table border bleed.
    """
    assert side in {"left", "right"}

    out: List[Line] = []
    for pno in range(page_from - 1, page_to):
        page = pdf.load_page(pno)
        rect = page.rect
        w = rect.width
        h = rect.height
        gutter = w * gutter_ratio
        mid = rect.x0 + w / 2

        if side == "left":
            clip = fitz.Rect(rect.x0, rect.y0, mid - gutter, rect.y1)
        else:
            clip = fitz.Rect(mid + gutter, rect.y0, rect.x1, rect.y1)

        data = page.get_text("dict", clip=clip)

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
                out.append(Line(text=text, page=pno + 1, is_bold=is_bold, font_size=font_size))

    return out


# =========================
# Header detection tuned for bilingual layout
# =========================
def looks_like_definition_line(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if not DEF_DASH_RE.search(t):
        return False
    return bool(DEF_AFTER_DASH_RE.search(t))


def compute_pdf_header_threshold(lines: List[Line]) -> float:
    sizes = [ln.font_size for ln in lines if ln.font_size > 0]
    if len(sizes) < 12:
        return 0.0
    return float(median(sizes)) * 1.20


def header_quality_score(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    if t.endswith((".", ";")):
        return 0.2
    tokens = [x for x in re.split(r"\s+", t) if x]
    if len(tokens) <= 2:
        return 0.35
    letters = re.findall(r"[A-Za-zА-Яа-яЁёҢңҮүӨө]", t)
    uppers = re.findall(r"[A-ZА-ЯЁҢҮӨ]", t)
    if letters:
        ratio = len(uppers) / len(letters)
        return 0.5 + min(0.5, ratio)
    return 0.5


def extract_section_key(text: str) -> Optional[str]:
    """
    Prefer numeric section ids at the start:
      "1." / "1.2" / "2)" etc.
    """
    t = (text or "").strip()
    m = NUM_HEADER_RE.match(t)
    if not m:
        return None
    return m.group(1)


def is_header_line(line: Line, pdf_size_threshold: float) -> bool:
    t = (line.text or "").strip()
    if not t:
        return False

    if RU_HEADER_RE.match(t):
        return True

    # In this layout, TOC-like numbered lines are valid headers
    if NUM_HEADER_RE.match(t) and not looks_like_definition_line(t):
        # avoid too short (e.g., "1." alone)
        if len(t) >= 5:
            return True

    # Glossary/definition lines are not headers
    if looks_like_definition_line(t):
        return False

    # Font-size based (PDF)
    if pdf_size_threshold > 0 and line.font_size >= pdf_size_threshold and len(t) <= 240:
        return True

    # Bold-only must be strong title-like (avoid terminology pages)
    if line.is_bold and len(t) <= 160 and TITLEISH_RE.match(t):
        return header_quality_score(t) >= 0.80

    return False


# =========================
# Split & dedupe
# =========================
def split_into_sections(lines: List[Line], split: bool) -> List[Section]:
    if not lines:
        return [Section(name="Full Document", section_key=None, page_numbers=set(), content="")]

    if not split:
        pages = {ln.page for ln in lines}
        content = normalize_multiline("\n".join(ln.text for ln in lines))
        return [Section(name="Full Document", section_key=None, page_numbers=pages, content=content)]

    thr = compute_pdf_header_threshold(lines)
    sections: List[Section] = []
    current_name = "Full Document"
    current_key: Optional[str] = None
    current_pages: Set[int] = set()
    current_lines: List[str] = []

    def flush():
        nonlocal current_name, current_key, current_pages, current_lines
        content = normalize_multiline("\n".join(current_lines))
        if content.strip():
            sections.append(
                Section(
                    name=current_name or "Section",
                    section_key=current_key,
                    page_numbers=set(current_pages),
                    content=content,
                )
            )

    for ln in lines:
        if is_header_line(ln, thr):
            # start new section
            if current_lines:
                flush()
            header = normalize_text(ln.text)
            current_name = header
            current_key = extract_section_key(header)
            current_pages = {ln.page}
            # include header in content
            current_lines = [header]
            continue

        current_pages.add(ln.page)
        current_lines.append(ln.text)

    flush()

    # If still empty, return single
    if not sections:
        pages = {ln.page for ln in lines}
        content = normalize_multiline("\n".join(ln.text for ln in lines))
        return [Section(name="Full Document", section_key=None, page_numbers=pages, content=content)]

    return sections


def dedupe_sections(sections: List[Section]) -> List[Section]:
    """
    - Remove exact duplicates by normalized hash (keep longer)
    - Remove contained duplicates: content A inside B and pages(A) ⊆ pages(B) => drop A
    """
    if not sections:
        return sections

    best: Dict[str, Section] = {}
    for s in sections:
        h = sha1_norm(s.content)
        if h not in best or len(s.content) > len(best[h].content):
            best[h] = s

    uniq = list(best.values())
    uniq_sorted = sorted(uniq, key=lambda x: len(x.content), reverse=True)

    keep: List[Section] = []
    for a in uniq_sorted:
        a_text = normalize_for_dedupe(a.content)
        a_pages = set(a.page_numbers)

        contained = False
        for b in keep:
            b_text = normalize_for_dedupe(b.content)
            b_pages = set(b.page_numbers)
            if a_text and b_text and a_text in b_text and a_pages.issubset(b_pages):
                contained = True
                break

        if not contained:
            keep.append(a)

    # stable ordering by first page then section key/name
    def key(s: Section) -> Tuple[int, str]:
        first = min(s.page_numbers) if s.page_numbers else 10**9
        return first, (s.section_key or ""), s.name

    return sorted(keep, key=key)


# =========================
# Pairing KY/RU sections
# =========================
def pair_sections_by_key(ky: List[Section], ru: List[Section]) -> List[Tuple[Optional[Section], Optional[Section], str]]:
    """
    Pair by section_key if present, else by ordinal index.
    Returns list of (ky_section, ru_section, pair_id)
    """
    ky_by_key: Dict[str, Section] = {s.section_key: s for s in ky if s.section_key}
    ru_by_key: Dict[str, Section] = {s.section_key: s for s in ru if s.section_key}

    keys = sorted(set(ky_by_key.keys()) | set(ru_by_key.keys()), key=lambda x: [int(p) for p in x.split(".")])

    paired: List[Tuple[Optional[Section], Optional[Section], str]] = []
    if keys:
        for k in keys:
            paired.append((ky_by_key.get(k), ru_by_key.get(k), k))
        return paired

    # fallback: align by index
    n = max(len(ky), len(ru))
    for i in range(n):
        pair_id = str(i + 1)
        paired.append((ky[i] if i < len(ky) else None, ru[i] if i < len(ru) else None, pair_id))
    return paired


# =========================
# JSON build & write
# =========================
def build_json(source_name: str, section: Section, forced_lang: Optional[str] = None) -> Dict:
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


# =========================
# MAIN
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse bilingual two-column PDF (left/right languages) into JSON files."
    )
    parser.add_argument("file_path", help="Path to bilingual PDF.")
    parser.add_argument("--out-dir", default="processed_data", help="Base output directory.")
    parser.add_argument("--split", action="store_true", help="Split into sections (per side). Without this: one file per side.")
    parser.add_argument(
        "--take",
        choices=["left", "right", "both"],
        default="both",
        help="Which side(s) to extract.",
    )
    parser.add_argument("--left-lang", default="ky", help="Language code for left side (default: ky).")
    parser.add_argument("--right-lang", default="ru", help="Language code for right side (default: ru).")
    parser.add_argument("--page-from", type=int, default=None, help="Start page (1-based, inclusive).")
    parser.add_argument("--page-to", type=int, default=None, help="End page (1-based, inclusive).")
    parser.add_argument("--gutter-ratio", type=float, default=0.02, help="Center gutter ratio to ignore (default 0.02).")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    file_path = Path(args.file_path)
    if file_path.suffix.lower() != ".pdf":
        raise ValueError("This script is for bilingual PDFs only (.pdf).")

    base_out_dir = Path(args.out_dir)

    with fitz.open(file_path) as pdf:
        pf, pt = validate_page_range(args.page_from, args.page_to, pdf.page_count)

        left_sections: List[Section] = []
        right_sections: List[Section] = []

        if args.take in {"left", "both"}:
            left_lines = extract_lines_side(pdf, pf, pt, side="left", gutter_ratio=args.gutter_ratio)
            left_sections = dedupe_sections(split_into_sections(left_lines, split=args.split))

        if args.take in {"right", "both"}:
            right_lines = extract_lines_side(pdf, pf, pt, side="right", gutter_ratio=args.gutter_ratio)
            right_sections = dedupe_sections(split_into_sections(right_lines, split=args.split))

    source_name = file_path.name
    stem = file_path.stem

    written: List[Path] = []

    def out_folder(lang_code: str) -> Path:
        return base_out_dir / lang_to_folder(lang_code)

    if args.take == "left":
        # write KY only
        if not args.split:
            sec = left_sections[0] if left_sections else Section("Full Document", None, set(), "")
            obj = build_json(source_name, sec, forced_lang=args.left_lang)
            out = out_folder(args.left_lang) / f"{sanitize_filename(stem)}__{args.left_lang}.json"
            write_json(obj, out)
            written.append(out)
        else:
            for s in left_sections:
                key = s.section_key or sanitize_filename(s.name)[:24]
                out = out_folder(args.left_lang) / f"{sanitize_filename(stem)}__{key}__{args.left_lang}__{sanitize_filename(s.name)}.json"
                write_json(build_json(source_name, s, forced_lang=args.left_lang), out)
                written.append(out)

    elif args.take == "right":
        # write RU only
        if not args.split:
            sec = right_sections[0] if right_sections else Section("Full Document", None, set(), "")
            obj = build_json(source_name, sec, forced_lang=args.right_lang)
            out = out_folder(args.right_lang) / f"{sanitize_filename(stem)}__{args.right_lang}.json"
            write_json(obj, out)
            written.append(out)
        else:
            for s in right_sections:
                key = s.section_key or sanitize_filename(s.name)[:24]
                out = out_folder(args.right_lang) / f"{sanitize_filename(stem)}__{key}__{args.right_lang}__{sanitize_filename(s.name)}.json"
                write_json(build_json(source_name, s, forced_lang=args.right_lang), out)
                written.append(out)

    else:
        # both: write matched KY and RU per section number (or index fallback)
        if not args.split:
            ky_sec = left_sections[0] if left_sections else Section("Full Document", None, set(), "")
            ru_sec = right_sections[0] if right_sections else Section("Full Document", None, set(), "")

            ky_out = out_folder(args.left_lang) / f"{sanitize_filename(stem)}__{args.left_lang}.json"
            ru_out = out_folder(args.right_lang) / f"{sanitize_filename(stem)}__{args.right_lang}.json"

            write_json(build_json(source_name, ky_sec, forced_lang=args.left_lang), ky_out)
            write_json(build_json(source_name, ru_sec, forced_lang=args.right_lang), ru_out)
            written.extend([ky_out, ru_out])
        else:
            pairs = pair_sections_by_key(left_sections, right_sections)
            for ky_s, ru_s, pair_id in pairs:
                if ky_s is not None:
                    ky_out = out_folder(args.left_lang) / (
                        f"{sanitize_filename(stem)}__{pair_id}__{args.left_lang}__{sanitize_filename(ky_s.name)}.json"
                    )
                    write_json(build_json(source_name, ky_s, forced_lang=args.left_lang), ky_out)
                    written.append(ky_out)

                if ru_s is not None:
                    ru_out = out_folder(args.right_lang) / (
                        f"{sanitize_filename(stem)}__{pair_id}__{args.right_lang}__{sanitize_filename(ru_s.name)}.json"
                    )
                    write_json(build_json(source_name, ru_s, forced_lang=args.right_lang), ru_out)
                    written.append(ru_out)

    LOGGER.info("Wrote %d file(s) under %s", len(written), base_out_dir)
    for p in written:
        LOGGER.info("  %s", p)


if __name__ == "__main__":
    main()

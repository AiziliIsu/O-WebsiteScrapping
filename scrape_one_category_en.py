# file: scrape_english_category.py

import argparse
import json
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup
from bs4.element import Tag
from tenacity import retry, stop_after_attempt, wait_exponential

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


# =========================
# CONFIG DEFAULTS (ENGLISH)
# =========================
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\1000001392\.vscode\RQ1 draft\dataset\english")

REQUEST_DELAY_SEC = 0.4
TIMEOUT_SEC = 30
MAX_PAGES = 2000
PRINT_DISCOVERED_URLS = True

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

DEFAULT_ACCEPT_LANGUAGE = "en-US,en;q=0.9,ru;q=0.6,ky;q=0.5"

# JS/scroll settings
MIN_TEXT_LEN_HTTP = 200
NAV_TIMEOUT_MS = 45000
JS_NETWORKIDLE_TIMEOUT_MS = 12000
JS_EXPAND_ROUNDS = 10

SCROLL_STEP_PX = 900
SCROLL_MAX_STEPS = 60
SCROLL_PAUSE_MS = 250

MIN_LINKS_BEFORE_JS_DISCOVERY = 8


# =========================
# UTIL
# =========================
def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def canonical_url(url: str) -> str:
    """
    Canonicalize to reduce duplicates.
    - strip query/fragment
    - lowercase scheme/host
    - remove trailing slash except root
    """
    p = urlparse(url)
    scheme = (p.scheme or "https").lower()
    netloc = (p.netloc or "").lower()

    path = p.path or "/"
    path = re.sub(r"/{2,}", "/", path)
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    return urlparse("")._replace(
        scheme=scheme, netloc=netloc, path=path, params="", query="", fragment=""
    ).geturl()


def sanitize_filename(name: str) -> str:
    name = normalize_space(name)
    name = re.sub(r'[\\/*?:"<>|]+', "_", name)
    name = name.strip(" .")
    return name[:140] if len(name) > 140 else name


def looks_like_file(url: str) -> bool:
    return bool(
        re.search(r"\.(pdf|doc|docx|xls|xlsx|ppt|pptx|zip|rar|7z|png|jpg|jpeg|webp)$", url, re.I)
    )


def same_domain(a: str, b: str) -> bool:
    return urlparse(canonical_url(a)).netloc == urlparse(canonical_url(b)).netloc


def normalize_scope_path(path: str) -> str:
    """
    Normalize /en/chastnym-klientam/... to /en/... (and same for ru/kg/en)
    so subtree checks don't accidentally exclude valid pages.
    """
    if not path:
        return "/"
    path = re.sub(r"/{2,}", "/", path)
    path = path.rstrip("/") or "/"

    path = re.sub(r"^/ru/chastnym-klientam(?=/)", "/ru", path, flags=re.I)
    path = re.sub(r"^/kg/chastnym-klientam(?=/)", "/kg", path, flags=re.I)
    path = re.sub(r"^/en/chastnym-klientam(?=/)", "/en", path, flags=re.I)

    return path


def within_root_subtree(url_path: str, root_path: str) -> bool:
    up = normalize_scope_path(url_path)
    rp = normalize_scope_path(root_path)

    if rp == "/":
        return True

    rp_prefix = rp if rp.endswith("/") else (rp + "/")
    return up == rp or up.startswith(rp_prefix)


def page_looks_interactive(soup: BeautifulSoup) -> bool:
    if soup.select_one('[aria-expanded="false"], [data-bs-toggle="collapse"], [data-toggle="collapse"]'):
        return True
    if soup.select_one(".accordion, .faq, .tabs, .tab, .collapse, .spoiler, .toggle"):
        return True
    if soup.find(string=re.compile(r"(more|read|details|expand|show|load|подробнее|далее)", re.I)):
        return True
    return False


# =========================
# EXTRACTION
# =========================
def extract_title(soup: BeautifulSoup) -> Optional[str]:
    h1 = soup.find("h1")
    if h1:
        t = normalize_space(h1.get_text(" ", strip=True))
        if t:
            return t
    if soup.title and soup.title.string:
        return normalize_space(soup.title.string)
    return None


def extract_breadcrumbs(soup: BeautifulSoup) -> List[str]:
    crumbs: List[str] = []

    nav = soup.find("nav", attrs={"aria-label": re.compile("breadcrumb", re.I)})
    if nav:
        for el in nav.find_all(["a", "span"], recursive=True):
            t = normalize_space(el.get_text(" ", strip=True))
            if t and t not in crumbs:
                crumbs.append(t)
        return crumbs

    ul = soup.find("ul", class_=re.compile("breadcrumb", re.I))
    if ul:
        for li in ul.find_all("li"):
            t = normalize_space(li.get_text(" ", strip=True))
            if t and t not in crumbs:
                crumbs.append(t)
        return crumbs

    return crumbs


def clean_dom_for_content_only(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(["script", "style", "noscript", "svg"]):
        try:
            tag.decompose()
        except Exception:
            pass

    # Keep this conservative: do NOT remove header/footer/nav/aside blindly
    # because some sites place content in unusual containers.
    # Instead, rely on extractor + footer trimming later.

    bad_kw = re.compile(
        r"(cookie|consent|banner|popup|modal|overlay|subscribe|newsletter|"
        r"social|share|pagination|pager|search|lang|language|chat|callback|widget)",
        re.I,
    )

    for el in soup.find_all(True):
        if not isinstance(el, Tag):
            continue
        try:
            classes = el.get("class") or []
            cls = " ".join(classes) if isinstance(classes, (list, tuple)) else str(classes)
            eid = el.get("id") or ""
        except Exception:
            continue

        if bad_kw.search(cls) or bad_kw.search(eid):
            try:
                text_len = len(normalize_space(el.get_text(" ", strip=True)))
                if text_len < 2000:
                    el.decompose()
            except Exception:
                continue

    return str(soup)


def extract_main_text(url: str, html: str) -> str:
    """
    Recall-first extraction: keep more content.
    We run two passes and keep the longer.
    """
    cleaned_html = clean_dom_for_content_only(html)

    t1 = trafilatura.extract(
        cleaned_html,
        url=url,
        include_comments=False,
        include_tables=True,
        include_links=False,
        favor_precision=False,
    ) or ""

    t2 = trafilatura.extract(
        cleaned_html,
        url=url,
        include_comments=False,
        include_tables=True,
        include_links=False,
        favor_recall=True,
    ) or ""

    best = t2 if len(t2) >= len(t1) else t1
    if best.strip():
        return normalize_space(best)

    soup = BeautifulSoup(cleaned_html, "html.parser")
    return normalize_space(soup.get_text(" ", strip=True))


def short_about(content: str, max_len: int = 280) -> Optional[str]:
    if not content:
        return None
    return content[:max_len] + ("…" if len(content) > max_len else "")


def collect_links(base_url: str, soup: BeautifulSoup) -> Set[str]:
    links: Set[str] = set()
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        u = canonical_url(urljoin(base_url, href))
        if looks_like_file(u):
            continue
        links.add(u)
    return links


# =========================
# PLAYWRIGHT (JS + SCROLL + EXPAND)
# =========================
def _pw_scroll_to_bottom(page) -> None:
    last_height = page.evaluate("() => document.body ? document.body.scrollHeight : 0")
    stable = 0

    for _ in range(SCROLL_MAX_STEPS):
        page.evaluate(f"() => window.scrollBy(0, {SCROLL_STEP_PX})")
        page.wait_for_timeout(SCROLL_PAUSE_MS)

        new_height = page.evaluate("() => document.body ? document.body.scrollHeight : 0")
        if new_height <= last_height:
            stable += 1
        else:
            stable = 0
            last_height = new_height

        if stable >= 3:
            break

    page.evaluate("() => window.scrollTo(0, 0)")
    page.wait_for_timeout(150)


def _pw_expand_interactive(page, rounds: int = JS_EXPAND_ROUNDS) -> None:
    for _ in range(rounds):
        clicked = page.evaluate(
            """
            () => {
              const clickables = [];

              document.querySelectorAll('button[aria-expanded="false"]').forEach(b => clickables.push(b));
              document.querySelectorAll('[data-bs-toggle="collapse"], [data-toggle="collapse"]').forEach(el => clickables.push(el));
              document.querySelectorAll('[role="tab"], .tabs button, .tab button, .nav-tabs button, .nav-tabs a')
                .forEach(el => clickables.push(el));

              const textKw = /(more|read|details|expand|show|load|подробнее|далее|раскрыть|показать)/i;
              document.querySelectorAll('button, a').forEach(el => {
                const t = (el.innerText || '').trim();
                if (t && t.length <= 80 && textKw.test(t)) clickables.push(el);
              });

              const uniq = [];
              const seen = new Set();
              for (const el of clickables) {
                const key = el.tagName + '|' + (el.id||'') + '|' + (el.className||'') + '|' + (el.innerText||'').slice(0,60);
                if (!seen.has(key)) { seen.add(key); uniq.push(el); }
              }

              let did = 0;
              for (const el of uniq) {
                try {
                  el.scrollIntoView({block: "center"});
                  el.click();
                  did += 1;
                } catch (e) {}
              }
              return did;
            }
            """
        )
        if not clicked:
            break
        try:
            page.wait_for_load_state("networkidle", timeout=3500)
        except Exception:
            pass
        page.wait_for_timeout(300)


def _pw_remove_footerish_blocks(page) -> None:
    page.evaluate(
        """
        () => {
          const sels = [
            '[role="banner"]','[role="navigation"]','[role="contentinfo"]',
            'header','footer','nav','aside'
          ];
          sels.forEach(sel => document.querySelectorAll(sel).forEach(el => el.remove()));

          const kw = /(cookie|consent|privacy|terms|legal|policy|footer|copyright)/i;
          document.querySelectorAll('*').forEach(el => {
            const cls = (el.className || '').toString();
            const id  = (el.id || '').toString();
            if (kw.test(cls) || kw.test(id)) {
              const txt = (el.innerText || '').trim();
              if (txt.length < 5000) el.remove();
            }
          });
        }
        """
    )


def _pw_extract_full_text(page) -> str:
    text = page.evaluate(
        """
        () => {
          const body = document.body;
          return body ? (body.innerText || '').trim() : '';
        }
        """
    )
    return normalize_space(text)


def get_js_text_with_playwright(url: str, accept_language: str) -> str:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=USER_AGENT,
                extra_http_headers={"Accept-Language": accept_language},
            )
            page = context.new_page()

            page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
            try:
                page.wait_for_load_state("networkidle", timeout=JS_NETWORKIDLE_TIMEOUT_MS)
            except PlaywrightTimeoutError:
                pass

            _pw_scroll_to_bottom(page)
            _pw_expand_interactive(page)
            _pw_scroll_to_bottom(page)
            _pw_remove_footerish_blocks(page)

            text = _pw_extract_full_text(page)

            page.close()
            context.close()
            browser.close()
            return text
    except Exception:
        return ""


def get_js_html_with_playwright(url: str, accept_language: str) -> str:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=USER_AGENT,
                extra_http_headers={"Accept-Language": accept_language},
            )
            page = context.new_page()

            page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
            try:
                page.wait_for_load_state("networkidle", timeout=JS_NETWORKIDLE_TIMEOUT_MS)
            except PlaywrightTimeoutError:
                pass

            _pw_scroll_to_bottom(page)
            _pw_expand_interactive(page)
            _pw_scroll_to_bottom(page)
            _pw_remove_footerish_blocks(page)

            html = page.content()

            page.close()
            context.close()
            browser.close()
            return html
    except Exception:
        return ""


# =========================
# PAGE SCRAPE
# =========================
@dataclass
class PageData:
    url: str
    title: Optional[str]
    breadcrumbs: List[str]
    content_text: str
    about: Optional[str]


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=10))
def fetch_html(client: httpx.Client, url: str) -> Tuple[str, str]:
    r = client.get(url)
    r.raise_for_status()
    return r.text, str(r.url)


def scrape_page(client: httpx.Client, url: str, accept_language: str) -> PageData:
    html, final_url = fetch_html(client, url)
    soup = BeautifulSoup(html, "html.parser")

    title = extract_title(soup)
    breadcrumbs = extract_breadcrumbs(soup)

    content_text = extract_main_text(final_url, html)

    # JS fallback if content is short or page looks interactive
    if (not content_text) or (len(content_text) < MIN_TEXT_LEN_HTTP) or page_looks_interactive(soup):
        js_text = get_js_text_with_playwright(final_url, accept_language=accept_language)
        if js_text and len(js_text) > len(content_text or ""):
            content_text = js_text

    about = short_about(content_text)

    return PageData(
        url=canonical_url(final_url),
        title=title,
        breadcrumbs=breadcrumbs,
        content_text=content_text,
        about=about,
    )


# =========================
# CRAWL
# =========================
def crawl_from_root(client: httpx.Client, root_url: str, accept_language: str) -> List[PageData]:
    root_url = canonical_url(root_url)
    root_path = urlparse(root_url).path

    visited: Set[str] = set()
    q = deque([root_url])
    pages: List[PageData] = []
    total_discovered = 0

    while q and len(pages) < MAX_PAGES:
        url = canonical_url(q.popleft())
        if url in visited:
            continue
        visited.add(url)

        if not same_domain(url, root_url):
            continue
        if looks_like_file(url):
            continue
        if not within_root_subtree(urlparse(url).path, root_path):
            continue

        time.sleep(REQUEST_DELAY_SEC)

        try:
            page = scrape_page(client, url, accept_language=accept_language)
        except Exception as e:
            print(f"[WARN] Failed: {url} -> {e}")
            continue

        pages.append(page)
        print(f"[SCRAPED] {page.url} (len={len(page.content_text) if page.content_text else 0})")

        # link discovery: raw HTML
        links: Set[str] = set()
        try:
            html2, final2 = fetch_html(client, url)
            soup2 = BeautifulSoup(html2, "html.parser")
            links |= collect_links(final2, soup2)
        except Exception:
            pass

        # JS DOM link discovery if raw is sparse
        if len(links) < MIN_LINKS_BEFORE_JS_DISCOVERY:
            js_html = get_js_html_with_playwright(url, accept_language=accept_language)
            if js_html:
                soup_js = BeautifulSoup(js_html, "html.parser")
                links |= collect_links(url, soup_js)

        for link in links:
            link = canonical_url(link)
            if link in visited:
                continue
            if same_domain(link, root_url) and within_root_subtree(urlparse(link).path, root_path):
                q.append(link)
                total_discovered += 1
                if PRINT_DISCOVERED_URLS:
                    print(f"  [FOUND] {link}")

    print(f"[INFO] Visited: {len(visited)} | Discovered: {total_discovered} | Saved pages: {len(pages)}")
    return pages


# =========================
# SAVE
# =========================
def infer_category_name(root_url: str, root_page: Optional[PageData]) -> str:
    if root_page and root_page.breadcrumbs:
        for candidate in reversed(root_page.breadcrumbs):
            if candidate and candidate.lower() not in {"home"}:
                return candidate
    if root_page and root_page.title:
        return root_page.title
    last_seg = urlparse(root_url).path.strip("/").split("/")[-1] or "category"
    return last_seg


def save_as_jsonl(output_dir: Path, category_name: str, root_url: str, pages: List[PageData], lang: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{sanitize_filename(category_name)}.jsonl"

    # final dedupe by canonical URL
    unique = {canonical_url(p.url): p for p in pages}
    pages = list(unique.values())

    with out_path.open("w", encoding="utf-8") as f:
        for p in pages:
            rel_path = urlparse(p.url).path.replace(urlparse(root_url).path.rstrip("/"), "").strip("/")
            subcategory_path = p.breadcrumbs if p.breadcrumbs else ([x for x in rel_path.split("/") if x] if rel_path else [])

            doc = {
                "source_url": canonical_url(p.url),
                "root_category_url": canonical_url(root_url),
                "lang": lang,
                "category": category_name,
                "subcategory_path": subcategory_path,
                "title": p.title,
                "about": p.about,
                "content_text": p.content_text,
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"[OK] Saved {len(pages)} pages into: {out_path}")
    return out_path


# =========================
# MAIN
# =========================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root_url", help="Root URL to crawl (category subtree).")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR), help=f"Output folder path (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--lang", default="en", help="Language code for metadata (default: en).")
    parser.add_argument("--accept-language", default=DEFAULT_ACCEPT_LANGUAGE, help="Override Accept-Language header.")
    args = parser.parse_args()

    root_url = canonical_url(args.root_url)
    output_dir = Path(args.out)
    accept_language = args.accept_language

    headers = {"User-Agent": USER_AGENT, "Accept-Language": accept_language}
    with httpx.Client(headers=headers, timeout=TIMEOUT_SEC, follow_redirects=True) as client:
        try:
            root_page = scrape_page(client, root_url, accept_language=accept_language)
        except Exception:
            root_page = None

        category_name = infer_category_name(root_url, root_page)
        pages = crawl_from_root(client, root_url, accept_language=accept_language)

        if root_page:
            root_key = canonical_url(root_url)
            if all(canonical_url(p.url) != root_key for p in pages):
                pages.insert(0, root_page)

        save_as_jsonl(output_dir=output_dir, category_name=category_name, root_url=root_url, pages=pages, lang=args.lang)


if __name__ == "__main__":
    main()

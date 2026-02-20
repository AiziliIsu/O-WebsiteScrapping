# file: scrape_one_category_test.py

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
# CONFIG
# =========================
OUTPUT_DIR = Path(r"C:\Users\1000001392\.vscode\RQ1 draft\dataset\russian")

REQUEST_DELAY_SEC = 0.4
TIMEOUT_SEC = 30
MAX_PAGES = 2000
PRINT_DISCOVERED_URLS = True

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# JS rendering fallback
MIN_TEXT_LEN_HTTP = 120
NAV_TIMEOUT_MS = 45000
JS_NETWORKIDLE_TIMEOUT_MS = 12000
JS_EXPAND_ROUNDS = 10

# Discovery robustness
MIN_LINKS_BEFORE_JS_DISCOVERY = 8

# Scroll/lazy-load handling
SCROLL_STEP_PX = 900
SCROLL_MAX_STEPS = 60
SCROLL_PAUSE_MS = 250


# =========================
# UTIL
# =========================
def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def canonical_url(url: str) -> str:
    p = urlparse(url)
    scheme = (p.scheme or "https").lower()
    netloc = (p.netloc or "").lower()

    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    if netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

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
    if not path:
        return "/"
    path = re.sub(r"/{2,}", "/", path)
    path = path.rstrip("/") or "/"

    path = re.sub(r"^/ru/chastnym-klientam(?=/)", "/ru", path, flags=re.I)
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
    if soup.find(string=re.compile(r"(подробнее|далее|раскрыть|показать|more|read)", re.I)):
        return True
    return False


# =========================
# FOOTER / DISCLAIMER CLEANING
# =========================
FOOTER_TAIL_PATTERNS = [
    r"©\s*\d{4}",
    r"все права защищены",
    r"all rights reserved",
    r"публичная оферта",
    r"пользователь(ское|ский)\s+соглаш",
    r"политик[а-и]\s+конфиденц",
    r"privacy\s+policy",
    r"terms\s+of\s+use",
    r"условия\s+использования",
    r"cookie",
    r"настоящ(ая|ие)\s+информац",
    r"осоо|оао|зао|ип",
    r"инн|огрн|юридическ",
    r"адрес[:\s]",
    r"телефон[:\s]",
]
FOOTER_TAIL_RE = re.compile("(" + "|".join(FOOTER_TAIL_PATTERNS) + ")", re.I)


def trim_footer_noise(text: str) -> str:
    if not text:
        return text

    t = text.strip()
    n = len(t)
    if n < 300:
        return t

    tail_start = int(n * 0.7)
    tail = t[tail_start:]
    m = FOOTER_TAIL_RE.search(tail)
    if not m:
        return t

    cut_at = tail_start + m.start()
    if cut_at < int(n * 0.4):
        return t

    return t[:cut_at].rstrip()


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

    for tag in soup.find_all(["header", "footer", "nav", "aside"]):
        try:
            tag.decompose()
        except Exception:
            pass

    bad_kw = re.compile(
        r"(cookie|consent|banner|popup|modal|overlay|subscribe|newsletter|"
        r"menu|navbar|sidebar|social|share|pagination|pager|search|lang|language|"
        r"chat|callback|widget|footer|copyright|legal|policy|agreement|terms|privacy|"
        r"offerta|оферт|политик|соглаш)",
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
                if text_len < 2500:
                    el.decompose()
            except Exception:
                continue

    return str(soup)


def extract_main_text(url: str, html: str) -> str:
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
# PLAYWRIGHT
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
    page.wait_for_timeout(200)


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

              const textKw = /(more|read|details|expand|show|load|далее|подробнее|раскрыть|показать|загрузить)/i;
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

        page.wait_for_timeout(500)


def _pw_remove_footerish_blocks(page) -> None:
    page.evaluate(
        """
        () => {
          const sels = [
            'header','footer','nav','aside',
            '[role="banner"]','[role="navigation"]','[role="contentinfo"]'
          ];
          sels.forEach(sel => document.querySelectorAll(sel).forEach(el => el.remove()));

          const kw = /(footer|copyright|legal|policy|agreement|terms|privacy|cookie|оферт|политик|соглаш)/i;
          document.querySelectorAll('*').forEach(el => {
            const cls = (el.className || '').toString();
            const id  = (el.id || '').toString();
            if (kw.test(cls) || kw.test(id)) {
              const txt = (el.innerText || '').trim();
              if (txt.length < 4000) el.remove();
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

    needs_js = (not content_text) or (len(content_text) < MIN_TEXT_LEN_HTTP) or page_looks_interactive(soup)
    if needs_js:
        js_text = get_js_text_with_playwright(final_url, accept_language=accept_language)
        if js_text and len(js_text) > len(content_text or ""):
            content_text = js_text

    content_text = trim_footer_noise(content_text)
    about = short_about(content_text)

    return PageData(
        url=canonical_url(final_url),
        title=title,
        breadcrumbs=breadcrumbs,
        content_text=content_text,
        about=about,
    )


# =========================
# PHASE 1: DISCOVER URLS ONLY
# =========================
def discover_urls(client: httpx.Client, root_url: str, accept_language: str) -> List[str]:
    root_url = canonical_url(root_url)
    root_path = urlparse(root_url).path

    discovered: Set[str] = set()
    ordered: List[str] = []
    q = deque([root_url])

    while q and len(ordered) < MAX_PAGES:
        url = canonical_url(q.popleft())
        if url in discovered:
            continue

        if not same_domain(url, root_url):
            continue
        if looks_like_file(url):
            continue
        if not within_root_subtree(urlparse(url).path, root_path):
            continue

        discovered.add(url)
        ordered.append(url)

        if PRINT_DISCOVERED_URLS:
            print(f"[DISCOVER] {url}")

        time.sleep(REQUEST_DELAY_SEC)

        links: Set[str] = set()

        try:
            html, final_url = fetch_html(client, url)
            soup = BeautifulSoup(html, "html.parser")
            links |= collect_links(final_url, soup)
        except Exception:
            final_url = url

        if len(links) < MIN_LINKS_BEFORE_JS_DISCOVERY:
            js_html = get_js_html_with_playwright(url, accept_language=accept_language)
            if js_html:
                soup_js = BeautifulSoup(js_html, "html.parser")
                links |= collect_links(url, soup_js)

        for link in links:
            link = canonical_url(link)
            if link in discovered:
                continue
            if not same_domain(link, root_url):
                continue
            if looks_like_file(link):
                continue
            if not within_root_subtree(urlparse(link).path, root_path):
                continue
            q.append(link)

    print(f"[INFO] Discovered URLs: {len(ordered)} (cap={MAX_PAGES})")
    return ordered


# =========================
# PHASE 2: SCRAPE URLS ONE BY ONE + STREAM SAVE
# =========================
def infer_category_name(root_url: str, root_page: Optional[PageData]) -> str:
    if root_page and root_page.breadcrumbs:
        for candidate in reversed(root_page.breadcrumbs):
            if candidate and candidate.lower() not in {"главная", "home"}:
                return candidate
    if root_page and root_page.title:
        return root_page.title
    last_seg = urlparse(root_url).path.strip("/").split("/")[-1] or "category"
    return last_seg


def save_as_jsonl_stream(
    category_name: str,
    root_url: str,
    urls: List[str],
    client: httpx.Client,
    accept_language: str,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{sanitize_filename(category_name)}.jsonl"

    seen_saved: Set[str] = set()

    with out_path.open("w", encoding="utf-8") as f:
        for i, url in enumerate(urls, start=1):
            try:
                page = scrape_page(client, url, accept_language=accept_language)
            except Exception as e:
                print(f"[WARN] Scrape failed ({i}/{len(urls)}): {url} -> {e}")
                continue

            cu = canonical_url(page.url)
            if cu in seen_saved:
                continue
            seen_saved.add(cu)

            rel_path = urlparse(page.url).path.replace(urlparse(root_url).path.rstrip("/"), "").strip("/")
            subcategory_path = page.breadcrumbs if page.breadcrumbs else ([x for x in rel_path.split("/") if x] if rel_path else [])

            doc = {
                "source_url": cu,
                "category": category_name,
                "subcategory_path": subcategory_path,
                "title": page.title,
                "about": page.about,
                "content_text": page.content_text,
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            print(f"[SCRAPED] ({i}/{len(urls)}) {cu} (len={len(page.content_text) if page.content_text else 0})")

    print(f"[OK] Saved JSONL into: {out_path}")
    return out_path


# =========================
# MAIN
# =========================
def run(root_url: str) -> None:
    root_url = canonical_url(root_url)
    accept_language = "ru-RU,ru;q=0.9,en;q=0.8"
    headers = {"User-Agent": USER_AGENT, "Accept-Language": accept_language}

    with httpx.Client(headers=headers, timeout=TIMEOUT_SEC, follow_redirects=True) as client:
        try:
            root_page = scrape_page(client, root_url, accept_language=accept_language)
        except Exception:
            root_page = None

        category_name = infer_category_name(root_url, root_page)

        # Phase 1: discovery only (kept in-memory, not saved to txt)
        urls = discover_urls(client, root_url, accept_language=accept_language)

        # Phase 2: scrape one-by-one and save JSONL only
        save_as_jsonl_stream(category_name, root_url, urls, client, accept_language)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Example: python scrape_one_category_test.py https://o.kg/ru/uslugi/rouming-dlya-korporativnykh-abonentov")
        raise SystemExit(2)

    run(sys.argv[1])

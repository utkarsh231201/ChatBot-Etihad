"""Document loaders (PDF + website) with metadata extraction.

Every Document carries: source, doc_title, doc_type, section, page (1-based; 0 = not applicable
for web pages). Documents are split on detected headings *before* chunking, so each chunk
inherits the correct section even when a section spans several pages.
"""
import re
from collections import Counter
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from pypdf import PdfReader

HEADING_RE = re.compile(r"^\d{1,2}(?:\.\d{1,2})*\.?\s*[A-Z][A-Za-z0-9 ,&/()'\-]{2,80}$")
MAX_HEADING_CHARS = 100
HEADING_SCALE = 1.25  # a heading's font is >= 1.25x the body font size


def _compact(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def _large_font_texts(page, body_size: float) -> list[str]:
    """Text runs set in a font clearly bigger than body text (i.e. headings), whitespace-stripped."""
    runs, cur = [], []

    def visit(text, cm, tm, font_dict, font_size):
        t = text.strip()
        if not t:
            return
        if font_size and font_size >= body_size * HEADING_SCALE:
            cur.append(t)
        elif cur:
            runs.append(_compact("".join(cur)))
            cur.clear()

    page.extract_text(visitor_text=visit)
    if cur:
        runs.append(_compact("".join(cur)))
    return runs


def _body_font_size(reader: PdfReader) -> float:
    sizes: Counter = Counter()

    def visit(text, cm, tm, font_dict, font_size):
        if text.strip() and font_size:
            sizes[round(font_size, 1)] += len(text)

    for p in list(reader.pages)[:10]:
        p.extract_text(visitor_text=visit)
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _norm(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip())


def _strip_boilerplate(pages: list[list[str]]) -> list[list[str]]:
    """Drop running headers/footers: lines (digits normalised) present on >50% of pages."""
    if len(pages) < 4:
        return pages
    counts = Counter(n for p in pages for n in {_norm(l) for l in p if l.strip()})
    common = {n for n, c in counts.items() if c > len(pages) * 0.5}
    return [[l for l in p if _norm(l) not in common] for p in pages]


def load_pdf(path: str | Path) -> list[Document]:
    """Section = nearest preceding heading. Headings are detected by font size (reliable for this
    kind of policy PDF); if the PDF has no size variation we fall back to numbered-line regex."""
    path = Path(path)
    reader = PdfReader(str(path))
    meta = reader.metadata
    title = (meta.title if meta and meta.title else None) or path.stem.replace("_", " ")
    raw_pages = [(p.extract_text() or "").splitlines() for p in reader.pages]
    pages = _strip_boilerplate(raw_pages)
    body = _body_font_size(reader)
    big = [_large_font_texts(p, body) if body else [] for p in reader.pages]
    for i, runs in enumerate(big):  # a page that is mostly large text has no headings, it is all body copy
        total = sum(len(_compact(l)) for l in pages[i]) or 1
        if sum(len(r) for r in runs) / total > 0.5:
            big[i] = []
    use_fonts = any(big)

    def is_heading(line: str, page_idx: int) -> bool:
        c = _compact(line)
        if len(c) < 3 or len(line.strip()) > MAX_HEADING_CHARS:  # long large-font lines are body copy, not headings
            return False
        if use_fonts:
            return any(c in run for run in big[page_idx])
        return bool(HEADING_RE.match(line.strip())) and not line.strip().endswith((".", ":", ","))

    docs: list[Document] = []
    section = "Introduction"
    for page_idx, lines in enumerate(pages):
        buf: list[str] = []
        sec_for_buf = section

        def flush():
            text = "\n".join(buf).strip()
            if text:
                docs.append(Document(page_content=text, metadata={
                    "source": path.name, "doc_title": title, "doc_type": "pdf",
                    "section": sec_for_buf, "page": page_idx + 1}))

        for line in lines:
            if is_heading(line, page_idx):
                if buf and all(is_heading(l, page_idx) for l in buf):
                    # heading wrapped over several lines ("27." / "Pets in Cabin (PETC)")
                    joined = f"{section} {line.strip()}"
                    if len(joined) <= MAX_HEADING_CHARS:
                        section = sec_for_buf = joined
                else:
                    flush()
                    buf = []
                    section = sec_for_buf = line.strip()
            buf.append(line)
        flush()
    return docs

def load_website(url: str, timeout: int = 20) -> list[Document]:
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "etihad-policy-bot/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else url
    root = soup.find("main") or soup.body or soup

    docs: list[Document] = []
    section, buf = title, []

    def flush():
        text = "\n".join(buf).strip()
        if text:
            docs.append(Document(page_content=text, metadata={
                "source": url, "doc_title": title, "doc_type": "web",
                "section": section, "page": 0}))

    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "td"]):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name in {"h1", "h2", "h3", "h4"}:
            flush()
            buf, section = [], text
        else:
            buf.append(text)
    flush()
    return docs


def load_sources(pdf_paths: list[str | Path], urls: list[str]) -> list[Document]:
    docs: list[Document] = []
    for p in pdf_paths:
        docs += load_pdf(p)
    for u in urls:
        docs += load_website(u)
    return docs

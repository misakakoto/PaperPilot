"""Minimal PDF ingestion for the slim PaperPilot project."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from .models import PaperChunk, PaperDocument, PaperPage
from .store import LibrarySyncResult, PaperKnowledgeStore


def discover_pdfs(roots: Iterable[str | Path]) -> list[Path]:
    paths: set[Path] = set()
    for raw in roots:
        root = Path(raw).expanduser().resolve()
        if root.is_file() and root.suffix.casefold() == ".pdf":
            paths.add(root)
        elif root.is_dir():
            paths.update(path.resolve() for path in root.rglob("*.pdf") if path.is_file())
    return sorted(paths, key=lambda value: str(value).casefold())


def ingest_pdfs(
    store: PaperKnowledgeStore,
    roots: Iterable[str | Path],
    *,
    chunk_chars: int = 3_500,
    overlap_chars: int = 350,
    embedder: object | None = None,
) -> LibrarySyncResult:
    if chunk_chars < 500 or not 0 <= overlap_chars < chunk_chars:
        raise ValueError("invalid chunking configuration")
    documents: list[PaperDocument] = []
    chunks: list[PaperChunk] = []
    for paper_index, path in enumerate(discover_pdfs(roots), start=1):
        document, paper_chunks = _read_pdf(
            path,
            paper_id=f"P{paper_index}",
            chunk_chars=chunk_chars,
            overlap_chars=overlap_chars,
        )
        documents.append(document)
        chunks.extend(paper_chunks)
    return store.sync_documents(documents, chunks, embedder=embedder)


def _read_pdf(
    path: Path,
    *,
    paper_id: str,
    chunk_chars: int,
    overlap_chars: int,
) -> tuple[PaperDocument, list[PaperChunk]]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    reader = PdfReader(str(path))
    pages: list[PaperPage] = []
    chunks: list[PaperChunk] = []
    ordinal = 0
    for page_number, page in enumerate(reader.pages, start=1):
        text = re.sub(r"[ \t]+", " ", page.extract_text() or "").strip()
        pages.append(PaperPage(number=page_number, text=text))
        start = 0
        while start < len(text):
            excerpt = text[start : start + chunk_chars].strip()
            if excerpt:
                ordinal += 1
                chunks.append(
                    PaperChunk(
                        id=f"{paper_id}:p{page_number}:c{ordinal}",
                        paper_id=paper_id,
                        page_start=page_number,
                        page_end=page_number,
                        section="PDF page",
                        text=excerpt,
                        token_count_estimate=max(1, len(excerpt) // 4),
                    )
                )
            if start + chunk_chars >= len(text):
                break
            start += chunk_chars - overlap_chars
    title = next((line.strip() for line in (pages[0].text if pages else "").splitlines() if line.strip()), path.stem)
    document = PaperDocument(
        id=paper_id,
        file_name=path.name,
        sha256=digest,
        title=title[:1_000],
        page_count=max(1, len(reader.pages)),
        processed_pages=len(pages),
        pages=pages,
    )
    return document, chunks

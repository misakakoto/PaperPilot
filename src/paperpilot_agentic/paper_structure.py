"""Versioned paragraph/section locators; originals, never generated answers."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from paperpilot_retrieval.models import PaperChunk, PaperDocument, PaperPage

from .layout import parse_pdf_layout, fitz

VERSION = "paper-structure-pilot-v1"
HEADING = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*\.?|[IVXLC]+\.|[A-Z]\.)\s+\S.{1,95}|"
    r"Abstract|Introduction|Methods?|Results?|Discussion|Conclusions?|References|Bibliography|"
    r"Limitations?|Acknowledg(?:e)?ments?)$",
    re.I,
)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def graph_identity(graph):
    return digest(
        json.dumps(
            {k: v for k, v in graph.items() if k != "identity"}, sort_keys=True, ensure_ascii=False
        ).encode()
    )


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def recognized_heading(text, suggested):
    # An inherited heading is not evidence that the current text is a heading.
    if len(text) > 110 or len(text.split()) > 16 or text.endswith((".", ",", "-", ":")):
        return False
    if re.search(r"IEEE|TRANSACTIONS|\bet al\b|arXiv|\bVOL\b", text, re.I):
        return False
    numbered = bool(re.match(r"^(?:[IVX]+\.|[A-Z]\.|\d+(?:\.\d+)+\.?|\d+\.)\s+[A-Z]", text))
    return bool(
        HEADING.fullmatch(text)
        and (
            numbered
            or re.fullmatch(
                r"Abstract|References|Bibliography|Limitations?|Conclusions?", text, re.I
            )
        )
    )


def split_paragraph(text, limit=3500):
    """Yield exact character ranges, preferring sentence endings."""
    start = 0
    while start < len(text):
        end = min(len(text), start + limit)
        if end < len(text):
            endings = list(re.finditer(r"[.!?。！？]\s+", text[start:end]))
            if endings and endings[-1].end() > limit // 2:
                end = start + endings[-1].end()
        yield start, end
        start = end


def structure_from_objects(sha, title, objects, *, limit=3500):
    if not 500 <= limit <= 3500:
        raise ValueError("Paragraph chunk limit must be 500..3500 characters")
    sections, paragraphs, assets = [], [], []
    stack = []
    section = None
    for obj in objects:
        text = re.sub(r"\s+", " ", obj["text"]).strip()
        if not text:
            continue
        suggested = (obj.get("section_path") or [""])[-1]
        heading = recognized_heading(text, suggested)
        if heading or section is None:
            if heading:
                number = re.match(r"^(\d+(?:\.\d+)*)\.?\s", text)
                level = (
                    number[1].count(".") + 1
                    if number
                    else 1
                    if re.match(r"^[IVX]+\.\s", text)
                    else 2
                    if re.match(r"^[A-Z]\.\s", text)
                    else 1
                )
                stack = stack[: level - 1] + [text]
            section = {
                "section_id": f"section_{len(sections)}",
                "title": " ::: ".join(stack) if stack else "未识别章节",
                "heading_status": "STRUCTURAL_LOCATOR" if stack else "UNKNOWN",
                "paragraph_ids": [],
                "parent_title": " ::: ".join(stack[:-1]) or None,
            }
            sections.append(section)
        pid = f"paragraph_{len(paragraphs)}"
        span = {
            "page": obj["page"],
            "bbox": obj["bbox"],
            "object_id": obj["object_id"],
            "text_start": 0,
            "text_end": len(text),
            "coordinate_space": "pdf_points_top_left",
        }
        # Merge a cross-page continuation only when the last block is unfinished.
        prior = paragraphs[-1] if paragraphs else None
        if (
            not heading
            and prior
            and prior["section_id"] == section["section_id"]
            and prior["spans"][-1]["page"] + 1 == obj["page"]
            and not re.search(r"[.!?。！？:]$", prior["text"])
            and re.match(r"^[a-z]", text)
            and obj["object_type"] == "text"
        ):
            span["text_start"] = len(prior["text"]) + 1
            span["text_end"] += span["text_start"]
            prior["text"] += " " + text
            prior["spans"].append(span)
            pid = prior["paragraph_id"]
        else:
            paragraphs.append(
                {
                    "paragraph_id": pid,
                    "section_id": section["section_id"],
                    "text": text,
                    "spans": [span],
                }
            )
            section["paragraph_ids"].append(pid)
        if obj["object_type"] in {"table", "equation", "figure"}:
            assets.append(
                {
                    "object_id": obj["object_id"],
                    "kind": obj["object_type"],
                    "caption": obj.get("caption"),
                    "text": text,
                    "page": obj["page"],
                    "bbox": obj["bbox"],
                    "paragraph_id": pid,
                    "section_id": section["section_id"],
                    "artifact_path": obj.get("artifact_path"),
                    "status": "LOCATOR_ONLY_NOT_FACT_VERIFIED",
                    "parser": obj.get("parser"),
                    "coordinate_space": "pdf_points_top_left",
                }
            )
    chunks = []
    for sec in sections:
        pending, size = [], 0
        for paragraph in (p for p in paragraphs if p["section_id"] == sec["section_id"]):
            for start, end in split_paragraph(paragraph["text"], limit):
                segment = {
                    "paragraph_id": paragraph["paragraph_id"],
                    "start": start,
                    "end": end,
                    "spans": [
                        s
                        for s in paragraph["spans"]
                        if s["text_start"] < end and s["text_end"] > start
                    ],
                }
                part = paragraph["text"][start:end]
                if pending and size + len(part) + 2 > limit:
                    chunks.append(
                        {"section_id": sec["section_id"], "section": sec["title"], "parts": pending}
                    )
                    pending, size = [], 0
                pending.append({"text": part, **segment})
                size += len(part) + (2 if size else 0)
        if pending:
            chunks.append(
                {"section_id": sec["section_id"], "section": sec["title"], "parts": pending}
            )
    result = {
        "version": VERSION,
        "paper_sha256": sha,
        "title": title,
        "chunk_chars": limit,
        "sections": sections,
        "paragraphs": paragraphs,
        "objects": assets,
        "chunks": chunks,
    }
    result["identity"] = graph_identity(result)
    return result


def build_structure(source, folder, *, title, paper_id):
    source, folder = Path(source), Path(folder)
    manifest = parse_pdf_layout(source, folder / "layout", render_regions=False)
    graph = structure_from_objects(
        manifest.document_sha256, title, [o.to_dict() for o in manifest.objects]
    )
    with fitz.open(source) as pdf:
        pages = [PaperPage(number=i + 1, text=p.get_text()) for i, p in enumerate(pdf)]
    doc = PaperDocument(
        id=paper_id,
        file_name=source.name,
        sha256=graph["paper_sha256"],
        title=title,
        page_count=len(pages),
        processed_pages=len(pages),
        pages=pages,
    )
    chunks = []
    for ordinal, row in enumerate(graph["chunks"], 1):
        pages_used = [s["page"] for p in row["parts"] for s in p["spans"]]
        first, last = min(pages_used), max(pages_used)
        page_label = str(first) if first == last else f"{first}-p{last}"
        chunk = PaperChunk(
            id=f"{paper_id}:p{page_label}:c{ordinal}",
            paper_id=paper_id,
            page_start=first,
            page_end=last,
            section=row["section"],
            text="\n\n".join(p["text"] for p in row["parts"]),
            token_count_estimate=max(1, sum(len(p["text"]) for p in row["parts"]) // 4),
        )
        chunks.append(chunk)
        row["run_chunk_id"] = chunk.id
    folder.mkdir(parents=True, exist_ok=True)
    content = "\n".join(p["text"] for p in graph["paragraphs"]) + "\n"
    (folder / "paragraphs.txt").write_bytes(content.encode("utf-8"))
    graph["text_file_sha256"] = digest((folder / "paragraphs.txt").read_bytes())
    graph["identity"] = graph_identity(graph)
    save_json(folder / "structure.json", graph)
    return doc, chunks, graph


class PaperTextNavigator:
    """Only registered IDs cross the interface; file paths remain host-owned."""

    def __init__(self, root):
        self.root = Path(root).resolve()

    def _graph(self, paper_sha256):
        if not isinstance(paper_sha256, str) or not re.fullmatch("[a-f0-9]{64}", paper_sha256):
            raise ValueError("Expected registered paper SHA")
        folder = self.root / "papers" / paper_sha256
        if (
            folder.is_symlink()
            or (folder / "structure.json").is_symlink()
            or (folder / "paragraphs.txt").is_symlink()
        ):
            raise ValueError("Registered assets cannot be symlinks")
        graph = json.loads((folder / "structure.json").read_bytes())
        if graph["version"] != VERSION or graph["paper_sha256"] != paper_sha256:
            raise ValueError("Structure version/source mismatch")
        if graph["identity"] != graph_identity(graph):
            raise ValueError("Structure identity changed")
        if (self.root / "index.json").is_file():
            manifest = json.loads((self.root / "index.json").read_bytes())
            entry = next((p for p in manifest["papers"] if p["sha256"] == paper_sha256), None)
            if entry is None or entry["structure_sha256"] != digest(
                (folder / "structure.json").read_bytes()
            ):
                raise ValueError("Structure is outside the frozen index")
        content = (folder / "paragraphs.txt").read_bytes()
        expected = ("\n".join(p["text"] for p in graph["paragraphs"]) + "\n").encode()
        if digest(content) != graph["text_file_sha256"] or content.replace(b'\r\n', b'\n') != expected:
            raise ValueError("Registered text changed")
        return folder, graph

    def find_text(self, paper_sha256, query, *, limit=20):
        if not isinstance(query, str) or not query.strip() or len(query) > 160 or "\n" in query:
            raise ValueError("Supply one literal query of 1..160 characters")
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("Match limit must be 1..20")
        folder, graph = self._graph(paper_sha256)
        executable = shutil.which("rg")
        if not executable:
            raise RuntimeError("ripgrep unavailable; no unannounced backend substitution")
        completed = subprocess.run(
            [
                executable,
                "--json",
                "--fixed-strings",
                "--ignore-case",
                "--max-count",
                str(limit),
                "--",
                query,
                str(folder / "paragraphs.txt"),
            ],
            capture_output=True,
            timeout=10,
            check=False,
        )
        if completed.returncode not in (0, 1):
            raise RuntimeError("Registered literal search failed")
        hits = []
        for line in completed.stdout.splitlines():
            event = json.loads(line)
            if event["type"] == "match":
                data = event["data"]
                row = graph["paragraphs"][data["line_number"] - 1]
                hits.append(
                    {
                        "paragraph_id": row["paragraph_id"],
                        "section_id": row["section_id"],
                        "paper_sha256": paper_sha256,
                        "spans": row["spans"],
                        "matched_byte_ranges": [[m["start"], m["end"]] for m in data["submatches"]],
                        "preview": row["text"][:200],
                        "assurance": "NAVIGATION_ONLY_READ_ORIGINAL_REQUIRED",
                    }
                )
        return {
            "index_identity": graph["identity"],
            "matches": hits,
            "truncated": len(hits) == limit,
            "status": "LOCATED" if hits else "NOT_LOCATED",
        }

    def read(self, paper_sha256, paragraph_ids, *, neighbors=1, max_bytes=6000):
        _, graph = self._graph(paper_sha256)
        allowed = {p["paragraph_id"]: i for i, p in enumerate(graph["paragraphs"])}
        if (
            not isinstance(paragraph_ids, list)
            or len(paragraph_ids) > 20
            or any(p not in allowed for p in paragraph_ids)
        ):
            raise ValueError("Unknown paragraph ID")
        if type(neighbors) is not int or not 0 <= neighbors <= 2 or not 1 <= max_bytes <= 24000:
            raise ValueError("Invalid bounded original read")
        indices = sorted(
            {
                j
                for pid in paragraph_ids
                for j in range(
                    max(0, allowed[pid] - neighbors),
                    min(len(allowed), allowed[pid] + neighbors + 1),
                )
                if graph["paragraphs"][j]["section_id"]
                == graph["paragraphs"][allowed[pid]]["section_id"]
            }
        )
        selected, omitted, used = [], [], 0
        for i in indices:
            row = graph["paragraphs"][i]
            size = len(row["text"].encode())
            if used + size > max_bytes:
                omitted.append(row["paragraph_id"])
                continue
            selected.append(row)
            used += size
        return {
            "index_identity": graph["identity"],
            "paper_sha256": paper_sha256,
            "paragraphs": selected,
            "omitted_paragraph_ids": omitted,
            "content_bytes": used,
        }

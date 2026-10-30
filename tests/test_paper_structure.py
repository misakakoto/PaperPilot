import json

import pytest

from paperpilot_agentic.paper_structure import (
    PaperTextNavigator,
    digest,
    graph_identity,
    recognized_heading,
    save_json,
    split_paragraph,
    structure_from_objects,
)


def obj(text, page=1, kind="text", heading=None):
    return {
        "text": text,
        "page": page,
        "bbox": [20, 30, 500, 100],
        "object_id": digest((text + str(page)).encode()),
        "object_type": kind,
        "section_path": [heading] if heading else [],
        "parser": "fixture",
    }


def test_body_sentence_is_not_heading_and_cross_page_mapping_is_preserved():
    bad = "model the relationship between all the pixels of the whole"
    graph = structure_from_objects(
        "a" * 64,
        "Paper",
        [
            obj("1. Introduction", heading="1. Introduction"),
            obj(bad, heading=bad),
            obj("continued on another page.", page=2),
            obj("2. Methods", page=2, heading="2. Methods"),
            obj("A factual paragraph.", page=2),
        ],
    )
    assert [s["title"] for s in graph["sections"]] == ["1. Introduction", "2. Methods"]
    merged = graph["paragraphs"][1]
    assert [s["page"] for s in merged["spans"]] == [1, 2]
    assert bad in merged["text"] and "continued" in merged["text"]
    assert not recognized_heading(bad, bad)
    assert recognized_heading("III. EXPERIMENTS", "III. EXPERIMENTS")


def test_long_paragraph_is_lossless_and_objects_are_not_promoted_to_facts():
    text = ("An original complete sentence. " * 200).strip()
    assert "".join(text[a:b] for a, b in split_paragraph(text, 500)) == text
    graph = structure_from_objects(
        "a" * 64, "Paper", [obj(text), obj("Table III", kind="table")], limit=500
    )
    assert all(len("\n\n".join(p["text"] for p in c["parts"])) <= 500 for c in graph["chunks"])
    assert graph["sections"][0]["heading_status"] == "UNKNOWN"
    assert graph["objects"][0]["status"] == "LOCATOR_ONLY_NOT_FACT_VERIFIED"


def registered(tmp_path):
    root = tmp_path / "pilot"
    sha = "a" * 64
    folder = root / "papers" / sha
    folder.mkdir(parents=True)
    graph = structure_from_objects(
        sha,
        "Paper",
        [
            obj("2. Methods", heading="2. Methods"),
            obj("The exact module Alpha[1] uses Eq. (2)."),
            obj("The constraints are unchanged."),
        ],
    )
    content = ("\n".join(p["text"] for p in graph["paragraphs"]) + "\n").encode()
    (folder / "paragraphs.txt").write_bytes(content)
    graph["text_file_sha256"] = digest(content)
    graph["identity"] = graph_identity(graph)
    save_json(folder / "structure.json", graph)
    return root, sha


def test_literal_rg_is_navigation_then_bounded_original_read(tmp_path):
    root, sha = registered(tmp_path)
    nav = PaperTextNavigator(root)
    result = nav.find_text(sha, "Alpha[1]")
    assert result["status"] == "LOCATED" and len(result["matches"]) == 1
    assert result["matches"][0]["assurance"].startswith("NAVIGATION_ONLY")
    assert nav.find_text(sha, "Alpha.*")["status"] == "NOT_LOCATED"
    read = nav.read(sha, [result["matches"][0]["paragraph_id"]])
    assert any("Alpha[1]" in p["text"] for p in read["paragraphs"])
    assert nav.read(sha, ["paragraph_1"], max_bytes=1)["omitted_paragraph_ids"]
    with pytest.raises(ValueError):
        nav.find_text("../elsewhere", "secret")
    with pytest.raises(ValueError):
        nav.read(sha, ["invented"])
    (root / "papers" / sha / "paragraphs.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        nav.find_text(sha, "tampered")


def test_rewriting_graph_text_and_checksum_still_requires_frozen_asset_hash(tmp_path):
    root, sha = registered(tmp_path)
    folder = root / "papers" / sha
    save_json(
        root / "index.json",
        {
            "papers": [
                {
                    "sha256": sha,
                    "structure_sha256": digest((folder / "structure.json").read_bytes()),
                }
            ]
        },
    )
    graph = json.loads((folder / "structure.json").read_bytes())
    graph["title"] = "forged"
    graph["identity"] = graph_identity(graph)
    save_json(folder / "structure.json", graph)
    with pytest.raises(ValueError, match="frozen index"):
        PaperTextNavigator(root).find_text(sha, "Alpha")


def test_real_pdf_locator_build_without_rendering_caption_regions(tmp_path):
    from paperpilot_agentic.layout import fitz
    from paperpilot_agentic.paper_structure import build_structure

    source = tmp_path / "paper.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((40, 60), "1. Introduction", fontsize=16)
        page.insert_text((40, 100), "Original method paragraph.", fontsize=10)
        page.insert_text((40, 140), "Table III", fontsize=10)
        pdf.save(source)
    doc, chunks, graph = build_structure(source, tmp_path / "asset", title="Paper", paper_id="P1")
    assert doc.page_count == 1 and chunks
    assert any(o["kind"] == "table" for o in graph["objects"])
    assert not list((tmp_path / "asset").rglob("*.png"))


def test_registered_windows_newlines_preserve_hash_and_literal_locations(tmp_path):
    root, sha = registered(tmp_path)
    folder = root / "papers" / sha
    text = (folder / "paragraphs.txt").read_bytes().replace(b"\n", b"\r\n")
    (folder / "paragraphs.txt").write_bytes(text)
    graph = json.loads((folder / "structure.json").read_bytes())
    graph["text_file_sha256"] = digest(text)
    graph["identity"] = graph_identity(graph)
    save_json(folder / "structure.json", graph)
    assert PaperTextNavigator(root).find_text(sha, "Alpha[1]")["status"] == "LOCATED"

"""Offline source-span retrieval ablation; never grade answers from page hits."""

import argparse
import json
import re
import time
import unicodedata
import sqlite3
from pathlib import Path

from evals.interview.cases import CHANGE, FLOW
from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
from paperpilot_agentic.config import RuntimeSettings
from paperpilot_agentic.layout import fitz
from paperpilot_agentic.paper_structure import digest, save_json
from paperpilot_agentic.read_packet import build_read_packet
from paperpilot_agentic.source_navigation import locate_requested_objects
from paperpilot_agentic.structured_retrieval import ReadOnlyPilotStore, StructuredPilotAdapter
from paperpilot_retrieval import PaperKnowledgeStore
from paperpilot_retrieval.models import PaperChunk, PaperDocument


def controlled_baseline(output, settings):
    path = output / "baseline-twelve.sqlite3"
    if path.exists():
        return path
    sources = json.loads((output / "sources.json").read_bytes())["papers"]
    docs, chunks, expected = [], [], set()
    with sqlite3.connect(settings.memory_db.as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        for i, paper in enumerate(sources, 1):
            metadata = connection.execute(
                "select * from papers where namespace=? and sha256=? and active=1",
                (settings.memory_namespace, paper["sha256"]),
            ).fetchone()
            rows = connection.execute(
                "select c.* from chunks c where namespace=? and paper_row_id=? order by row_id",
                (settings.memory_namespace, metadata["row_id"]),
            ).fetchall()
            docs.append(
                PaperDocument(
                    id=f"P{i}",
                    sha256=paper["sha256"],
                    title=metadata["title"],
                    file_name=metadata["file_name"],
                    page_count=metadata["page_count"],
                    processed_pages=0,
                    pages=[],
                )
            )
            for row in rows:
                label = (
                    str(row["page_start"])
                    if row["page_start"] == row["page_end"]
                    else f"{row['page_start']}-p{row['page_end']}"
                )
                chunks.append(
                    PaperChunk(
                        id=f"P{i}:p{label}:c{row['ordinal']}",
                        paper_id=f"P{i}",
                        page_start=row["page_start"],
                        page_end=row["page_end"],
                        section=row["section"],
                        text=row["text"],
                        token_count_estimate=row["token_count_estimate"],
                    )
                )
                expected.add(row["stable_id"])
    store = PaperKnowledgeStore(path, namespace=settings.memory_namespace)
    store.sync_documents(docs, chunks)
    with store._connect() as connection:
        actual = {r[0] for r in connection.execute("select stable_id from chunks")}
    if actual != expected:
        raise ValueError("Controlled baseline changed original chunk identities")
    save_json(
        output / "baseline-control.json",
        {
            "documents": len(docs),
            "chunks": len(chunks),
            "same_source_chunk_ids": True,
            "same_twelve_pdfs": True,
            "source_library_readonly": True,
            "note": "Native text extraction and structure both differ; not a pure segmentation-only ablation",
        },
    )
    return path


def normal(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def anchor_match(text, anchor):
    if anchor.startswith("TABLE "):
        return bool(re.search(r"\bTABLE\s+" + anchor.split()[1] + r"\b", text, re.I))
    return normal(anchor) in normal(text)


def freeze_cases(output):
    path = output / "questions.json"
    if path.exists():
        return json.loads(path.read_bytes())
    sources = json.loads((output / "sources.json").read_bytes())["papers"]
    questions = json.loads((output / "original-eleven-questions.json").read_bytes())
    known = {
        "single_changemamba_detail": [
            (CHANGE, 7, ["Sequential", "Parallel"]),
            (CHANGE, 16, ["TABLE XII"]),
        ],
        "multi_change_flow_comparison": [
            (CHANGE, 7, ["Sequential", "Parallel"]),
            (CHANGE, 16, ["TABLE XII"]),
            (FLOW, 4, ["alignment"]),
            (FLOW, 11, ["TABLE VII"]),
        ],
        "followup_alignment_inference": [
            (CHANGE, 7, ["Sequential", "Parallel"]),
            (FLOW, 4, ["alignment"]),
        ],
        "unseen-backbone_tradeoff": [(CHANGE, 17, ["TABLE XIII"])],
        "unseen-backbone_budget": [(CHANGE, 17, ["TABLE XIII"])],
        "unseen-disaster_missing_class": [(FLOW, 11, ["TABLE VIII"])],
        "unseen-disaster_metric": [(FLOW, 11, ["TABLE VIII"])],
        "unseen-alignment_loss": [(FLOW, 4, ["alignment"])],
        "unseen-ssm_discretization": [(CHANGE, 4, ["timescale"])],
        "unseen-deployment_limits": [(FLOW, 11, ["Limitations"])],
        "unseen-robustness_conditions": [(CHANGE, 17, ["Gaussian"])],
    }
    heldout_specs = [
        (
            "CCR-Net如何通过cross-channel reconstruction进行跨模态信息交换？说明它使用的两组模态组合。",
            ["cross-channel", "reconstruction"],
        ),
        (
            "这篇光学变化检测综述如何比较传统CD与deep learning方法，并概括datasets与challenges？",
            ["review", "datasets"],
        ),
        (
            "该多分辨率配准方法如何组合stochastic gradient、mutual information和wavelet pyramid？作者比较了什么相似性度量？",
            ["mutual information", "gradient"],
        ),
        (
            "结构循环一致性回归包含哪两部分？forward、cycle与sparse三类约束的用途是什么？",
            ["cycle", "regression"],
        ),
        ("FAT与CGF两步框架分别负责什么？如何利用SAR辅助thick cloud removal？", ["FAT", "CGF"]),
        (
            "EBD数据集针对building damage assessment的哪些数据构建困难？semi-supervised fine-tuning扮演什么角色？",
            ["EBD", "dataset"],
        ),
        (
            "D3PM的unconditional DDPM和conditional DDPM两条流分别承担什么任务？",
            ["unconditional", "conditional"],
        ),
        (
            "MSCD-Net如何联合semantic与difference features，SDD模块解决什么问题？",
            ["semantic", "difference"],
        ),
        (
            "这种structure consistency无监督回归方法如何通过similarity graph与Markovian segmentation获得变化区域？",
            ["structure", "graph"],
        ),
        (
            "Conditional Diffusion如何以SAR为条件进行SAR-to-optical转换？GF3和SEN12上使用哪些评测指标？",
            ["GF3", "SEN12"],
        ),
    ]
    for i, (question, anchors) in enumerate(heldout_specs):
        source = sources[i + 2]
        case = {
            "id": f"heldout-{i + 1:02d}",
            "question": question,
            "paper_ids": [source["sha256"]],
            "original_demo": False,
            "heldout": True,
        }
        questions.append(case)
        known[case["id"]] = [(source["sha256"], 1, anchors)]
    # Three object/navigation questions are frozen from originals, not new-index ranks.
    extras = [
        (
            sources[2],
            "table",
            r"\bTABLE\s+I\b",
            "该文Table I的表题是什么？读取表格关联的原文位置，不将表题当成所有单元格已正确解析。",
        ),
        (
            sources[5],
            "equation",
            r"\(1\)",
            "该文式(1)附近解释了哪些量？保留公式和紧邻定义，不自行推导缺失的符号。",
        ),
        (
            sources[4],
            "context",
            r"multiresolution",
            "该文multiresolution pyramid与优化过程如何关联？补读命中段落的上下文，区分方法陈述与实验效果。",
        ),
    ]
    for i, (source, kind, pattern, question) in enumerate(extras):
        page = next(
            (
                p["page"]
                for p in json.loads(
                    (output / "original-pages" / (source["sha256"] + ".json")).read_bytes()
                )
                if re.search(pattern, p["text"], re.I)
            ),
            None,
        )
        case = {
            "id": f"heldout-object-{i + 1}",
            "question": question,
            "paper_ids": [source["sha256"]],
            "original_demo": False,
            "heldout": True,
            "object_kind": kind,
        }
        questions.append(case)
        known[case["id"]] = [
            (
                source["sha256"],
                page,
                [
                    "TABLE I"
                    if kind == "table"
                    else "(1)"
                    if kind == "equation"
                    else "multiresolution"
                ],
            )
        ]
    paths = {p["sha256"]: p["path"] for p in sources}
    for case in questions:
        gold = []
        for sha, page, anchors in known[case["id"]]:
            found = []
            if page:
                with fitz.open(paths[sha]) as pdf:
                    for block in pdf[page - 1].get_text("blocks"):
                        text = block[4].strip()
                        if all(anchor_match(text, a) for a in anchors):
                            found.append(
                                {
                                    "paper_sha256": sha,
                                    "page": page,
                                    "bbox": list(block[:4]),
                                    "quote": re.sub(r"\s+", " ", text)[:400],
                                    "anchors": anchors,
                                }
                            )
            gold.append(
                found[0]
                if found
                else {
                    "paper_sha256": sha,
                    "page": page,
                    "anchors": anchors,
                    "annotation_status": "NOT_LOCATED_IN_NATIVE_TEXT_REQUIRES_REVIEW",
                }
            )
        case["gold_source_spans"] = gold
    assert len(questions) == 24
    save_json(path, questions)
    save_json(
        output / "annotation-protocol.json",
        {
            "reviewer_kind": "AI-assisted original text location",
            "raw_gold_not_answer_correctness": True,
            "gold_basis": "PDF blocks independently read before ranking",
            "source_span_recall_definition": "Fraction of frozen quoted source fragments found in top eight; missing annotations excluded and counted explicitly",
            "warning": "Table caption location does not prove table value/condition extraction; full answer quality is separate",
            "questions_sha256": digest(path.read_bytes()),
        },
    )
    return questions


def run(output, settings):
    cases = freeze_cases(output)
    baseline_store = ReadOnlyPilotStore(controlled_baseline(output, settings), {})
    baseline_store.namespace = settings.memory_namespace
    baseline = PaperPilotKnowledgeSearchAdapter(baseline_store, paper_roots=settings.paper_roots)
    structured = StructuredPilotAdapter(settings.workspace / "structured-pilot", baseline)
    rows = []
    for arm in ("legacy", "structured", "structured_grep"):
        adapter = baseline if arm == "legacy" else structured
        for case in cases:
            start = time.perf_counter()
            hits = list(
                adapter.search(case["question"], top_k=8, paper_sha256s=set(case["paper_ids"]))
            )
            if arm == "structured_grep":
                extra = []
                for sha in case["paper_ids"]:
                    extra.extend(
                        locate_requested_objects(
                            adapter.store, sha, adapter.store.paper_outline(sha), case["question"]
                        )
                    )
                hits = list({h["stable_id"]: h for h in extra + hits}.values())[:8]
            originals = [{**h, "evidence_id": "e_" + h["stable_id"]} for h in hits]
            packet = build_read_packet(case["question"], originals)
            gold = [g for g in case["gold_source_spans"] if g.get("quote")]

            def included(g, items):
                return any(
                    h.get("paper_sha256", g["paper_sha256"]) == g["paper_sha256"]
                    and (
                        h.get("provenance", {}).get("page_range")
                        or [h.get("page", 0), h.get("page_end", 0)]
                    )[0]
                    <= g["page"]
                    <= (
                        h.get("provenance", {}).get("page_range")
                        or [h.get("page", 0), h.get("page_end", 0)]
                    )[1]
                    and normal(g["quote"]) in normal(h["content"])
                    for h in items
                )

            rows.append(
                {
                    "arm": arm,
                    "case": case["id"],
                    "heldout": case.get("heldout", False),
                    "gold_count": len(gold),
                    "unannotated_targets": len(case["gold_source_spans"]) - len(gold),
                    "source_span_hits_at_8": sum(included(g, hits) for g in gold),
                    "packet_source_span_hits": sum(included(g, packet.rows) for g in gold),
                    "elapsed_ms": round((time.perf_counter() - start) * 1000, 3),
                    "packet_bytes": packet.metadata["context_upper_bytes"],
                    "packet_identity": packet.identity,
                    "actual_evidence_ids": [h["evidence_id"] for h in packet.rows],
                }
            )
    result = {
        "retrieval_control": "BM25_ONLY_BOTH_ARMS; vector provenance unavailable, not production hybrid comparison",
        "rows": rows,
        "new_provider_requests": 0,
        "new_gpu_starts": 0,
        "live_qa_gate": "NOT_RUN_FULL_HYBRID_BASELINE_UNAVAILABLE_OR_ANNOTATIONS_INCOMPLETE",
    }
    result["evaluation_version"] = "source-span-v3-equal-twelve-corpus-object-links"
    result["pilot_identity"] = structured.index_identity
    result["condition_completeness"] = (
        "NOT_ESTABLISHED; caption/fragment location is not complete experimental conditions"
    )
    save_json(output / "retrieval-results-v3.json", result)
    print(
        json.dumps(
            {
                "cases": len(rows),
                "new_provider_requests": 0,
                "live_qa_gate": result["live_qa_gate"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--output", type=Path, default=Path("outputs/structure-pilot-20261020"))
    args = parser.parse_args()
    if args.command == "freeze":
        freeze_cases(args.output)
    else:
        run(args.output, RuntimeSettings.from_env())

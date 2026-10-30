"""Run the real production composition root on one QASPER paper (no gold evidence).

QASPER sections are virtual pages, not PDF page numbers. This adapter tests text
research only; figures are captions and cannot establish numerical table evidence.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from paperpilot_agentic.bootstrap import build_runtime  # noqa: E402
from paperpilot_agentic.config import RuntimeSettings  # noqa: E402
from paperpilot_retrieval import PaperKnowledgeStore  # noqa: E402
from paperpilot_retrieval.models import PaperChunk, PaperDocument, PaperPage  # noqa: E402


def paper_records(paper: dict, *, source_id: str | None = None):
    public = {k: paper[k] for k in ('title', 'abstract', 'full_text', 'figures_and_tables') if k in paper}
    encoded = json.dumps(public, ensure_ascii=False, sort_keys=True).encode()
    sections = [{'section_name': 'Abstract', 'paragraphs': [paper['abstract']]}] + paper['full_text']
    # Captions are public source text, not answer annotations or rendered tables.
    # Append virtual sections so existing full-text page ordinals stay stable.
    sections += [{'section_name': f'Figure/table caption {i} (text only)',
                  'paragraphs': [item['caption']]}
                 for i, item in enumerate(paper.get('figures_and_tables', []), 1)
                 if isinstance(item, dict) and isinstance(item.get('caption'), str)
                 and item['caption'].strip()]
    pages, chunks = [], []
    for n, section in enumerate(sections, 1):
        # QASPER contains untitled sections represented by JSON null. Keep
        # their original paragraphs; do not drop a paper or infer a title.
        section_name = section.get('section_name') or ''
        paragraphs = [str(p) for p in section['paragraphs'] if str(p).strip()]
        pages.append(PaperPage(number=n, text='\n\n'.join(paragraphs)[:100000]))
        for paragraph in paragraphs:
            for offset in range(0, len(paragraph), 10000):
                text = paragraph[offset:offset+10000]
                chunks.append(PaperChunk(id=f'P1:p{n}:c{len(chunks)+1}', paper_id='P1',
                                         page_start=n, page_end=n, section=section_name[:300],
                                         text=text, token_count_estimate=max(1, len(text)//4)))
    file_name = f'qasper-{source_id}.json' if source_id else 'qasper-paper.json'
    document = PaperDocument(id='P1', file_name=file_name, title=paper['title'],
                             sha256=hashlib.sha256(encoded).hexdigest(), page_count=len(pages),
                             processed_pages=len(pages), pages=pages,
                             warnings=['Virtual section ordinals, not PDF page numbers; text-only adapter'])
    return public, document, chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--question-id', required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--execute-paid', action='store_true')
    parser.add_argument('--answer-first', action='store_true',
                        help='Experimental original-query retrieval then answer/audit; not production orchestration')
    parser.add_argument('--outline-navigation', action='store_true')
    parser.add_argument('--api-reader', action='store_true',
                        help='Exercise the formal async text-reader API with real runtime accounting')
    args = parser.parse_args()
    if args.outline_navigation and not args.answer_first:
        parser.error('--outline-navigation requires --answer-first')
    if args.api_reader and args.answer_first:
        parser.error('--api-reader and --answer-first are separate evaluation paths')
    if args.workspace.exists():
        raise FileExistsError('Use an isolated new workspace')
    data = json.loads(args.dataset.read_text(encoding='utf-8-sig'))
    matches = [(pid, paper, qa) for pid, paper in data.items() for qa in paper['qas']
               if qa['question_id'] == args.question_id]
    if len(matches) != 1:
        raise ValueError('Expected exactly one question')
    pid, paper, qa = matches[0]
    public, document, chunks = paper_records(paper, source_id=pid)
    from dotenv import load_dotenv
    load_dotenv(args.env_file, override=False)
    settings = RuntimeSettings.from_env(require_api_key=args.execute_paid)
    # Preserve the existing model caches while isolating all mutable research state.
    os.environ.setdefault('PAPERPILOT_EMBEDDING_CACHE', str(settings.workspace / 'model-cache/embedding'))
    os.environ.setdefault('PAPERPILOT_RERANKER_CACHE', str(settings.workspace / 'model-cache/reranker'))
    args.workspace.mkdir(parents=True)
    corpus = args.workspace / 'corpus'
    corpus.mkdir()
    (corpus / document.file_name).write_text(json.dumps(public, ensure_ascii=False), encoding='utf-8')
    settings = dataclasses.replace(settings, workspace=args.workspace.resolve(),
                                   runtime_db=(args.workspace/'runtime.sqlite3').resolve(),
                                   memory_db=(args.workspace/'knowledge.sqlite3').resolve(),
                                   memory_namespace='qasper-case', paper_roots=(corpus.resolve(),))
    store = PaperKnowledgeStore(settings.memory_db, namespace=settings.memory_namespace)
    runtime = build_runtime(settings)
    store.sync_documents([document], chunks, embedder=runtime.search_adapter.embedder)
    manifest = {'question_id': qa['question_id'], 'question': qa['question'], 'paper_id': pid,
                'adapter_version': 'qasper-text-with-captions-v2',
                'caption_text_indexed': True, 'figure_images_loaded': False,
                'corpus_sha256': document.sha256, 'chunks': len(chunks),
                'gold_answers_sent': False, 'gold_evidence_sent': False,
                'model': settings.model, 'dense_enabled': runtime.search_adapter.embedder is not None,
                'reranker_enabled': runtime.search_adapter.reranker is not None,
                'api_reader': args.api_reader,
                'boundary': ('single-paper text-only experimental retrieval/reader/audit; no production orchestration'
                             if args.answer_first else
                             'single-paper text-only production runtime; section ordinals are virtual')}
    (args.workspace/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if args.api_reader and args.execute_paid:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from paperpilot_agentic.web_api import create_fastapi_router
        app = FastAPI()
        app.include_router(create_fastapi_router(runtime_factory=lambda: runtime))
        with TestClient(app) as client:
            response = client.post('/agentic/runs', json={
                'question': qa['question'], 'engine': 'answer_first_text',
                'metadata': {'paper_sha256s': [document.sha256], 'benchmark': 'qasper-dev',
                             'question_id': qa['question_id'], 'page_numbering': 'qasper_section_ordinal'}})
            response.raise_for_status()
            if response.status_code != 202:
                raise ValueError('Expected asynchronous API acceptance')
            pending = response.json()
            run_id = pending['run']['run_id']
            (args.workspace/'pending.json').write_text(json.dumps(pending, ensure_ascii=False), encoding='utf-8')
            terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'BUDGET_EXHAUSTED'}
            deadline = time.monotonic() + pending['run']['budget']['max_seconds'] + 30
            while True:
                current = client.get(f'/agentic/runs/{run_id}')
                current.raise_for_status()
                result = current.json()
                temp = args.workspace/'api-result.pending'
                temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
                temp.replace(args.workspace/'result.json')
                if result['run']['status'] in terminal:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('Observation deadline exceeded; inspect saved run, do not restart')
                time.sleep(1)
        reader = result['run']['metadata'].get('reader_report', {})
        artifact = {'question': qa['question'], 'run_id': run_id,
                    'evaluation_scope': 'FORMAL_ASYNC_API_IN_PROCESS_REAL_PROVIDER_TEXT_ONLY',
                    'reader': {'provider_attempts': 0, 'calls': [], **reader},
                    'persisted_run': result['run'], 'evidence': result['evidence']}
        (args.workspace/'reader-result.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'run_id': run_id, 'status': result['run']['status'],
                          'calls': result['run']['used_llm_calls'], 'tokens': result['run']['used_tokens']}))
        return
    if args.answer_first:
        from paperpilot_agentic.providers import OpenAICompatibleLLM
        from paperpilot_agentic.durable_reader import (
            DurableResearchModel, reader_completion_state, publish_reader_answer, run_durable_reader)
        from paperpilot_agentic.llm_runtime import PersistentLLMRuntime
        from paperpilot_agentic.models import Budget, RunStatus
        durable_run_id = None
        if args.execute_paid:
            durable_run_id = runtime.db.create_run(qa['question'], Budget(), metadata={
                'mode': 'experimental-answer-first', 'question_id': qa['question_id'],
                'factual_review': 'NOT_INDEPENDENTLY_EVALUATED'})
            runtime.db.update_run_status(durable_run_id, RunStatus.RUNNING)
            llm_runtime = PersistentLLMRuntime(runtime.db, OpenAICompatibleLLM(settings, max_retries=0))
            reader_model = DurableResearchModel(llm_runtime, durable_run_id, stage='answer_first',
                                                reserve_class='synthesis')
            router_model = DurableResearchModel(llm_runtime, durable_run_id, stage='section_router')
        if durable_run_id:
            runtime.db.increment_usage(durable_run_id, tool_calls=1)
        hits = runtime.search_adapter.search(qa['question'], top_k=8,
                                             paper_sha256s={document.sha256})
        evidence = [{**hit, 'evidence_id': 'e_' + hit['stable_id']} for hit in hits]
        artifact = {'mode': 'experimental-original-query-answer-audit',
                    'question': qa['question'], 'evidence': evidence,
                    'run_id': durable_run_id,
                    'tool_calls': 1, 'production_orchestration': False,
                    'gold_evidence_sent': False, 'gold_answers_sent': False}

        def save_reader(report):
            if durable_run_id:
                runtime.db.update_run_metadata(durable_run_id, {'reader_progress': report['status']})
            payload = {**artifact, 'reader': report}
            pending_path = args.workspace/'reader-result.pending'
            pending_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
            pending_path.replace(args.workspace/'reader-result.json')

        def finalize_reader(report):
            if durable_run_id:
                status, outcome = reader_completion_state(report)
                publish_reader_answer(runtime.db, durable_run_id, report)
                runtime.db.update_run_outcome(durable_run_id, outcome)
                runtime.db.update_run_status(durable_run_id, status, error=report.get('error_type'))
                artifact['persisted_run'] = runtime.db.get_run(durable_run_id)
            save_reader(report)

        if args.execute_paid:
            if args.outline_navigation:
                from paperpilot_agentic.section_router import select_sections, read_selected_sections, structural_overviews
                runtime.db.increment_usage(durable_run_id, tool_calls=1)
                outline = store.paper_outline(document.sha256)
                artifact['outline'] = outline

                def save_routing(routing):
                    artifact['routing'] = routing
                    save_reader({'status': 'WAITING_ROUTING', 'provider_attempts': 0, 'calls': []})

                routing = select_sections(qa['question'], outline,
                                          router_model, on_progress=save_routing)
                if routing['status'] == 'FAILED':
                    finalize_reader({'status': 'ROUTING_FAILED', 'provider_attempts': 0, 'calls': []})
                    print(json.dumps({'status': 'ROUTING_FAILED', 'attempts': routing['provider_attempts']}))
                    return
                overview_ids = structural_overviews(outline)
                artifact['structural_overview_section_ids'] = overview_ids
                selected_ids = list(dict.fromkeys(overview_ids + routing['section_ids']))
                runtime.db.increment_usage(durable_run_id, tool_calls=1)
                extra, omitted = read_selected_sections(store, document.sha256, outline, selected_ids)
                artifact['initial_evidence'] = evidence
                by_id = {e['evidence_id']: e for e in evidence}
                by_id.update({e['evidence_id']: e for e in extra})
                evidence = list(by_id.values())
                artifact.update(evidence=evidence, omitted_section_chunks=omitted, tool_calls=3)
            report = run_durable_reader(runtime.db, durable_run_id,
                [{**item, 'page_numbering': 'qasper_section_ordinal'} for item in evidence],
                reader_model, protocol_id=f'qasper-text-answer-audit-v2:{settings.model}',
                on_progress=save_reader)
            artifact['evidence'] = runtime.db.list_evidence(durable_run_id)
            artifact['persisted_run'] = runtime.db.get_run(durable_run_id)
            save_reader(report)
        else:
            report = {'status': 'DRY_RUN', 'provider_attempts': 0}
            save_reader(report)
        print(json.dumps({'workspace': str(args.workspace), 'status': report['status'],
                          'attempts': report['provider_attempts'] + artifact.get('routing', {}).get('provider_attempts', 0)}))
        return
    if not args.execute_paid:
        manifest['preflight_retrieved_count'] = len(runtime.search_adapter.search(qa['question'], top_k=8))
        print(json.dumps(manifest))
        return
    pending = runtime.enqueue(question=qa['question'], metadata={
        'benchmark': 'qasper-dev', 'paper_id': pid, 'paper_title': document.title,
        'paper_file_name': document.file_name, 'paper_sha256': document.sha256})
    run_id = pending['run']['run_id']
    (args.workspace/'pending.json').write_text(json.dumps(pending, ensure_ascii=False, default=str), encoding='utf-8')
    result = runtime.resume(run_id)
    (args.workspace/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print(json.dumps({'run_id': run_id, 'workspace': str(args.workspace), 'status': result['run']['status']}))


if __name__ == '__main__':
    main()

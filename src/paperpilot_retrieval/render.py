"""Deterministic Markdown, Mermaid, and evidence auditing for paper artifacts."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from io import StringIO

from .models import (
    ComponentCodeMapping,
    EvidencePoint,
    PaperAnalysis,
    PaperAnswer,
    PaperAnswerAudit,
    PaperChunk,
    PaperCitationAudit,
    PaperDocument,
    PaperProfile,
)


def audit_paper_analysis(
    analysis: PaperAnalysis, chunks: list[PaperChunk]
) -> PaperCitationAudit:
    valid = {chunk.id for chunk in chunks}
    rules = list(_analysis_binding_rules(analysis))
    bindings = [ids for ids, _, _, _ in rules]
    cited = {item for binding in bindings for item in binding}
    invalid = sorted(cited - valid, key=_evidence_sort_key)
    misbound: list[str] = []
    out_of_scope: set[str] = set()
    supported = 0
    for binding, allowed_papers, minimum_distinct_papers, allowed_scope in rules:
        existing = set(binding)
        cited_papers = {item.split(":", 1)[0] for item in binding}
        wrong_papers = cited_papers - allowed_papers if allowed_papers is not None else set()
        outside = existing - allowed_scope
        out_of_scope.update(outside)
        if wrong_papers:
            misbound.extend(
                f"expected {','.join(sorted(allowed_papers or set()))}: {item}"
                for item in binding
                if item.split(":", 1)[0] in wrong_papers
            )
        if (
            existing
            and existing.issubset(valid)
            and not wrong_papers
            and not outside
            and len(cited_papers) >= minimum_distinct_papers
        ):
            supported += 1
    coverage = supported / len(bindings) if bindings else 0.0
    profile_ids = {profile.paper_id for profile in analysis.profiles}
    if len(profile_ids) <= 1:
        cross_valid = True
    else:
        cross_valid = bool(analysis.cross_paper_comparison) and all(
            len({item.split(":", 1)[0] for item in point.evidence_ids} & profile_ids) >= 2
            for point in analysis.cross_paper_comparison
        )
    component_keys = {
        (profile.paper_id, component.name.casefold())
        for profile in analysis.profiles
        for component in profile.model_components
    }
    mapped_keys = {
        (mapping.paper_id, mapping.component_name.casefold())
        for mapping in analysis.code_mappings
    }
    mapping_coverage = (
        len(component_keys & mapped_keys) / len(component_keys) if component_keys else 1.0
    )
    expected_papers = set(analysis.document_ids)
    profiled_papers = {profile.paper_id for profile in analysis.profiles}
    profile_coverage = (
        len(expected_papers & profiled_papers) / len(expected_papers)
        if expected_papers
        else 0.0
    )
    missing_content = _missing_content_sections(analysis)
    reproduction_complete = not any(
        item.startswith("reproduction.") for item in missing_content
    )
    content_complete = profile_coverage == 1.0 and not missing_content
    evidence_valid = (
        bool(bindings)
        and not invalid
        and not misbound
        and not out_of_scope
        and coverage == 1.0
        and cross_valid
        and mapping_coverage == 1.0
    )
    return PaperCitationAudit(
        valid=evidence_valid and content_complete,
        evidence_valid=evidence_valid,
        content_complete=content_complete,
        valid_evidence_ids=sorted(valid, key=_evidence_sort_key),
        cited_evidence_ids=sorted(cited & valid, key=_evidence_sort_key),
        invalid_evidence_ids=invalid,
        misbound_evidence_ids=sorted(set(misbound)),
        out_of_scope_evidence_ids=sorted(out_of_scope, key=_evidence_sort_key),
        evidence_points=len(bindings),
        supported_evidence_points=supported,
        structured_support_coverage=coverage,
        cross_paper_comparison_valid=cross_valid,
        component_mapping_coverage=mapping_coverage,
        retrieval_scope_valid=not out_of_scope,
        reproduction_complete=reproduction_complete,
        paper_profile_coverage=profile_coverage,
        missing_content_sections=missing_content,
    )


def audit_paper_answer(answer: PaperAnswer, chunks: list[PaperChunk]) -> PaperAnswerAudit:
    """Require the answer itself and every structured key point to use retrieved evidence."""

    valid = {chunk.id for chunk in chunks}
    bindings = [answer.evidence_ids, *(point.evidence_ids for point in answer.key_points)]
    cited = {item for binding in bindings for item in binding}
    invalid = sorted(cited - valid, key=_evidence_sort_key)
    supported = sum(bool(binding) and set(binding).issubset(valid) for binding in bindings)
    coverage = supported / len(bindings) if bindings else 0.0
    has_supported_answer = bool(answer.evidence_ids) and set(answer.evidence_ids).issubset(valid)
    return PaperAnswerAudit(
        valid=has_supported_answer and not invalid and coverage == 1.0,
        valid_evidence_ids=sorted(valid, key=_evidence_sort_key),
        cited_evidence_ids=sorted(cited & valid, key=_evidence_sort_key),
        invalid_evidence_ids=invalid,
        evidence_points=len(bindings),
        supported_evidence_points=supported,
        support_coverage=coverage,
        has_supported_answer=has_supported_answer,
    )


def _missing_content_sections(analysis: PaperAnalysis) -> list[str]:
    missing: list[str] = []
    profile_by_id = {profile.paper_id: profile for profile in analysis.profiles}
    for paper_id in sorted(set(profile_by_id) - set(analysis.document_ids)):
        missing.append(f"{paper_id}.unexpected_profile")
    for paper_id in analysis.document_ids:
        profile = profile_by_id.get(paper_id)
        if profile is None:
            missing.append(f"{paper_id}.profile")
            continue
        if not profile.prerequisites:
            missing.append(f"{paper_id}.prerequisites")
        if not profile.contributions:
            missing.append(f"{paper_id}.contributions")
        if not profile.learning_cards:
            missing.append(f"{paper_id}.learning_cards")
        if profile.model_section_status == "present" and not profile.training_objectives:
            missing.append(f"{paper_id}.training_objectives")
        if profile.paper_id not in analysis.profile_evidence_scopes:
            missing.append(f"{paper_id}.retrieval_scope")
    plan = analysis.reproduction
    for field_name in (
        "suggested_project_tree",
        "environment",
        "data_preparation",
        "steps",
        "acceptance_checks",
        "risks",
    ):
        if not getattr(plan, field_name):
            missing.append(f"reproduction.{field_name}")
    if plan.readiness in {"partial", "blocked"} and not plan.missing_information:
        missing.append("reproduction.missing_information")
    return missing


def render_architecture_mermaid(profiles: list[PaperProfile]) -> str:
    lines = ["flowchart LR"]
    for paper_index, profile in enumerate(profiles, 1):
        lines.append(f'    subgraph paper_{paper_index}["{_label(profile.paper_id + " " + profile.title)}"]')
        names: dict[str, str] = {}
        for component_index, component in enumerate(profile.model_components, 1):
            node_id = f"p{paper_index}_m{component_index}"
            names[component.name.casefold()] = node_id
            citation = _page_label(component.evidence_ids)
            suffix = f"\\n{citation}" if citation else ""
            lines.append(
                f'        {node_id}["{_label(component.name)}\\n{_label(component.role[:100])}{suffix}"]'
            )
        for component_index, component in enumerate(profile.model_components, 1):
            target = f"p{paper_index}_m{component_index}"
            for upstream in component.depends_on:
                source = names.get(upstream.casefold())
                if source:
                    lines.append(f'        {source} -->|"data flow"| {target}')
        lines.append("    end")
    if len(lines) == 1:
        lines.append('    empty["未提取到可验证的模型组件"]')
    return "\n".join(lines) + "\n"


def render_mindmap_mermaid(profiles: list[PaperProfile]) -> str:
    lines = ["mindmap", "  root((论文学习地图))"]
    for index, profile in enumerate(profiles, 1):
        lines.append(f'    p{index}["{_label(profile.paper_id + " " + profile.title)}"]')
        lines.append(f'      problem{index}["问题: {_label(profile.research_problem.text[:120])}"]')
        if profile.contributions:
            lines.append(f'      contribution_group{index}["核心贡献"]')
            for item_index, contribution in enumerate(profile.contributions[:6], 1):
                lines.append(
                    f'        contribution{index}_{item_index}["{_label(contribution.text[:120])}"]'
                )
        if profile.model_components:
            lines.append(f'      model_group{index}["模型组件"]')
            for item_index, component in enumerate(profile.model_components[:10], 1):
                lines.append(
                    f'        component{index}_{item_index}["{_label(component.name)}"]'
                )
        if profile.experiments:
            lines.append(f'      experiment_group{index}["实验设计"]')
            for item_index, experiment in enumerate(profile.experiments[:6], 1):
                lines.append(
                    f'        experiment{index}_{item_index}["{_label(experiment.question[:120])}"]'
                )
        if profile.limitations:
            lines.append(f'      limitation_group{index}["局限与未知"]')
            for item_index, limitation in enumerate(profile.limitations[:4], 1):
                lines.append(
                    f'        limitation{index}_{item_index}["{_label(limitation.text[:120])}"]'
                )
    return "\n".join(lines) + "\n"


def render_code_map_mermaid(
    profiles: list[PaperProfile], mappings: list[ComponentCodeMapping]
) -> str:
    lines = ["flowchart LR", '    subgraph paper["论文模型组件"]']
    component_nodes: dict[tuple[str, str], str] = {}
    component_counter = 0
    for profile in profiles:
        for component in profile.model_components:
            component_counter += 1
            node = f"component_{component_counter}"
            component_nodes[(profile.paper_id, component.name.casefold())] = node
            lines.append(
                f'        {node}["{_label(profile.paper_id + " · " + component.name)}"]'
            )
    for mapping in mappings:
        key = (mapping.paper_id, mapping.component_name.casefold())
        if key in component_nodes:
            continue
        component_counter += 1
        node = f"component_{component_counter}"
        component_nodes[key] = node
        lines.append(
            f'        {node}["{_label(mapping.paper_id + " · " + mapping.component_name)} '
            '(训练/评测概念)"]'
        )
    if component_counter == 0:
        lines.append('        no_component["未识别模型组件"]')
    lines.append("    end")
    lines.append('    subgraph code["只读代码符号索引"]')
    code_nodes: dict[str, str] = {}
    missing_nodes: list[tuple[tuple[str, str], str]] = []
    for map_index, mapping in enumerate(mappings, 1):
        if mapping.code_refs:
            for ref in mapping.code_refs:
                if ref not in code_nodes:
                    node = f"code_{len(code_nodes) + 1}"
                    code_nodes[ref] = node
                    lines.append(f'        {node}["{_label(ref)}"]')
        else:
            node = f"missing_{map_index}"
            missing_nodes.append(
                ((mapping.paper_id, mapping.component_name.casefold()), node)
            )
            lines.append(f'        {node}["未找到已验证代码映射"]')
    if not mappings:
        lines.append('        no_repo["未提供代码仓库"]')
    lines.append("    end")

    for mapping in mappings:
        component = component_nodes.get(
            (mapping.paper_id, mapping.component_name.casefold())
        )
        if not component:
            continue
        for ref in mapping.code_refs:
            code = code_nodes[ref]
            lines.append(f'    {component} -->|"{mapping.status}"| {code}')
    for component_name, missing in missing_nodes:
        component = component_nodes.get(component_name)
        if component:
            lines.append(f"    {component} -.-> {missing}")
    lines.append("    classDef missing fill:#fff1f0,stroke:#cf1322,stroke-dasharray: 5 5")
    if missing_nodes:
        lines.append("    class " + ",".join(node for _, node in missing_nodes) + " missing")
    return "\n".join(lines) + "\n"


def render_experiment_mermaid(profiles: list[PaperProfile]) -> str:
    lines = ["flowchart LR"]
    counter = 0
    for profile in profiles:
        for experiment in profile.experiments:
            counter += 1
            prefix = f"e{counter}"
            baselines = ", ".join(experiment.baselines[:3]) or "未明确"
            metrics = ", ".join(experiment.metrics[:3]) or "未明确"
            lines.extend(
                [
                    f'    {prefix}_data["{_label(experiment.dataset)}"]',
                    f'    {prefix}_setup["{_label(experiment.setup[:100])}"]',
                    f'    {prefix}_base["Baselines: {_label(baselines)}"]',
                    f'    {prefix}_metric["Metrics: {_label(metrics)}"]',
                    f'    {prefix}_result["{_label(experiment.result[:100])}"]',
                    f"    {prefix}_data --> {prefix}_setup --> {prefix}_base --> {prefix}_metric --> {prefix}_result",
                ]
            )
    if not counter:
        lines.append('    empty["论文片段中未提取到实验设计"]')
    return "\n".join(lines) + "\n"


def render_learning_cards_tsv(analysis: PaperAnalysis) -> str:
    """Export active-recall cards as a spreadsheet-safe, Anki-mappable TSV."""

    rows: list[list[str]] = [["Front", "Back", "Tags", "Paper", "Evidence"]]
    for profile in analysis.profiles:
        tags = f"paper_{profile.paper_id} evidence_grounded"
        paper = f"{profile.paper_id} · {profile.title}"
        rows.extend(
            [
                card.question,
                card.answer,
                tags,
                paper,
                "; ".join(card.evidence_ids),
            ]
            for card in profile.learning_cards
        )
    return _render_delimited(rows, delimiter="\t")


def render_experiment_matrix_csv(analysis: PaperAnalysis) -> str:
    """Flatten experimental designs and explicit evidence gaps into a review table."""

    rows: list[list[str]] = [
        [
            "Paper ID",
            "Paper Title",
            "Section Status",
            "Status Reason",
            "Question",
            "Dataset",
            "Setup",
            "Baselines",
            "Metrics",
            "Result",
            "Interpretation",
            "Evidence",
        ]
    ]
    for profile in analysis.profiles:
        if profile.experiments:
            for experiment in profile.experiments:
                rows.append(
                    [
                        profile.paper_id,
                        profile.title,
                        profile.experiment_section_status,
                        profile.experiment_section_reason,
                        experiment.question,
                        experiment.dataset,
                        experiment.setup,
                        "; ".join(experiment.baselines),
                        "; ".join(experiment.metrics),
                        experiment.result,
                        experiment.interpretation,
                        "; ".join(experiment.evidence_ids),
                    ]
                )
        else:
            rows.append(
                [
                    profile.paper_id,
                    profile.title,
                    profile.experiment_section_status,
                    profile.experiment_section_reason,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )
    return _render_delimited(rows, delimiter=",")


def render_reproduction_plan_csv(analysis: PaperAnalysis) -> str:
    """Export the reproduction contract as a step-oriented tracking matrix."""

    plan = analysis.reproduction
    rows: list[list[str]] = [
        [
            "Readiness",
            "Readiness Reason",
            "Step ID",
            "Title",
            "Action",
            "Inputs",
            "Outputs",
            "Verification",
            "Evidence",
            "Global Acceptance Checks",
            "Missing Information",
            "Risks",
        ]
    ]
    steps = list(plan.steps) or [None]
    for step in steps:
        rows.append(
            [
                plan.readiness,
                plan.readiness_reason,
                step.id if step else "",
                step.title if step else "",
                step.action if step else "",
                "; ".join(step.inputs) if step else "",
                "; ".join(step.outputs) if step else "",
                step.verification if step else "",
                "; ".join(step.evidence_ids) if step else "",
                "; ".join(_plan_text(item, "CHECK:") for item in plan.acceptance_checks),
                "; ".join(
                    _plan_text(item, "TODO_UNKNOWN:")
                    for item in plan.missing_information
                ),
                "; ".join(_plan_text(item, "RISK:") for item in plan.risks),
            ]
        )
    return _render_delimited(rows, delimiter=",")


def render_paper_report(
    analysis: PaperAnalysis,
    documents: list[PaperDocument],
    chunks: list[PaperChunk],
    audit: PaperCitationAudit,
    *,
    architecture: str,
    mindmap: str,
    code_map: str,
    question: str | None,
    warnings: list[str],
) -> str:
    valid = set(audit.valid_evidence_ids)
    document_by_id = {document.id: document for document in documents}
    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    lines = [
        "# 本地论文学习与复现报告",
        "",
        "> 本报告仅依据本地 PDF 中被检索到的片段生成。页码证据表示文本片段来源，",
        "> 不代表结论已经由人工复核，也不代表建议代码已经执行或复现指标已经达到。",
        "",
    ]
    if question:
        lines.extend([f"**本次关注问题：** {_safe_text(question)}", ""])

    lines.extend(
        [
            "## 可复用学习产物",
            "",
            "- [主动回忆卡 TSV](learning-cards.tsv)：映射到 Anki Front / Back / Tags；",
            "- [实验对比矩阵 CSV](experiment-matrix.csv)：集中核对数据、基线、指标和结论；",
            "- [复现步骤矩阵 CSV](reproduction-plan.csv)：跟踪步骤、输入输出、验收和风险。",
            "",
        ]
    )

    lines.extend(["## 论文清单", ""])
    for document in documents:
        author_text = (
            "、".join(_safe_text(item) for item in document.authors)
            if document.authors
            else "PDF 元数据未提供"
        )
        lines.append(
            f"- **{document.id} · {_safe_text(document.title)}**："
            f"`{_safe_text(document.file_name)}`，"
            f"{document.processed_pages}/{document.page_count} 页已解析；作者：{author_text}"
        )
    lines.append("")

    lines.extend(["## 五分钟速览", ""])
    for profile in analysis.profiles:
        lines.extend(
            [
                f"### {profile.paper_id} · {_safe_text(profile.title)}",
                "",
                (
                    f"{_safe_text(profile.one_sentence_summary)} "
                    f"{_citations(profile.summary_evidence_ids, valid)}"
                ),
                "",
                "**研究问题**",
                "",
                _point(profile.research_problem, valid),
                "",
                "**核心贡献**",
                "",
            ]
        )
        lines.extend(f"- {_point(item, valid)}" for item in profile.contributions)
        if not profile.contributions:
            lines.append("- 当前检索片段不足，未形成有证据的贡献条目。")
        lines.append("")

    lines.extend(["## 前置知识与学习路径", ""])
    for profile in analysis.profiles:
        lines.append(f"### {profile.paper_id}")
        lines.append("")
        lines.extend(f"- {_safe_text(item)}" for item in profile.prerequisites)
        if not profile.prerequisites:
            lines.append("- 未提取到明确的前置知识建议。")
        lines.append("")

    lines.extend(["## 模型思路与数据流", "", "```mermaid", architecture.rstrip(), "```", ""])
    for profile in analysis.profiles:
        lines.extend([f"### {profile.paper_id} 模型组件", ""])
        lines.append("| 组件 | 作用 | 输入 | 输出 | 关键操作 | 论文证据 |")
        lines.append("|---|---|---|---|---|---|")
        for component in profile.model_components:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _cell(component.name),
                        _cell(component.role),
                        _cell(", ".join(component.inputs) or "未明确"),
                        _cell(", ".join(component.outputs) or "未明确"),
                        _cell("; ".join(component.operations) or "未明确"),
                        _citations(component.evidence_ids, valid) or "未绑定",
                    ]
                )
                + " |"
            )
        if not profile.model_components:
            lines.append("| 未提取到 | - | - | - | - | - |")
        lines.append("")

        lines.extend(["**训练目标/损失**", ""])
        lines.extend(f"- {_point(item, valid)}" for item in profile.training_objectives)
        if not profile.training_objectives:
            lines.append("- 当前片段未明确训练目标或损失细节。")
        lines.append("")

    lines.extend(
        [
            "## 实验设计",
            "",
            "```mermaid",
            render_experiment_mermaid(analysis.profiles).rstrip(),
            "```",
            "",
            "| 论文 | 验证问题 | 数据集 | 基线 | 指标 | 结果与解释 | 证据 |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for profile in analysis.profiles:
        for experiment in profile.experiments:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _safe_text(profile.paper_id),
                        _cell(experiment.question),
                        _cell(experiment.dataset),
                        _cell(", ".join(experiment.baselines) or "未明确"),
                        _cell(", ".join(experiment.metrics) or "未明确"),
                        _cell(f"{experiment.result}；{experiment.interpretation}"),
                        _citations(experiment.evidence_ids, valid) or "未绑定",
                    ]
                )
                + " |"
            )
    lines.append("")

    lines.extend(["## 跨论文比较", ""])
    if analysis.cross_paper_comparison:
        lines.extend(f"- {_point(item, valid)}" for item in analysis.cross_paper_comparison)
    else:
        lines.append("- 本次只有一篇论文，或现有证据不足以形成可靠的跨论文比较。")
    lines.append("")

    lines.extend(["## 论文思维导图", "", "```mermaid", mindmap.rstrip(), "```", ""])
    lines.extend(["## 模型—代码逐项关联", "", "```mermaid", code_map.rstrip(), "```", ""])
    if analysis.code_mappings:
        lines.extend(
            [
                "| 论文 | 论文组件 | 映射状态 | 已校验代码位置 | 映射依据 | 论文证据 |",
                "|---|---|---|---|---|---|",
            ]
        )
        for mapping in analysis.code_mappings:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _cell(mapping.paper_id),
                        _cell(mapping.component_name),
                        _safe_text(mapping.status),
                        _cell("; ".join(mapping.code_refs) or "未找到"),
                        _cell(mapping.rationale),
                        _citations(mapping.evidence_ids, valid) or "未绑定",
                    ]
                )
                + " |"
            )
    else:
        lines.append("未提供代码仓库；当前仅生成论文模型结构和复现计划。")
    lines.append("")

    plan = analysis.reproduction
    lines.extend(
        [
            "## 复现计划",
            "",
            (
                f"**准备度（规划判断）：{_safe_text(plan.readiness)}。** "
                f"{_safe_text(plan.readiness_reason)}"
            ),
            "",
            "### 建议工程结构",
            "",
            "```text",
            *(_safe_text(item) for item in (plan.suggested_project_tree or ["TODO_UNKNOWN"])),
            "```",
            "",
            "### 建议环境与数据（工程建议，不计入论文证据覆盖）",
            "",
        ]
    )
    lines.extend(f"- {_plan_text(item, 'SUGGESTED:')}" for item in plan.environment)
    lines.extend(
        f"- {_plan_text(item, 'SUGGESTED:')}" for item in plan.data_preparation
    )
    lines.extend(["", "### 分阶段任务", ""])
    for step in plan.steps:
        lines.extend(
            [
                f"#### {step.id} · {_safe_text(step.title)}",
                "",
                _safe_text(step.action),
                "",
                f"- 输入：{_safe_text(', '.join(step.inputs) or '无')}",
                f"- 输出：{_safe_text(', '.join(step.outputs) or '无')}",
                f"- 验收：{_safe_text(step.verification)}",
                f"- 论文证据：{_citations(step.evidence_ids, valid) or '工程建议，无直接论文证据'}",
                "",
            ]
        )
    lines.extend(["### 复现验收条件", ""])
    lines.extend(
        f"- [ ] {_plan_text(item, 'CHECK:')}" for item in plan.acceptance_checks
    )
    lines.extend(["", "### 缺失信息与风险", ""])
    lines.extend(
        f"- TODO_UNKNOWN：{_plan_text(item, 'TODO_UNKNOWN:')}"
        for item in plan.missing_information
    )
    lines.extend(f"- 风险：{_plan_text(item, 'RISK:')}" for item in plan.risks)
    lines.append("")

    lines.extend(["## 局限、未知与主动回忆", ""])
    for profile in analysis.profiles:
        lines.extend([f"### {profile.paper_id} 局限与未决问题", ""])
        lines.extend(f"- {_point(item, valid)}" for item in profile.limitations)
        lines.extend(f"- 未决：{_safe_text(item)}" for item in profile.unresolved_questions)
        lines.extend(["", "**学习卡片**", ""])
        for index, card in enumerate(profile.learning_cards, 1):
            lines.extend(
                [
                    f"{index}. **问：{_safe_text(card.question)}**",
                    f"   答：{_safe_text(card.answer)} {_citations(card.evidence_ids, valid)}",
                ]
            )
        lines.append("")

    lines.extend(
        [
            "## 证据质量审计",
            "",
            (
                f"- 结构化证据绑定：{audit.supported_evidence_points}/{audit.evidence_points} "
                f"({audit.structured_support_coverage:.1%})"
            ),
            f"- 非法证据 ID：{', '.join(audit.invalid_evidence_ids) or '0'}",
            f"- 跨论文比较绑定：{audit.cross_paper_comparison_valid}",
            f"- 模型组件映射说明覆盖率：{audit.component_mapping_coverage:.1%}",
            f"- Profile 覆盖率：{audit.paper_profile_coverage:.1%}",
            f"- 检索范围合法：{audit.retrieval_scope_valid}",
            f"- 证据合法：{audit.evidence_valid}",
            f"- 内容完整：{audit.content_complete}",
            f"- 缺失内容：{', '.join(audit.missing_content_sections) or '0'}",
            f"- 错配证据：{'; '.join(audit.misbound_evidence_ids) or '0'}",
            f"- 超出检索范围：{', '.join(audit.out_of_scope_evidence_ids) or '0'}",
            f"- 质量门：{'通过' if audit.valid else '未通过'}",
            "",
        ]
    )
    if warnings:
        lines.extend(["## 解析与运行警告", ""])
        lines.extend(
            f"- {_safe_text(warning)}" for warning in list(dict.fromkeys(warnings))
        )
        lines.append("")

    lines.extend(["## 证据目录", ""])
    for evidence_id in audit.cited_evidence_ids:
        chunk = chunk_by_id[evidence_id]
        document = document_by_id[chunk.paper_id]
        lines.append(
            f"- {_citations([evidence_id], valid)} `{_safe_text(document.file_name)}`，"
            f"SHA-256 `{document.sha256[:12]}`，第 {chunk.page_start} 页，"
            f"章节：{_safe_text(chunk.section)}"
        )
    lines.append("")
    return _finalize_markdown("\n".join(lines))


def render_answer_markdown(
    question: str,
    answer: PaperAnswer,
    chunks: list[PaperChunk],
    *,
    documents: list[PaperDocument],
    audit: PaperAnswerAudit,
    warnings: list[str] | None = None,
) -> str:
    valid = {chunk.id for chunk in chunks}
    used = set(answer.evidence_ids)
    for point in answer.key_points:
        used.update(point.evidence_ids)
    document_by_id = {document.id: document for document in documents}
    lines = [
        f"# 论文问答：{_safe_text(question)}",
        "",
        _safe_text(answer.answer),
        "",
        f"答案证据：{_citations(answer.evidence_ids, valid) or '未绑定'}",
        "",
        "## 关键依据",
        "",
    ]
    for point in answer.key_points:
        lines.append(f"- {_point(point, valid)}")
    lines.extend(["", "## 不确定性", ""])
    lines.extend(f"- {_safe_text(item)}" for item in answer.uncertainties)
    lines.extend(["", "## 推荐追问", ""])
    lines.extend(f"- {_safe_text(item)}" for item in answer.follow_up_questions)
    lines.extend(
        [
            "",
            "## 答案证据审计",
            "",
            (
                f"- 支持覆盖率：{audit.supported_evidence_points}/{audit.evidence_points} "
                f"({audit.support_coverage:.1%})"
            ),
            f"- 主答案有合法证据：{audit.has_supported_answer}",
            f"- 非法或未检索证据：{', '.join(audit.invalid_evidence_ids) or '0'}",
            f"- 质量门：{'通过' if audit.valid else '未通过'}",
        ]
    )
    if warnings:
        lines.extend(["", "## 检索与解析警告", ""])
        lines.extend(
            f"- {_safe_text(item)}" for item in list(dict.fromkeys(warnings))
        )
    lines.extend(["", "## 检索证据", ""])
    for chunk in chunks:
        if chunk.id in used:
            document = document_by_id[chunk.paper_id]
            lines.append(
                f"- {_citations([chunk.id], valid)} {document.id} · "
                f"`{_safe_text(document.file_name)}` · {_safe_text(document.title)} · "
                f"SHA-256 `{document.sha256[:12]}` · 第 {chunk.page_start} 页 · "
                f"{_safe_text(chunk.section)}"
            )
    return _finalize_markdown("\n".join(lines) + "\n")


def _analysis_binding_rules(
    analysis: PaperAnalysis,
) -> Iterable[tuple[list[str], set[str] | None, int, set[str]]]:
    for profile in analysis.profiles:
        own_paper = {profile.paper_id}
        profile_scope = set(analysis.profile_evidence_scopes.get(profile.paper_id, []))
        yield profile.summary_evidence_ids, own_paper, 1, profile_scope
        yield profile.research_problem.evidence_ids, own_paper, 1, profile_scope
        for point in profile.contributions:
            yield point.evidence_ids, own_paper, 1, profile_scope
        for component in profile.model_components:
            yield component.evidence_ids, own_paper, 1, profile_scope
        for point in profile.training_objectives:
            yield point.evidence_ids, own_paper, 1, profile_scope
        for experiment in profile.experiments:
            yield experiment.evidence_ids, own_paper, 1, profile_scope
        for point in profile.limitations:
            yield point.evidence_ids, own_paper, 1, profile_scope
        for card in profile.learning_cards:
            yield card.evidence_ids, own_paper, 1, profile_scope
    profile_ids = {profile.paper_id for profile in analysis.profiles}
    synthesis_scope = set(analysis.synthesis_evidence_scope)
    for point in analysis.cross_paper_comparison:
        yield point.evidence_ids, profile_ids, min(2, len(profile_ids)), synthesis_scope
    for step in analysis.reproduction.steps:
        if step.evidence_ids:
            yield step.evidence_ids, profile_ids, 1, synthesis_scope
    for mapping in analysis.code_mappings:
        yield mapping.evidence_ids, {mapping.paper_id}, 1, synthesis_scope


def _render_delimited(rows: list[list[str]], *, delimiter: str) -> str:
    output = StringIO(newline="")
    writer = csv.writer(output, delimiter=delimiter, lineterminator="\n")
    for row in rows:
        writer.writerow([_spreadsheet_cell(value) for value in row])
    return output.getvalue()


def _spreadsheet_cell(value: str) -> str:
    clean = _safe_text(value)
    if re.match(r"^[=+\-@]", clean):
        return "'" + clean
    return clean


def _plan_text(value: str, prefix: str) -> str:
    clean = _safe_text(value)
    if clean.startswith(prefix):
        return clean[len(prefix) :].strip()
    return clean


def _point(point: EvidencePoint, valid: set[str]) -> str:
    citations = _citations(point.evidence_ids, valid)
    suffix = f" {citations}" if citations else "（未绑定有效论文证据）"
    boundary = f"；边界：{_safe_text(point.boundary)}" if point.boundary else ""
    return (
        f"{_safe_text(point.text)}{suffix}"
        f"（置信度：{_safe_text(point.confidence)}{boundary}）"
    )


def _citations(ids: list[str], valid: set[str]) -> str:
    return " ".join(f"\ue000{item}\ue001" for item in ids if item in valid)


def _page_label(ids: list[str]) -> str:
    labels = []
    for item in ids[:3]:
        match = re.match(r"^(P\d+):p(\d+)", item)
        if match:
            labels.append(f"{match.group(1)} p.{match.group(2)}")
    return ", ".join(dict.fromkeys(labels))


def _label(value: str) -> str:
    clean = re.sub(r"[\r\n\t]+", " ", _safe_text(value))
    clean = clean.replace("\\", "/").replace('"', "'")
    clean = clean.replace("<", "‹").replace(">", "›")
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:220]


def _cell(value: str) -> str:
    return _safe_text(value).replace("|", "\\|").strip()


def _safe_text(value: str) -> str:
    """Render untrusted PDF/LLM text as plain Markdown text, never as a citation."""

    clean = re.sub(r"[\ue000-\uf8ff]", "", str(value))
    clean = re.sub(r"\[P\d+:p\d+(?:-p\d+)?:c\d+\]", "", clean)
    clean = clean.replace("`", "'").replace("<", "‹").replace(">", "›")
    clean = clean.replace("[", "［").replace("]", "］")
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _finalize_markdown(markdown: str) -> str:
    clean = re.sub(r"\[P\d+:p\d+(?:-p\d+)?:c\d+\]", "", markdown)
    return re.sub(
        r"\ue000(P\d+:p\d+(?:-p\d+)?:c\d+)\ue001",
        lambda match: f"[{match.group(1)}]",
        clean,
    )


def _evidence_sort_key(value: str) -> tuple[int, int, int, str]:
    match = re.match(r"^P(\d+):p(\d+)(?:-p\d+)?:c(\d+)$", value)
    if not match:
        return (10**9, 10**9, 10**9, value)
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)), value)

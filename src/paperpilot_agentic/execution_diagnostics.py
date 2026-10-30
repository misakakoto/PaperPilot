"""Present spending, serving capacity and provider failures as distinct states."""

MESSAGES = {
    'NO_SUPPORTED_FACTS': ('审核未保留可信事实', '查看实际送入的来源与审核回执，定位缺失原文或图表；未通过的草案不会作为答案发布。'),
    'SOURCE_COVERAGE_INCOMPLETE': ('来源覆盖不足', '定位并补读缺失的原文或图表，核对来源位置与证据窗口；追加预算不保证恢复缺失证据。'),
    'ANSWER_FORMAT_INVALID': ('回答结构或引用格式不合格', '查看保留的失败回执与缺失引用；未核验草案不能作为最终回答。'),
    'MODEL_OR_ENDPOINT_UNAVAILABLE': (
        '模型或接口不可用（404）',
        '核对模型名、地域、业务空间与 API Key 权限；先完成模型预检，再创建新任务。追加预算不能解决此错误，刷新不会重发旧请求。',
    ),
    'MODEL_QUOTA_EXHAUSTED': ('模型免费额度耗尽', '核对免费模型额度；备用模型也不可用时需更新模型配置。'),
    'LOCAL_BUDGET_ADMISSION': ('本地预算不足', '保留已核验结果；需要新的预算或更小的证据工作集才能继续。'),
    'CONTEXT_TOO_LARGE': ('上下文过长', '按来源拆分材料并重载相关证据；提高预算不会扩大模型窗口。'),
    'UNKNOWN_RESULT': ('请求结果未知', '先核对原请求回执；刷新、切换模型或追加预算不会自动重发。'),
}


def execution_fault(run, snapshot):
    metadata = run.get('metadata') or {}
    report = metadata.get('reader_report') or {}
    error_type = report.get('error_type') or str(run.get('error') or '').split(':')[0]
    code = None
    if snapshot and snapshot.get('unknown_results'):
        code = 'UNKNOWN_RESULT'
    elif error_type == 'SourceCoverageError':
        code = 'SOURCE_COVERAGE_INCOMPLETE'
    elif error_type == 'AuditSupportError':
        code = 'NO_SUPPORTED_FACTS'
    elif error_type in {'JSONDecodeError', 'LLMOutputTruncated'}:
        code = 'ANSWER_FORMAT_INVALID'
    elif run['status'] == 'BUDGET_EXHAUSTED' or error_type == 'LLMBudgetExhausted':
        code = 'LOCAL_BUDGET_ADMISSION'
    elif error_type == 'ContextWindowExceeded':
        code = 'CONTEXT_TOO_LARGE'
    elif run['status'] == 'FAILED':
        provider = metadata.get('last_provider_failure') or {}
        if error_type == 'ModelQuotaExhausted' or provider.get('provider_code') == 'AllocationQuota.FreeTierOnly':
            code = 'MODEL_QUOTA_EXHAUSTED'
        elif (provider.get('category') == 'MODEL_OR_ENDPOINT_UNAVAILABLE'
              or provider.get('http_status') == 404):
            code = 'MODEL_OR_ENDPOINT_UNAVAILABLE'
    if code is None:
        return None
    title, action = MESSAGES[code]
    return {'code': code, 'title': title, 'action': action}

# 证据核验反馈与预算重试

`run_rag_with_retry` 包装统一管线。`run_full_rag.py` 和 benchmark 都使用同一套实现。
默认 `max_retry: 0`；通过配置或单题命令的 `--max-retry 1` 启用。

```yaml
retry:
  max_retry: 1
  strategy: evidence_budget
  top_k_increment: 2
  max_top_k: 5
  max_total_pages: 8
  retryable_reasons: [retrieval_miss, no_citations, grounding_failed]
```

上述参数适合初始 `top_k: 3` 的 6GB 配置。`max_retry` 不包括首次尝试。
`max_total_pages` 统计各轮实际检索页暴露量之和，重复页也计数；不等于全部 VLM 核验 token 或调用预算。
调度下一轮时按其请求 top-k 预留预算，因此剩余页不足时可能提前保守停止。

## 行为

| 原因 | evidence_budget 策略 |
|---|---|
| retrieval_miss | 即使尚无生成/核验反馈，也可有限扩大 top-k |
| no_citations | 保持当前页集大小，传递引用格式反馈 |
| grounding_failed | 传递失败断言/引用反馈，并扩大 top-k |
| model_refusal | 默认停止，避免强迫模型猜测 |
| 相同页集、原因、草稿再次出现 | 停止无效循环 |

`increase_top_k` 是统一扩展对照；`adjust_fusion` 保留为旧的固定权重调整实验选项，
不是学习得到的策略。不会自动删除失败引用页，因为失败可能来自答案而不是原文。

每轮参数从原始配置深拷贝构造，top-k 为初始值加线性增量，不累计修改后续问题的参数。
强制重建仅在首次尝试执行。`retry_triggered` 表示确实发生过额外尝试，成功恢复后也保持 true。

## 返回结果

`retry_metadata` 保存最终轮次、是否触发、是否耗尽、停止原因、总页数和逐轮 history。
history 含页码、top-k、拒答原因、反馈、失败引用、耗时和 GPU 分配内存峰值。
顶层耗时覆盖整个请求的所有尝试；峰值取各轮最大值。最终答案仍必须通过同一套门控。

主要停止原因：`accepted`、`non_retryable`、`max_retry`、`page_budget`、`unchanged_evidence`。
目前单一开发集阈值文件只支持单次检索；改变 top-k/融合设置的重试不能静默复用它，会明确报错。

## 对照实验

`full` 为无重试；`uniform_retry` 为统一扩展；`budgeted_retry` 为按原因调度；
`large_context_once` 一次使用允许的最大 top-k。必须同时报告回答质量、覆盖率、页数和耗时，
不能把额外证据带来的收益全部归给反馈机制。

抽取式 CPU 后端不使用语言反馈，主要用于核验拒答、预算、配置隔离等回归检查。
真实 VLM 反馈纠错收益需在独立中文标注测试集上验证。
研究定位与相关工作见 [research_design.md](research_design.md)；本机制没有已被实验确认的优越性结论。

# PDF-Grounded RAG

面向复杂 PDF 的多模态 RAG：页级 grounding（答案可追溯到具体页面）+ 多信号不可回答拒答，并提供公开/自建数据集上的对比与消融工具。

## 核心能力

| 模块 | 说明 |
|------|------|
| 多模态检索 | ColPali / CLIP 视觉 + BM25 文本，按页文本密度自适应融合 |
| 页级 Grounding | 强制 `[p.N]` 引用 → 检索集校验 → Cite-then-Verify（VLM 或词汇重叠） |
| 多信号拒答 | 检索低分 / 模型 UNANSWERABLE / 缺引用 / 验证失败 |
| 评估与消融 | EM、F1、页召回/精度、引用 F1、拒答 P/R/F1；基线与组件消融 |

低内存笔记本上，`runtime.auto` 会自动回退到 **CLIP + 抽取式生成**，保证管线可跑通。

## 流程

```
PDF -> 页图 + 文本 -> ColPali|CLIP / BM25 索引
            |
     Adaptive Hybrid Top-K
            |
   Retrieval gate (低分 => 拒答)
            |
   Qwen2-VL 或 Extractive 生成（带页引用）
            |
 Parse [p.N] -> Cite-then-Verify (VLM / lexical)
            |
 缺引用 / 验证失败 => 拒答；否则答案 + 页证据
```

## 环境

- Python 3.10+
- 完整 ColPali + Qwen2-VL：建议系统内存 ≥16GB，GPU ≥8GB
- 笔记本可运行路径：CLIP + extractive（`configs/laptop.yaml`）

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## 快速开始（推荐：本机可跑）

```bash
# 1) 生成样例 PDF + 自建评测集（10 题，含不可回答）
python scripts/setup_sample_data.py

# 2) CPU 冒烟测试
python scripts/smoke_test.py

# 3) 笔记本安全全流程（CLIP + 抽取式 + grounding + 拒答）
python scripts/run_full_rag.py --demo --pdf data/pdfs/sample_doc.pdf --question "What is the main contribution of this document?"

# 4) 自建集 benchmark + demo 消融
python scripts/run_benchmark.py --laptop --dataset data/eval/self_built_dataset.json --method demo --output results/self_built_demo.json
python scripts/run_benchmark.py --laptop --dataset data/eval/self_built_dataset.json --ablation-demo --output-dir results/ablation_demo
python scripts/report_results.py --input-dir results/ablation_demo
```

## 完整 GPU 路径（ColPali + Qwen2-VL）

机器内存充足时，默认 `runtime.visual_backend=auto` 会选用 ColPali：

```bash
python scripts/run_full_rag.py --pdf data/pdfs/sample_doc.pdf --question "What is the main contribution of this document?"
python scripts/run_benchmark.py --dataset data/eval/self_built_dataset.json --method full --output results/self_built_full.json
python scripts/run_benchmark.py --ablation-all --dataset data/eval/self_built_dataset.json --output-dir results/ablation
```

也可在配置中强制：

```yaml
runtime:
  visual_backend: colpali   # 或 clip / none
  generation_backend: vlm   # 或 extractive
```

## 消融配置

| 名称 | 含义 |
|------|------|
| `full` | Hybrid + Cite-then-Verify + 全拒答信号 |
| `no_grounding` / `no_refusal` / `no_verify` / `no_citation_gate` | 组件消融 |
| `retrieval_gate_only` / `model_refusal_only` | 单信号拒答 |
| `visual_only` / `ocr_only` / `hybrid_baseline` | 检索基线 |
| `demo_*` | 笔记本安全消融（CLIP + extractive） |

## MMLongBench-Doc

```bash
python scripts/prepare_mmlongbench.py --from-json path/to/raw.json --pdf-root data/pdfs/mmlongbench --output data/eval/mmlongbench_subset.json --limit 100
python scripts/prepare_mmlongbench.py --from-hf --limit 50
```

评测条目 schema：

```json
{
  "id": "q1",
  "pdf_path": "../pdfs/sample_doc.pdf",
  "question": "...",
  "answer": "...",
  "gold_pages": [1, 3],
  "answerable": true
}
```

## 目录

```
configs/          default.yaml / laptop.yaml
data/pdfs/        PDFs
data/eval/        评测 JSON
src/ingest/       PDF -> 页图 + 文本
src/retrieval/    ColPali / CLIP / BM25 / hybrid fusion
src/generation/   Qwen2-VL + extractive
src/grounding/    引用解析 + Cite-then-Verify
src/refusal/      多信号拒答
src/pipeline/     端到端编排
src/eval/         指标、消融、批量评测
scripts/          CLI
tests/            单元测试（无需 GPU）
results/          实验结果
```

## 测试

```bash
python scripts/smoke_test.py
python -m pytest tests/ -q
```

## 研究点对应

1. **复杂 PDF 多模态 RAG** — 页图 + 文本，密度自适应融合；低资源机用 CLIP 替代 ColPali。
2. **页级 Grounding** — 强制 `[p.N]`，必须落在检索集，VLM YES/NO 或词汇重叠验证。
3. **不可回答检测** — 检索 / 模型 / 缺引用 / 验证失败，输出结构化 `refusal_reason`。
4. **实验** — 自建 10 题集 + MMLongBench 适配；基线与消融；EM/F1、页召回、引用 F1、拒答 P/R/F1。

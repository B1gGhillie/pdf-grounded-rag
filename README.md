# PDF-Grounded RAG

面向中文行业文档与学位论文的多模态 RAG：查询与文本可靠性调节的检索融合、断言与页引用绑定核验、多信号拒答和有预算的反馈重试。

研究方法、已有工作对照、创新边界和真实数据实验步骤见 [研究设计](docs/research_design.md)。目前样例仅用于软件回归，尚不能证明真实数据上的性能提升。

## 核心能力

| 模块 | 说明 |
|------|------|
| 中文与扫描件 | 字符/双字检索、全页查询相关片段、原生文本或真实 Tesseract OCR |
| 多模态检索 | ColPali / CLIP + BM25，密度、查询覆盖、抽取质量调权；固定融合/RRF 对照 |
| Grounding | `[p.N]` 绑定事实句 → 每条核验 → 数字单位/否定检查或 VLM；保留失败证据 |
| 多信号拒答 | 检索低分 / 模型 UNANSWERABLE / 缺引用 / 验证失败 |
| 反馈重试 | 按失败原因控制扩展，页数/次数预算、配置隔离、全过程记录 |
| 评估与消融 | 文档隔离、开发集阈值、EM/F1、固定 k 页召回、引用、拒答、覆盖/风险及文档 bootstrap |

低内存环境可使用 `configs/chinese_cpu.yaml` 的纯文本/真实 OCR 与抽取式回归路径。
`runtime.auto` 根据可用资源选择后端；正式实验请显式固定后端并检查结果记录，避免混用不同模型。

## 中文文档快速开始

```powershell
# 仅下载官方中英文 OCR 语言数据到项目内；原生 PDF 的 text 模式不需要此步骤
./scripts/setup_ocr.ps1
python scripts/setup_chinese_fixtures.py
python scripts/run_benchmark.py --config configs/chinese_cpu.yaml --dataset data/fixtures/chinese/dataset.json --output results/chinese_regression.json
python scripts/run_full_rag.py --config configs/chinese_cpu.yaml --pdf data/fixtures/chinese/industry_native.pdf --question "项目总投资是多少？" --output results/answer.json
```

中文样例包含四份虚构原生/扫描 PDF、12 道回归题；不能作为论文评测集。
真实数据可用 `scripts/prepare_chinese_eval.py` 检查并按 PDF 内容划分开发/测试集，
再用 `scripts/calibrate_refusal.py` 拟合拒答阈值，详见研究设计。

## 流程

```
PDF -> 页图 + 文本 -> ColPali|CLIP / BM25 索引
            |
     Query/Quality-aware Hybrid Top-K
            |
   Retrieval gate (低分 => 拒答)
            |
   Qwen2-VL 或 Extractive 生成（带页引用）
            |
 Parse claim + [p.N] -> Verify each claim (VLM / guarded lexical)
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

### 6GB 显存 GPU：真实 VLM 评测

`configs/gpu_6gb.yaml` 为 RTX 3060 6GB 等小显存 GPU 提供了可执行配置：检索使用
CLIP，生成与证据核验使用 4-bit Qwen2-VL。它是实际 VLM 路径，不是抽取式 demo；为
避免显存竞争，不在该配置中加载 ColPali。

```bash
# 新建或修复 Windows CUDA 环境时，使用匹配的官方 PyTorch wheel
pip install -r requirements-gpu-cu124.txt

# 基础逻辑与 CUDA 分别验证；GPU 测试不会在普通 pytest 中被意外执行
python -m pytest -q
python -m pytest --run-gpu tests/gpu -q

# 首次运行会下载公开的 CLIP 和 Qwen2-VL 权重
python scripts/run_full_rag.py --config configs/gpu_6gb.yaml --pdf data/pdfs/sample_doc.pdf --question "What is the main contribution of this document?"

# 真正的 GPU/VLM 基线与消融结果（写入一个与 demo 结果隔离的目录）
python scripts/run_benchmark.py --config configs/gpu_6gb.yaml --dataset data/eval/self_built_dataset.json --method full --output results/self_built_gpu_6gb.json --force-rebuild
python scripts/run_benchmark.py --config configs/gpu_6gb.yaml --dataset data/eval/self_built_dataset.json --ablation-all --output-dir results/ablation_gpu_6gb --force-rebuild
python scripts/report_results.py --input-dir results/ablation_gpu_6gb
```

真实 VLM 结果应与 `results/ablation_demo/` 分开解读：后者使用抽取式生成与规则核验，
只用于低资源回归测试。CUDA 冒烟通过不表示 VLM 回答正确；小模型漏引用、核验误拒需独立人工评价。
GPU 配置默认原生文本；扫描件需设置 `pdf.text_mode: auto` 和 `pdf.tessdata: data/tessdata`。
研究套件包含真实 OCR 基线，运行前须安装上述语言数据，并在配置中指定 tessdata。

## 消融配置

| 名称 | 含义 |
|------|------|
| `full` | Hybrid + Cite-then-Verify + 全拒答信号 |
| `no_grounding` / `no_refusal` / `no_verify` / `no_citation_gate` | 组件消融 |
| `retrieval_gate_only` / `model_refusal_only` | 单信号拒答 |
| `visual_only` / `ocr_only` / `hybrid_baseline` | 检索基线 |
| `demo_*` | 笔记本安全消融（CLIP + extractive） |
| `fixed_fusion` / `rrf_fusion` / `density_fusion` | 融合对照 |
| `no_query_reliability` / `no_text_quality` | 可靠性特征消融 |
| `page_verification` / `verified_citation_recovery` | 核验粒度 / 经证据检查的引用恢复 |
| `budgeted_retry` / `uniform_retry` / `large_context_once` | 重试收益与额外证据量的对照 |

`--ablation-all` 默认只运行 `research` 套件；`--suite all` 才包含历史和 demo 配置。
基线重试默认关闭；嵌套覆盖保留其他参数。`no_refusal` 关闭策略门控，仍如实统计模型主动拒答。

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

测试分层使用 pytest markers：`unit` 不依赖模型、网络或 GPU；`integration` 使用假对象
覆盖组件边界；`gpu` 必须以 `--run-gpu` 显式启用。真实模型的端到端验证使用上面的 GPU
评测命令，不混入默认单元测试。

## 研究点对应

1. **候选方法贡献**：查询覆盖与抽取质量共同调节页融合，验证噪声中文文档上的收益。
2. **候选方法贡献**：断言证据绑定与预算约束反馈，验证错误接受率、覆盖率和成本的关系。
3. **工程与评测支撑**：真实 OCR、模型/预处理缓存指纹、文档隔离、阈值文件及完整实验记录。

ColPali、带引用生成、拒答和重试均已有相关研究，不能单独包装成原创贡献。
创新是否成立须用真实标注数据和对照实验回答，详见 [研究设计](docs/research_design.md)。

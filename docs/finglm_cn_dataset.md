# FinGLM 中文财报数据

本项目使用 MetaGLM/FinGLM 的固定版本 `ff48a9dc5e6c7f9d9a9c6120024dccefd8cbaa80`。原项目公开了中文上市公司年报问答和 ModelScope 年报文件库；数据仅用于研究，使用时应遵守上游项目及年报来源的条款。

## 已下载内容

- `data/finglm_cn/raw/pre.jsonl`：5,000 条中文问答。
- `data/finglm_cn/raw/a.jsonl`、`b.jsonl`、`c.jsonl`：另外 5,000 条比赛问答。
- `data/finglm_cn/raw/reports_list.csv`：11,588 份年报索引。
- `data/finglm_cn/pdfs/`：与首批问题对应的 5 份中文年报，共约 30 MB；下载地址和 SHA-256 在 `pdf_manifest.json`。
- 当前下载计划已扩展到 25 份年报，工作区保留了 26 份已下载年报（约 155 MB）；实际文件和 SHA-256 以最新 `pdf_manifest.json` 为准。

## 训练划分

`scripts/prepare_finglm_sft.py` 从 `pre.jsonl` 取 1,000 条训练样本，按原始 `id % 10 == 0` 固定保留 500 条开发样本，避免随机重划导致结果不可复现。训练记录在 `data/finglm_cn/sft/manifest.json`。

该问答标注没有统一的 PDF 证据页字段，所以这批 SFT 数据用于中文金融问答表达和数值回答格式训练；它不能单独证明页面检索或引用准确率。页面级 RAG 评测继续使用 MMLongBench-Doc。

## LoRA 训练

```powershell
.\.venv\Scripts\python.exe scripts/prepare_finglm_sft.py --limit 1000
.\.venv\Scripts\python.exe scripts/train_finglm_lora.py `
  --data data/finglm_cn/sft/train.jsonl `
  --output models/adapters/finglm-qwen2vl-2b `
  --epochs 1 --max-length 512 --grad-accum 8
```

已生成的适配器位于 `models/adapters/finglm-qwen2vl-2b`。要在 RAG 推理中启用它，使用 `configs/gpu_6gb_finglm_lora.yaml`；基线实验仍使用 `configs/gpu_6gb.yaml`，以便做公平对照。

## 证据约束实验

`scripts/build_finglm_grounded_sft.py` 根据答案中的数值和页内文本匹配出 55 条弱证据样本，输出 `grounded_train.jsonl`。这些页码是自动匹配结果，不是人工金标，不能用于报告页召回率。第二阶段适配器位于 `models/adapters/finglm-qwen2vl-2b-grounded`，对应配置为 `configs/gpu_6gb_finglm_grounded.yaml`，只用于消融实验。

推理代码现在对空检索结果直接返回 `UNANSWERABLE`，避免适配器在没有证据时猜测数字。任何正式结论仍需在 MMLongBench-Doc 上报告答案、页码、引用和拒答指标。

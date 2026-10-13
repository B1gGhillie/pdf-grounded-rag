# MMLongBench-Doc 项目数据集

本项目以 MMLongBench-Doc 为主实验数据，不再限制中文。已将官方问答和 PDF
导出为项目可读取的 JSON；这是数据接入与评测协议，不是模型训练或方法有效性的结论。

## 来源与固定版本

- 标注与 PDF 索引：[官方 Hugging Face 数据集](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc/tree/2ff6aa9237fc777b6627dc57a486e9225ac5fb86)。
- HF revision：`2ff6aa9237fc777b6627dc57a486e9225ac5fb86`。
- 原始标注：`data/train-00000-of-00001.parquet`，实际读取 **1,091** 条。
- 作者 PDF 备份与说明：[官方 GitHub 固定版本](https://github.com/mayubo2333/MMLongBench-Doc/tree/d73f0dc0be7e0a2ff6a403d5fe65fcd96461f384)。
- GitHub revision：`d73f0dc0be7e0a2ff6a403d5fe65fcd96461f384`。
- 论文：[MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations](https://arxiv.org/abs/2407.01523)。使用数据时应引用原论文。
- 数据许可：官方 README 标明 **CC BY-NC 4.0**，限非商业研究；本项目不改变原始数据及文档权利。

HF 的上游 split 名为 `train`，不表示它是官方为微调划定的训练集。这里将其用于
RAG 开发与评测。不要在完整数据上微调后，再用同一批题报告泛化效果。

该 GitHub 版本的 `data/samples.json` 实际为 1,082 条，与 HF 的 1,091 条不同。
本项目使用 HF 原始 Parquet，未混合两个标注版本。

## 本次完整导出（2026-10-13）

| 集合 | 题数 | PDF 文件名数 | 不同 PDF 内容数 | 可回答 | 不可回答 |
|---|---:|---:|---:|---:|---:|
| full | 1,091 | 135 | 133 | 847 | 244 |
| dev | 331 | 40 | 40 | 258 | 73 |
| test | 760 | 95 | 93 | 589 | 171 |

全部文件约 **635.5 MiB**，按文件名累计 6,529 页，最长文档 468 页。
所有 135 份 PDF 都通过 HF Git/LFS 内容哈希核对，并能被 PyMuPDF 打开。
下载中两份文件使用了作者 GitHub 备份，最终内容哈希仍与 HF 对应项一致；其中一份
HF 索引的文件大小元数据过时。来源、实际大小和校验值记录在 `source_manifest.json`。

开发/测试集按 PDF 内容 SHA-256 分组，随机种子 42，开发集约占不同文档内容的 30%。
两组内容哈希无交集，题目并集等于完整集，没有丢题。它们是**项目自定义划分**，
不是 MMLongBench 官方划分，也不与官方全量排行榜直接等价。

发现两组不同文件名对应完全相同的 PDF 内容，原样保留并放在同一集合：

- `SAO-StudentSupport_Guidebook-Content.pdf` / `StudentSupport_Guidebook.pdf`。
- `digitalmeasurementframework22feb2011v6novideo-110221233835-phpapp01_95.pdf` /
  `dr-vorapptchapter1emissionsources-121120210508-phpapp02_95.pdf`。

第二组文件名主题不同，使用这些文档做定性分析时应人工检查问答是否对应实际内容。
本次是结构与文件完整性审计，未逐题人工复核所有答案。

## 文件位置

```text
data/raw/mmlongbench/
  annotations.parquet       官方原始标注
  samples.hf.json            Parquet 的完整 JSON 导出
  source_manifest.json       固定版本、源文件及 PDF 校验值
  README.upstream.md         作者数据说明与许可声明
data/pdfs/mmlongbench/       135 份原始 PDF
data/eval/mmlongbench/
  full.json                 项目格式的完整 1,091 题
  dev.json                  331 题：调参、阈值和提示词开发
  test.json                 760 题：冻结后的测试
  smoke.json                从开发集选出的 4 题接入检查
  manifest.json             统计、划分、证据异常与重复内容清单
```

所有数据集内的 PDF 路径都相对于 JSON 文件所在目录，移动完整项目后仍可读取。
数据文件留在本地并由 Git 忽略，仓库保存下载器、转换器及本文档，避免把大型原始
PDF 误提交。重新获取数据使用下列固定版本命令即可。

## 标签转换与异常处理

- `doc_id` 已带 `.pdf` 时不重复追加后缀；`id` 对应固定原始行号，`source_index` 从 0 开始。
- `answer == "Not answerable"` 或 `answer_format == "None"` 映射为不可回答；项目 `answer` 置空，原答案仍保留在 `upstream_answer`。
- 安全解析 `"[3, 5]"` 和 Python 列表形式的 `evidence_sources`，不执行标注中的代码。
- 有效 `gold_pages` 使用从 1 开始的实际 PDF 页序号；不会因为个别 `[0]` 就对整套数据统一加 1。
- `doc_type`、`answer_format`、`evidence_sources` 及原始证据字段保留，用于分类分析与审计。
- 1 组同文档重复问题保留原始行，不静默合并。

可回答题中 **18 条**没有可用的页证据标签：9 条缺失，9 条为 0 或超出对应 PDF 页数。
它们保留在所有适用集合，仍参与答案与拒答指标；设置 `evidence_annotation_status`
为 `missing` 或 `invalid`，保存具体原因，项目 `gold_pages` 置空。
无法核对的页证据不推测修正，不将这些题改标为不可回答。

相应检索页指标和引用指标为 `null`，不混入均值。结果新增 `page_metric_count` 与
`citation_metric_count` 表明分母。全量可参与页证据计分的题为 829 条，开发集 252 条，
测试集 577 条。7 道不可回答题上游附带的页码仅作为原始元数据保留，不当作答案证据。

本项目使用本地 EM/token-F1、拒答、页召回和引用指标。MMLongBench 官方具有自己的
答案提取与评分流程，因此**不能把本项目 token-F1 直接写成官方 MMLongBench F1**。

## 重现与运行

在项目根目录执行（Windows PowerShell 可将 `python` 替换为 `& .\.venv\Scripts\python.exe`；注意开头的 `& .\`）：

如果终端没有位于项目目录，先执行 `Set-Location 'C:\Users\Big\Projects\pdf-grounded-rag'`。

```powershell
python -m pip install -r requirements-data.txt
# 下载全量、核对内容哈希、导出和按文档划分
python scripts/prepare_mmlongbench.py --from-hf

# 已下载时可完全离线重建项目 JSON
python scripts/prepare_mmlongbench.py --from-json data/raw/mmlongbench/samples.hf.json

# CPU 小样本：确认真实 PDF 能被完整加载、检索和评分
python scripts/run_benchmark.py --config configs/mmlongbench_cpu.yaml --dataset data/eval/mmlongbench/smoke.json --output results/mmlongbench_smoke.json

# 开发集获取候选结果，再拟合拒答阈值
python scripts/run_benchmark.py --config configs/gpu_6gb.yaml --dataset data/eval/mmlongbench/dev.json --no-retrieval-gate --output results/mmlongbench_dev.json
python scripts/calibrate_refusal.py --dev-results results/mmlongbench_dev.json --output results/mmlongbench_threshold.json --max-false-refusal-rate 0.1

# 冻结设置后评测测试集
python scripts/run_benchmark.py --config configs/gpu_6gb.yaml --dataset data/eval/mmlongbench/test.json --calibration results/mmlongbench_threshold.json --output results/mmlongbench_test.json
```

阈值拟合若提示约束不可满足，应报告问题并只在开发集处理，不能使用测试标签调参。
消融改变检索打分时，需要分别校准；不同配置不能共用不匹配的阈值文件。
`max_pages_per_doc` 应保留为 `null`，不能为节省时间静默截断长文档。
正式多模态实验使用 GPU 配置；`mmlongbench_cpu.yaml` 为原生文本抽取式接入检查，
不能代表图表问答能力。扫描文档/OCR 实验仍需单独配置 OCR 数据和模式。

## 第二步：40 题 GPU 诊断

在 4 题 GPU 接入检查之后，先从开发集做小规模诊断，再决定模型、提示词和门控需要怎样修改。
`scripts/prepare_mmlongbench_pilot.py` 已生成 `data/eval/mmlongbench/pilot40.json`：
40 题来自 17 份 PDF、7 种文档类型，32 题可回答、8 题不可回答，包含 12 道跨页题。
正文/表格/图表/图像各 6 题，混合证据/版面各 4 题，另外 8 题不可回答。

该轮只选不超过 80 页、证据标注可用的文档，17 份 PDF 共 432 页，完整建索引而不截断页面。
这是为了控制本机初次实验成本的平衡诊断样本，其分数不能代表完整开发集或测试集。
选题种子、来源哈希和筛选条件记录在 JSON 的 `selection` 中。

Windows PowerShell 按顺序执行：

```powershell
Set-Location 'C:\Users\Big\Projects\pdf-grounded-rag'
# 仅在需要重建相同诊断题时执行；已生成时可跳过
& .\.venv\Scripts\python.exe scripts/prepare_mmlongbench_pilot.py

# 1. 使用与 4 题检查相同的模型和门控，运行 40 题
& .\.venv\Scripts\python.exe scripts/run_benchmark.py --config configs/gpu_6gb.yaml --dataset data/eval/mmlongbench/pilot40.json --output results/mmlongbench_pilot40_gpu.json

# 2. 上一步完成后，生成逐题诊断；不再次调用模型
& .\.venv\Scripts\python.exe scripts/diagnose_benchmark.py --results results/mmlongbench_pilot40_gpu.json --dataset data/eval/mmlongbench/pilot40.json --output results/mmlongbench_pilot40_diagnostics.json
```

先看逐题 `rows` 中的 `gold_answer`、`raw_answer`、`gold_pages`、`retrieved_pages`、
`refusal_reason` 与 `verification_results`，人工复核 10～15 题并在 `manual_review` 记录判断。
`triage` 是定位提示，不是因果判定；原始 EM 不匹配可能只是答案形式不同，不能直接判错。
重新生成诊断 JSON 会覆盖其中的人工字段，需要保留人工记录时先复制该文件。

优先区分：未找到完整证据、找到证据但原始答案不正确、答案正确但被引用/核验机制拒绝，
以及不可回答题生成了无依据内容。诊断同时记录每题耗时、索引耗时与 PyTorch 峰值分配显存。
先处理主要错误类型，再安排基线和消融；不要只为降低拒答率而关闭核验。

用户已执行的 4 题 GPU 结果中，3 道可回答题的证据页全部被召回，但原始输出仍有明显错误：
标准答案 `466.43` 被答成 `5.1`，`Orange` 被答成 `blue`。4 题均被拒答，拒答原因是
3 次缺引用和 1 次核验失败。这不证明所有错误都来自核验器。
用户随后完成了 40 题 GPU 运行，诊断记录如下。

### 40 题运行后的定位与补充检查

结果文件：`results/mmlongbench_pilot40_gpu.json` 和 `results/mmlongbench_pilot40_diagnostics.json`。
40 题全部被拒答，32 道可回答题的完整证据召回为 14/32，平均页召回为 0.475。
33 次实际生成均未自行输出 `[p.N]` 引用；另外 7 题在生成前被检索门控拒绝，
其中 5 题可回答，且其中 1 题已经召回全部标注页。这说明 `retrieval_miss` 只是
分数门控给出的拒答原因，不能等同于客观上没有检索到正确页面。

唯一原始精确匹配的答案 `Berlin School of Experimental Psychology` 在引用恢复到第 2 页后，
VLM 核验仍返回 `NO`。但精确匹配只有 1 题也不代表只有 1 题语义正确：例如
标准答案 `4` 对应原始回答 `The course requires 4 areas.`；标准答案 `5` 对应
`The document contains five appendices.`。人工答案判定与证据充分性需分别记录。
另有 1 道不可回答题用自然语言说明信息未提供，但不符合当前结构化拒答标记，需人工审计。

实际每题平均耗时 12.4354 秒，中位数 13.7611 秒；累计索引/缓存读取耗时 48.4008 秒，
PyTorch 峰值分配显存 3,100.3 MiB（不包含所有驱动/外部分配）。这轮没有显存不足的证据。

已在同一台 GPU 上做两组**事后选题诊断**，不修改用户的基线结果，不使用测试集：

1. `results/mmlongbench_prompt_probe_en.json`：6 道选题沿用原检索页、文本窗口、
   2B 量化模型和解码设置，试用英文短答案/引用提示，并在相同旧断言与引用页上重验。
   正确的 Berlin 答案由拒绝变为通过，但模型仍未稳定生成有效引用，还出现照抄格式说明
   以及不可回答题生成错误数值。因此该变体只是诊断开关，未替换默认中文提示配置。
   此处改变了提示措辞和语言，不能把变化单独归因为语言。
2. `results/mmlongbench_gold_evidence_probe.json`：6 道可回答题使用标准证据页，
   保留原中文提示、模型和文本窗口，**不向模型提供标准答案**。只有 Berlin 一题原始精确匹配；
   其余可见错误包括将 `8980` 回答为 `100`、复制问题中的示例列表以及只返回页面标签。
   这表明该设置的错误不能全部归因于检索，尚需区分图像分辨率、文本片段、提示与模型能力。
   标准证据页实验使用了证据标签，不能作为端到端 RAG 成绩。

另在同一批 32 道可回答题做了原生文本检索检查，结果保存在
`results/mmlongbench_pilot40_retrieval_diagnostic.json`：当前融合 top-3 完整召回 14 题，
BM25 top-3 为 10 题，top-5 为 13 题，top-10 为 17 题。增大候选集仍不等于生成能力提升。
其中 3 道题标注证据超过 3 页，固定返回 3 页必然无法完整覆盖，后续需要证据预算/全局计数策略。

复现两组提示/证据诊断（使用现有页面缓存；仅供开发定位）：

```powershell
& .\.venv\Scripts\python.exe scripts/probe_grounding_prompt.py --results results/mmlongbench_pilot40_gpu.json --dataset data/eval/mmlongbench/pilot40.json --ids mmlongbench_00013 mmlongbench_00016 mmlongbench_00289 mmlongbench_00412 mmlongbench_00290 mmlongbench_00491 --output results/mmlongbench_prompt_probe_en.json
& .\.venv\Scripts\python.exe scripts/probe_grounding_prompt.py --results results/mmlongbench_pilot40_gpu.json --dataset data/eval/mmlongbench/pilot40.json --ids mmlongbench_00013 mmlongbench_00289 mmlongbench_00069 mmlongbench_00018 mmlongbench_01000 mmlongbench_00294 --prompt-language zh --gold-evidence --output results/mmlongbench_gold_evidence_probe.json
```

下一阶段先审计语义答案与拒答标签，再测试结构化引用输出、问答式证据核验和多页证据预算。
提示变体、阅读分辨率或模型变化应分别控制，先用小型开发诊断集，再复测完整 40 题。
如果给足清晰证据后仍不能可靠完成基础任务，再比较更强的模型；GPU 租赁由该对照的实际资源需求决定。
当前不宜直接开始 760 题正式测试或完整 18 组消融，也不能通过关闭门控把生成的错误答案当作改进。

## 本次验证

全量文件完整性、PDF 可打开性、有效证据页边界、ID 唯一性、相对路径以及
开发/测试内容隔离已验证。4 道开发集真实问题已跑通 CPU 端到端流程。
该抽取式检查的答案 EM/token-F1 为 0，不能作为系统效果已达标的证据；完整 GPU
基线、消融和人工答案核验尚未在此任务中执行。

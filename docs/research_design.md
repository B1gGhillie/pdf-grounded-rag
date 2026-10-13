# 长篇 PDF 问答：研究定位和验证方案

本项目的建议论文题目为：**面向复杂 PDF 的证据可靠性感知检索与可核验问答研究**。
2026-10-13 起，主实验数据改为 MMLongBench-Doc，不再强制中文；中文行业文档和学位论文作为可选扩展。
已导入官方 1,091 道题与 135 份 PDF，数据版本、文档隔离与异常标注口径见 [数据集说明](mmlongbench_dataset.md)。
目前已实现方法、基线和评测工具，但尚未完成证明方法优于已有工作的真实数据对照实验。
下文中文语料构建步骤保留为扩展实验方案；主实验使用数据集说明中的 MMLongBench 开发/测试集。
下面的两项是可检验的候选贡献，不应写成已经证实的创新成果，更不能声称“首次”。

## 1. 已有工作与贡献边界

截至 2026-10-08，对以下原始论文摘要页做了定向核对，未进行穷尽的新颖性检索：

| 已有工作 | 已覆盖内容 | 本项目需要额外证明什么 |
|---|---|---|
| [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449)，ICLR 2025 | 文档页图、多向量视觉检索和 late interaction | 接入 ColPali 本身不是创新；需证明新的融合策略在中文与抽取噪声条件下有增益 |
| [Enabling Large Language Models to Generate Text with Citations](https://arxiv.org/abs/2305.14627)，EMNLP 2023，ALCE | 带引用生成及答案/引用质量评价 | 添加页码引用不是创新；需验证断言与证据绑定及错误控制机制 |
| [R2VC: Modular Fact-Checking with Retrieval, Verification, and Confidence Calibration](https://arxiv.org/abs/2609.11955)，2026 预印本 | 混合检索、外部 NLI 核验、置信度与选择性拒答 | “检索+核验+拒答”的组合已有近似工作；本文候选差异是中文 PDF 抽取可靠性、低资源约束和证据预算策略 |

这些工作支持问题的重要性，也限制了可以声称的贡献范围。R2VC 面向开放域事实核查，
与本项目的单文档问答任务不同；任务差异本身不能证明方法新颖。
正式开题/投稿前仍应进一步检索视觉文档 RAG、选择性问答、引用核验和自适应检索工作并阅读全文。
原重试说明中的 CaVe-VLM-CoT 具体对照未得到原文验证，已移除，不能作为论据使用。

## 2. 候选贡献 A：查询与抽取可靠性共同调节的页级融合

问题：文本密度高不等于文本可用。例如中文扫描页乱码密集、页眉重复，
或某页正文很多但与当前问题无关。只按长度分配文本权重会错误放大这些页面。

实现：`src/retrieval/hybrid_fusion.py`，配置 `retrieval.fusion.mode: reliability`。
对页面 p 和查询 q 定义：

```text
d(p) = 页文本字符数 / 全文最大页文本字符数
b(p) = w_min + (w_max - w_min) * d(p)
r(p) = 可用字母/数字字符占比 * (1 - 替换字符占比)
c(q,p) = 查询内容词集合与页内容词集合交集大小 / 查询内容词集合大小
w_text(q,p) = w_min + (b(p) - w_min) * r(p) * c(q,p)

s_text = max(BM25,0) / (max(BM25,0) + text_scale)
s_visual = clamp(CLIP cosine,0,1)
          或 max(ColPali,0) / (max(ColPali,0) + visual_scale)
s(p,q) = w_text * s_text + (1 - w_text) * s_visual
```

BM25 使用常规正 IDF `log(1 + (N-df+0.5)/(df+0.5))`，避免单页或小文档中常见词产生非正得分。
中文使用字符与相邻双字片段；英文使用词。该分词方案是工程基线，不是学习得到的分词器。
`r(p)` 只是可观察的字符质量代理，**不是 OCR 置信度**，无法识别形似字误识别。
所有缩放常数和权重均需仅在开发集确定；这些得分**不是回答正确的概率**。

`candidate_k` 固定候选规模，排序使用 `fused_score`，拒答使用单独保留的
`evidence_confidence`。RRF 的排序值不直接拿来作拒答阈值；旧 min-max 仅保留为对照。
单通道缺失时使用剩余通道。输出保留原始分数、权重、查询覆盖率、抽取来源和全文证据。

**假设 H1**：在扫描噪声、稠密无关页、页眉干扰等子集上，可靠性融合比固定融合、RRF
和仅密度融合有更高 Recall@1/3/5，并减少无依据回答；在干净文本上不明显退化。

必须比较：`fixed_fusion`、`rrf_fusion`、`density_fusion`、`full`、
`no_query_reliability`、`no_text_quality`。保持候选页数、生成器、提示、核验器一致。
噪声强度应预先定义，包含真实 OCR 误识别，不可只构造适合当前质量公式的替换字符。
若优势仅来自 BM25 或多取了页面，不能归因于新融合。

## 3. 候选贡献 B：绑定断言与证据的预算约束核验反馈

问题：整段答案的一部分受支持，不代表每个断言都受支持；一个引用正确也不代表其余引用正确。
固定扩展 top-k 还可能在引用格式错误时徒增图片和推理开销。

实现：`src/grounding/claims.py`、`src/grounding/pipeline.py`、`src/pipeline/rag_retry.py`。

- 引用组绑定其前面的句子；尾部无引用句子保留为失败项。它是确定性的句子切分，不能称为语义原子断言分解。
- 每个断言单独核验；全部通过才接受。多个引用页可合并交给 VLM 判断跨页支持。
- CPU 核验检查相关句子的词汇覆盖、数字及单位、显式缺失信息和否定差异。
  这是保守规则代理，不能证明语义蕴含；换算、推导、同义改写可能误拒，相关但答非所问也可能漏检。
- VLM 核验读取问题、该断言、其引用页图及对应抽取文本，检查支持与问题关联。
  生成器与核验器目前使用同一模型，不能称为独立核验，不能把内部 YES/NO 当成人工标注。
- `evidence_budget` 遇到缺引用时保持 top-k，给出格式反馈；检索不足/核验失败时有限扩展。
  相同证据、原因和草稿再次出现时停止；模型明确拒答默认不重试。
- 记录每轮页码、反馈、原因、耗时、显存峰值；达到次数或累计证据页预算后返回拒答。

预算按**每轮交给生成阶段的检索页数之和**计算，重复页也计费；它不是总 token 数、
核验调用次数或全部 GPU FLOPs 的严格预算。当前策略没有学习失败原因，也没有自动删除所谓坏页，
避免把“答案错了”误解成“证据页不可再用”。

**假设 H2**：在缺引用、部分断言不受支持、数值/否定冲突的子集上，
相比整段/页级核验和统一扩展 top-k，断言绑定核验与按原因重试可降低错误接受率，
并在相近答案覆盖率和资源预算下减少无效扩展。

必须比较：`page_verification`、`full`、`uniform_retry`、`budgeted_retry`、
`large_context_once`。同时报告质量、覆盖率、总页暴露量、端到端耗时和显存。
CPU 抽取式生成不消费重试提示，无法用于证明 VLM 的反馈纠错收益。
`verified_citation_recovery` 是独立可选消融：缺引用时先做文本支持检查，再走正常核验。
默认研究消融禁用该功能；6GB 交互配置启用它缓解小模型漏引用，不能无条件贴 top-1 页码。

## 4. 拒答阈值与可信评测

已实现的是**开发集阈值选择**，不是概率校准或风险保证。给定允许的开发集误拒率，
枚举与运行时 `< threshold` 一致的候选阈值，在约束内最大化不可回答检测 F1；
并列时选择更少误拒和更小阈值。不可回答问题也可能有很高检索分数，阈值无法保证分离。

阈值文件记录开发 PDF 内容哈希、分数配置签名、结果文件哈希和扫描结果。
在测试 PDF 与开发 PDF 内容重复、配置不一致、缺少两类开发样本时直接报错。
重命名复制同一 PDF 不会绕过文档隔离。单一阈值文件只用于单次检索；
自适应重试改变证据设置时需要每种设置分别标定，目前明确拒绝复用单一阈值文件。

指标口径：

- 总体 EM/F1 的分母包括所有可回答题，误拒计零；同时给出条件 EM/F1 和回答覆盖率。
- F1 按词频计交集，中文按字，数字保留小数/正负号/百分号；`0.84` 不等于 `84`。
- 页 Recall@1/3/5 与固定分母 Precision@1/3/5。少于 k 个结果时空位记零。
- 最终引用得分只统计交付答案，不能把被拒绝草稿的引用当作成功输出。
- 拒答 P/R/F1、不可回答题被回答比例、全样本端到端 EM 正确率、覆盖率、选择性 EM 风险。
- 风险覆盖曲线仅对已生成且非拒答的候选草稿计算，明确输出样本数；提前拒答没有反事实答案，不能伪造其正确率。
- 95% 区间按 PDF 聚类 bootstrap，避免把同 PDF 多题视为独立文档；不足两个 PDF 不给区间。
  显著性比较仍需配对的文档级差值分析，现有单方法区间不能替代它。

## 5. 真实中文数据的标注与划分

建议起步规模为约 40 份有使用许可的文档、300 道人工复核问答，行业报告与学位论文各占一部分。
这只是建议规模，**仓库尚没有这套真实数据**。最终样本量应根据开发集方差与预期效应决定。

按原生文本、扫描件、表格、图示、跨页、数字单位、否定、近领域不可回答分层统计。
建议不可回答题约占 30%，同时包含“文档提到了实体但没有所问属性”和超出文档范围的问题。
不要只用天气/股价等容易的负例。答案与最小充分页集合由人工确认，抽样由第二人独立复核并记录分歧。

页码为物理 PDF 的 1-based 页序，不是印刷页码；无需为了适配默认长度删掉尾部答案。
默认取消 50 页截断，评测对任何超配置页数的 PDF 报错，而不是默默把可回答题变成不可回答。
内容哈希相同的 PDF 按整篇划到同一集合；同一论文的扫描版/电子版、同项目多版本需人工归为同组，
字节哈希不能识别这类语义重复。若还需训练或调试提示，应再建立独立训练/探索集合。

```json
{
  "items": [{
    "id": "report_001_q01",
    "pdf_path": "../pdfs/real/report_001.pdf",
    "question": "原文中可核实的问题",
    "answer": "人工核验的简短答案",
    "answerable": true,
    "gold_pages": [12],
    "category": "numeric_unit"
  }]
}
```

不可回答题 `answer` 为空字符串、`answerable` 为 JSON 布尔值 false，通常 `gold_pages: []`。
数据校验会检查 ID 唯一性、字段类型、页码范围、空数据集和预测/标注 ID 的完整对应。

## 6. 可运行步骤

以下路径相对于仓库根目录，Windows PowerShell 可直接运行。先用虚构样例验证环境：

```powershell
./scripts/setup_ocr.ps1
python scripts/setup_chinese_fixtures.py
python scripts/run_benchmark.py --config configs/chinese_cpu.yaml --dataset data/fixtures/chinese/dataset.json --output results/chinese_regression.json
python scripts/run_full_rag.py --config configs/gpu_6gb.yaml --pdf data/fixtures/chinese/industry_native.pdf --question "项目总投资是多少？" --output results/chinese_gpu.json
```

`chinese_cpu.yaml` 使用真实 Tesseract OCR（`auto` 只对稀疏文本页执行）和抽取式生成。
`ocr` 方法强制所有页 OCR；`text` 方法仅使用原生文本。两者不再混称 OCR 基线。
GPU 配置默认 `text_mode: native`；处理扫描件时设置 `text_mode: auto`、`tessdata: data/tessdata`。
OCR 模型数据来自官方 `tesseract-ocr/tessdata_fast`，保存在本项目，记录 SHA-256；
未安装语言包时给出明确错误，不能把空文本当成 OCR 成功。纯文本 CPU 模式不需要下载神经模型。

真实数据准备完成后：

```powershell
python scripts/prepare_chinese_eval.py --dataset data/eval/chinese_real.json --output-dir data/eval/chinese_split --dev-fraction 0.3
python scripts/run_benchmark.py --config configs/gpu_6gb.yaml --dataset data/eval/chinese_split/dev.json --no-retrieval-gate --output results/dev.json
python scripts/calibrate_refusal.py --dev-results results/dev.json --output results/refusal_threshold.json --max-false-refusal-rate 0.1
python scripts/run_benchmark.py --config configs/gpu_6gb.yaml --dataset data/eval/chinese_split/test.json --calibration results/refusal_threshold.json --output results/test.json
python scripts/run_benchmark.py --config configs/gpu_6gb.yaml --dataset data/eval/chinese_split/test.json --ablation-all --suite research --output-dir results/research_ablation
```

注意：上述套件默认使用配置中的演示阈值，并不自动为每个基线标定。正式比较必须为各检索设置在开发集
分别拟合并冻结阈值，或关闭检索阈值统一比较检索/核验机制。**不能把某一融合的阈值文件传给整个消融套件**。
CLIP 在本配置只是轻量视觉对照，尚未验证其中文检索质量；单凭这套配置不能声称达到 ColPali 或中文专用视觉编码器的效果。

结果保存配置、源代码哈希、包版本、模型 revision 字段、数据/文档哈希、GPU 信息、每题证据和耗时。
CLIP 与 Qwen 的 6GB 配置已固定为本机验证过的模型 revision。
ColPali 使用前仍应固定 `model_revision`，该重型路径未在本轮进行真实模型测试。
每题结束写入 `.partial.json` 检查点；完成后写正式 JSON 并移除检查点。当前不支持自动从检查点续跑。
模型每题按阶段加载/释放，耗时包括加载和首题建索引；比较延迟时应单独报告冷热条件。

## 7. 可以和不可以写入论文的结论

可以写：实现了中文文档的双通道检索、可靠性特征调权、断言证据绑定、失败反馈和预算限制；
建立了可复现、文档隔离的实验工具和指标口径，并给出了基线及待验证假设。

暂不可以写：方法已显著优于现有方法、已经解决幻觉、获得校准概率或严格风险保证、
生成与核验模型独立、虚构样例上的高分等价于真实中文文档泛化能力。
已有 10 道英文样例及新中文样例均属于软件回归数据；其中虚构的准确率/投资金额是 PDF 内容，不是本系统的实验结果。

最终创新是否成立取决于 H1/H2 在真实、独立、人工标注的测试集上是否成立；
若没有增益，应报告负结果并调整方法，而不是仅增加系统组件或修改指标口径。

# Experiment log

## Setup

- Date: 2026-09-01
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU (6GB)
- System RAM: ~16GB (often ~6GB free; ColPali load hits pagefile / ACCESS_VIOLATION)
- Code path: laptop-safe `CLIP + extractive + lexical verify`
- Config: `configs/laptop.yaml`

## Datasets

- Self-built: `data/eval/self_built_dataset.json` (N=10, 7 answerable / 3 unanswerable)

## Commands

```bash
python scripts/setup_sample_data.py
python scripts/smoke_test.py
python scripts/run_full_rag.py --demo --pdf data/pdfs/sample_doc.pdf --question "What is the main contribution of this document?"
python scripts/run_benchmark.py --laptop --dataset data/eval/self_built_dataset.json --method demo --output results/self_built_demo.json
python scripts/run_benchmark.py --laptop --dataset data/eval/self_built_dataset.json --ablation-demo --output-dir results/ablation_demo
python scripts/report_results.py --input-dir results/ablation_demo
```

## Results (demo ablation)

See `results/ablation_demo.md`. Key observation on this machine:

- `demo_full` keeps high citation F1 and stronger refusal metrics than `demo_no_refusal` / `demo_no_grounding`.
- Exact Match is limited by extractive span matching vs short gold answers; F1 / citation / refusal are the more informative laptop metrics.
- Full ColPali + Qwen2-VL needs more system RAM (~16GB free recommended). Keep `runtime.visual_backend=colpali` and `generation_backend=vlm` for the GPU paper path when hardware allows.

## Error analysis

- ColPali/Qwen load: Windows error 1455 (pagefile too small) / native crash when free RAM ~6GB.
- Fallback: CLIP visual + BM25 hybrid + extractive cited answers + lexical Cite-then-Verify.
- Unanswerable miss cases: questions with weak lexical cues may still extract a tangential sentence unless refusal/model gate fires.

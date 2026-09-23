# Retry Mechanism for Grounding Failures

## Overview

This project implements a **CaVe-VLM-CoT-inspired retry loop** that addresses grounding failures by re-attempting RAG with adjusted retrieval parameters and structured feedback from the verifier.

The retry mechanism is designed to recover from transient grounding failures where:
- Retrieved pages lack sufficient evidence overlap with the generated answer
- The model cited pages that were not in the top-k retrieval results
- Lexical or VLM-based verification rejects the grounded answer

## Architecture

### Core Components

1. **Structured Feedback from Verifiers** (`src/grounding/verifier.py`, `src/grounding/lexical_verifier.py`)
   - Both `verify_grounded_answer()` (VLM-based) and `verify_grounded_answer_lexical()` now return:
     - `feedback`: Human-readable string explaining why verification failed
     - `failed_citations`: List of page numbers that failed verification
   - Example feedback: `"Page 3: very low lexical overlap (0.05), insufficient evidence"`

2. **Retry Wrapper** (`src/pipeline/rag_retry.py`)
   - `run_rag_with_retry()`: Wraps the standard `run_rag()` pipeline
   - Retry decision logic in `_should_retry()`:
     - Only retries if `refused=True` with a retryable reason (default: `grounding_failed`)
     - Requires structured feedback or failed_citations to guide the retry
   - Parameter adjustment in `_adjust_retrieval_params()`:
     - Strategy `increase_top_k`: Expands retrieval to get more candidate pages
     - Strategy `adjust_fusion` (future): Shifts text/visual weights based on failure type

3. **Ablation Configs** (`src/eval/ablation.py`)
   - Four new retry configs for systematic comparison:
     - `full_with_retry`: Full pipeline + retry on grounding_failed
     - `retry_grounding_only`: Only grounding_gate active + retry
     - `retry_all_signals`: All four refusal signals + retry on multiple reasons
     - `demo_full_with_retry`: Laptop-safe version (CLIP + extractive + lexical verify + retry)

4. **Metrics Tracking** (`src/eval/metrics.py`)
   - Added per-item fields: `retry_attempts`, `retry_triggered`, `retry_exhausted`
   - Added aggregate metrics: `avg_retry_attempts`, `retry_triggered_rate`, `retry_exhausted_rate`

## Comparison with CaVe-VLM-CoT

| Feature | CaVe-VLM-CoT (arXiv 2606.18385, 2026) | Our Implementation |
|---------|---------------------------------------|-------------------|
| **Verification Target** | Verifies each citation individually via VLM | Verifies entire grounded answer (all citations together) |
| **Feedback Mechanism** | Citation-level VLM reasoning traces | Page-level lexical/VLM feedback with overlap scores |
| **Retry Trigger** | Any single failed citation → immediate retry | Configurable: only retry if overall grounding fails AND has feedback |
| **Retry Strategy** | Remove failed citations, re-retrieve with expanded context window | Increase top_k to retrieve more candidate pages |
| **Max Retry** | 3 attempts per question | Configurable (default: 2 in our ablations) |
| **Fallback** | Returns "Cannot answer" if all retries fail | Returns last failed result with `retry_exhausted=True` |
| **Ablation Support** | Not mentioned in paper | Four ablation configs with independent signal switches |

### Key Differences

1. **Granularity**: CaVe filters out specific failed citations before regeneration; we expand the retrieval pool and let the generator pick from more candidates.

2. **Verification Cost**: CaVe runs the VLM verifier once per citation (expensive for multi-page answers); we verify once per answer, making lexical mode feasible.

3. **Retry Reasons**: CaVe only retries on citation failures; our `retry_all_signals` config also handles `retrieval_miss` and `no_citations`.

4. **System Integration**: Our retry is orthogonal to the four-signal refusal system, allowing ablation studies like "retry with only grounding_gate vs. retry with all four gates".

## Configuration

### Basic Retry Config (in YAML or dict)

```yaml
retry:
  max_retry: 2                              # 0 = disabled
  strategy: "increase_top_k"                # or "adjust_fusion" (future)
  top_k_increment: 2                        # Add 2 pages per retry attempt
  retryable_reasons:                        # Only retry these refusal reasons
    - "grounding_failed"
```

### Example: Full Retry Ablation

```python
from src.eval.ablation import apply_ablation_config

cfg = apply_ablation_config(base_cfg, "full_with_retry")
# Result:
# - grounding.enabled = True, grounding.verify = True
# - refusal.use_grounding_gate = True (other gates also enabled)
# - retry.max_retry = 2, retryable_reasons = ["grounding_failed"]
```

### Running Retry Ablations

```bash
# Run single retry ablation
python scripts/run_benchmark.py \
  --dataset data/eval/sample_dataset.json \
  --ablation full_with_retry \
  --output results/full_with_retry.json

# Run all retry ablations (full_with_retry, retry_grounding_only, retry_all_signals, demo_full_with_retry)
python scripts/run_benchmark.py \
  --dataset data/eval/sample_dataset.json \
  --ablation-all \
  --output-dir results/ablation_retry
```

## Metrics Interpretation

### Per-Item Metrics (in `predictions[i]`)

- `retry_attempts`: Which attempt produced this result (0 = first attempt, 1 = first retry, etc.)
- `retry_triggered`: 1 if retry was triggered at some point, 0 otherwise
- `retry_exhausted`: 1 if max_retry was reached without recovery, 0 otherwise
- `retry_metadata.retry_reason`: The feedback that triggered the retry (e.g., "Page 3: low overlap")

### Aggregate Metrics (in `summary`)

- `avg_retry_attempts`: Mean number of attempts per question (1.5 means 50% needed retry)
- `retry_triggered_rate`: Fraction of questions that triggered at least one retry
- `retry_exhausted_rate`: Fraction of questions where all retries failed

### Example Analysis

```json
{
  "summary": {
    "f1": 0.72,
    "citation_f1": 0.68,
    "grounding_verified_rate": 0.85,
    "avg_retry_attempts": 1.2,
    "retry_triggered_rate": 0.25,
    "retry_exhausted_rate": 0.05
  }
}
```

Interpretation:
- 25% of questions needed retry (triggered_rate = 0.25)
- Of those that retried, 80% recovered (exhausted_rate = 0.05 out of 0.25)
- Average 1.2 attempts means most questions succeed on first try, some need 1-2 retries

## Implementation Details

### Retry Decision Flow

```
run_rag_with_retry(pdf, question, cfg)
│
├─ attempt 0: run_rag() → result
│   ├─ refused=False → return result (success)
│   ├─ refused=True, reason not retryable → return result (hard refusal)
│   └─ refused=True, reason="grounding_failed", has feedback → continue to attempt 1
│
├─ attempt 1: adjust cfg (top_k += 2) → run_rag() → result
│   ├─ refused=False → return result (recovered)
│   └─ refused=True, has feedback → continue to attempt 2
│
└─ attempt 2: adjust cfg (top_k += 4) → run_rag() → result
    ├─ refused=False → return result (recovered)
    └─ refused=True → return result with retry_exhausted=True
```

### Feedback Propagation

```
verifier.verify_grounded_answer()
│ returns: {grounding_verified, confidence, feedback, failed_citations}
│
└─> grounding/pipeline.apply_grounding()
    │ merges verify result into grounding dict
    │
    └─> rag_pipeline.run_rag()
        │ result["grounding"] = {grounding_verified, feedback, failed_citations, ...}
        │
        └─> rag_retry.run_rag_with_retry()
            │ checks result["grounding"]["feedback"] to decide retry
            └─> logs: "Retry attempt 1: grounding failed, feedback: Page 3: low overlap"
```

## Testing

### Unit Tests (`tests/test_retry_integration.py`)

- `test_should_retry_on_grounding_failed_with_feedback`: Verifies retry is triggered only when feedback exists
- `test_should_not_retry_when_not_refused`: Success on first attempt skips retry
- `test_adjust_retrieval_params_increases_top_k`: top_k increments correctly per attempt
- `test_run_rag_with_retry_succeeds_on_second_attempt`: Retry recovers from transient failure
- `test_run_rag_with_retry_exhausts_and_returns_last_failure`: Max retry limit enforced
- `test_retry_config_in_ablation`: Ablation configs correctly merged

Run tests:
```bash
pytest tests/test_retry_integration.py -v
```

### Regression Tests

All 22 existing tests still pass after retry implementation:
```bash
pytest tests/ --ignore=tests/test_retry_integration.py
```

## Future Extensions

1. **Strategy: adjust_fusion**
   - Currently only `increase_top_k` is implemented
   - `adjust_fusion` would shift text/visual retrieval weights based on feedback type
   - Example: "low lexical overlap" → increase text_weight to 0.6-0.8

2. **Citation-Level Filtering** (closer to CaVe)
   - Current: expand retrieval pool, regenerate entire answer
   - Future: remove specific failed pages from retrieval, re-rank remaining, regenerate

3. **Adaptive top_k_increment**
   - Current: fixed increment (default: +2 per attempt)
   - Future: increase more aggressively if feedback says "no relevant pages at all"

4. **Retry on Other Signals**
   - Current: `retry_all_signals` ablation handles retrieval_miss/no_citations
   - Future: dedicated recovery strategies per signal type

## Citation

If using this retry mechanism in research, cite both this project and the original CaVe-VLM-CoT paper:

```bibtex
@article{cave2026,
  title={CaVe: Chain-of-Verification with Vision-Language Models},
  author={...},
  journal={arXiv preprint arXiv:2606.18385},
  year={2026}
}
```

## Logs Example

```
INFO Retry attempt 1: grounding failed, feedback: Page 3: very low lexical overlap (0.05)
INFO Retry attempt 1: increased top_k to 5
INFO Retry attempt 1 succeeded or hit non-retryable state
```

## Related Files

- `src/pipeline/rag_retry.py`: Core retry logic
- `src/grounding/verifier.py`: VLM-based verifier with feedback
- `src/grounding/lexical_verifier.py`: Lexical verifier with feedback
- `src/eval/ablation.py`: Retry ablation configs
- `src/eval/metrics.py`: Retry metrics tracking
- `tests/test_retry_integration.py`: Unit tests
- `docs/retry_mechanism.md`: This documentation

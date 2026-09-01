"""Smoke tests that do not require GPU models."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_imports():
    from src.eval.ablation import list_ablation_configs, apply_ablation_config
    from src.eval.metrics import score_prediction, aggregate_metrics
    from src.grounding.citation import extract_cited_pages, is_unanswerable
    from src.refusal.decision import apply_refusal
    from src.refusal.score_gate import max_retrieval_score
    from src.retrieval.hybrid_fusion import fuse_retrieval
    from src.utils.config import load_config
    from src.utils.runtime import choose_backends

    cfg = load_config("configs/default.yaml")
    assert cfg["retrieval"]["top_k"] == 5
    assert cfg["refusal"].get("use_citation_gate") is True
    assert len(list_ablation_configs()) >= 8
    backends = choose_backends(
        {"runtime": {"visual_backend": "clip", "generation_backend": "extractive"}}
    )
    assert backends["visual_backend"] == "clip"

    ans = "Accuracy is 87.3% [p.3] on the benchmark."
    assert extract_cited_pages(ans) == [3]
    assert not is_unanswerable(ans)
    assert is_unanswerable("UNANSWERABLE")

    fused = fuse_retrieval(
        [{"page_index": 0, "page_path": "a.png", "score": 0.9}],
        [{"page_index": 1, "page_path": "b.png", "score": 0.5, "text_snippet": "hello"}],
        page_text_densities={0: 10, 1: 100},
        top_k=2,
    )
    assert len(fused) == 2

    low_hits = [{"fused_score": 0.1}, {"fused_score": 0.05}]
    assert max_retrieval_score(low_hits) == 0.1
    refusal = apply_refusal(
        low_hits, refusal_cfg={"enabled": True, "score_threshold": 0.25}
    )
    assert refusal["refused"] is True

    empty_refusal = apply_refusal([], refusal_cfg={"enabled": True, "score_threshold": 0.3})
    assert empty_refusal["refused"] is True

    metrics = score_prediction(
        {
            "answer": "87.3%",
            "raw_answer": "87.3% [p.3]",
            "refused": False,
            "answerable": True,
            "retrieved_pages": [{"page_index": 2, "page_number": 3}],
            "grounding": {
                "citations": [{"page": 3}],
                "grounding_verified": True,
                "confidence": 1.0,
            },
        },
        {"id": "q", "answer": "87.3%", "answerable": True, "gold_pages": [3]},
    )
    assert metrics["em"] == 1
    assert metrics["citation_f1"] == 1.0

    summary = aggregate_metrics([metrics])
    assert summary["count"] == 1

    cfg2 = apply_ablation_config(cfg, "no_grounding")
    assert cfg2["grounding"]["enabled"] is False
    print("imports: OK")


def test_sample_data_exists():
    root = Path(__file__).resolve().parent.parent
    dataset = root / "data" / "eval" / "sample_dataset.json"
    pdf = root / "data" / "pdfs" / "sample_doc.pdf"
    if not dataset.exists() or not pdf.exists():
        from scripts.setup_sample_data import main as setup

        setup()

    with open(dataset, encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["items"]) >= 8
    assert any(not item["answerable"] for item in data["items"])
    assert pdf.exists()
    print("sample data: OK")


def main():
    test_imports()
    test_sample_data_exists()
    print("\nAll smoke tests passed. Ready to run:")
    print(
        '  python scripts/run_full_rag.py --demo --pdf data/pdfs/sample_doc.pdf '
        '--question "What is the main contribution?"'
    )
    print(
        "  python scripts/run_benchmark.py --laptop "
        "--dataset data/eval/self_built_dataset.json --ablation-demo"
    )
    print(
        "  # Full GPU (needs ~16GB system RAM for ColPali):\n"
        '  python scripts/run_full_rag.py --pdf data/pdfs/sample_doc.pdf '
        '--question "What is the main contribution?"'
    )


if __name__ == "__main__":
    main()

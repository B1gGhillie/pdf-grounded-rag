"""Pipeline behaviour that must remain testable without PyTorch or model weights."""

import pytest
import copy

from src.pipeline import rag_pipeline
from src.pipeline.rag_retry import run_rag_with_retry
from src.eval.ablation import apply_ablation_config


pytestmark = pytest.mark.integration


def test_full_pipeline_uses_extractive_answer_and_verifies_citation(monkeypatch, tmp_path):
    """Exercise the orchestration boundary using deterministic collaborators."""
    pdf_path = tmp_path / "report.pdf"
    pdf_path.touch()
    hits = [
        {
            "page_index": 0,
            "page_number": 1,
            "page_path": "page_0001.png",
            "fused_score": 0.95,
            "visual_score": 0.9,
            "text_score": 1.0,
            "text_snippet": "The project uses page-level grounding.",
        }
    ]

    monkeypatch.setattr(
        rag_pipeline,
        "choose_backends",
        lambda cfg: {
            "visual_backend": "clip",
            "generation_backend": "extractive",
            "device": "cpu",
            "probe": {"free_ram_gb": 32.0},
        },
    )
    monkeypatch.setattr(
        rag_pipeline,
        "ingest_document",
        lambda *args, **kwargs: (["page_0001.png"], None, None, {0: 100}),
    )
    monkeypatch.setattr(
        rag_pipeline,
        "_retrieve_hybrid",
        lambda *args, **kwargs: ([], [], [dict(hit) for hit in hits]),
    )
    monkeypatch.setattr(
        rag_pipeline,
        "answer_extractive_with_citations",
        lambda question, retrieved_hits: "The project uses page-level grounding [p.1]",
    )

    cfg = {
        "retrieval": {"top_k": 1},
        "runtime": {"visual_backend": "clip", "generation_backend": "extractive"},
        "grounding": {"enabled": True, "verify": True, "verify_mode": "lexical"},
        "refusal": {"enabled": True, "score_threshold": 0.25},
    }
    result = rag_pipeline.run_rag(str(pdf_path), "What does it use?", cfg)

    assert result["answer"] == "The project uses page-level grounding"
    assert result["grounding"]["grounding_verified"] is True
    assert result["refused"] is False


def test_missing_vlm_citation_is_repaired_only_for_a_real_claim():
    hits = [{"page_index": 2, "page_number": 3, "fused_score": 0.8,
             "text": "Hybrid RAG reached 87.3%."}]

    repaired, changed = rag_pipeline._repair_missing_citation(
        "Hybrid RAG reached 87.3%.", hits, {"repair_missing_citations": True}
    )
    assert changed is True
    assert repaired == "Hybrid RAG reached 87.3%. [p.3]"

    bare_page, changed = rag_pipeline._repair_missing_citation(
        "[3]", hits, {"repair_missing_citations": True}
    )
    assert changed is False
    assert bare_page == "[3]"
    _, changed = rag_pipeline._repair_missing_citation(
        "Hybrid RAG reached 99.9%.", hits, {"repair_missing_citations": True}
    )
    assert changed is False
    _, changed = rag_pipeline._repair_missing_citation(
        "Hybrid RAG reached 87.3%.", [{"page_number": 3}], {"repair_missing_citations": True}
    )
    assert changed is False


def _stub_evidence(monkeypatch, tmp_path):
    path = tmp_path / "fixture.pdf"
    path.touch()
    monkeypatch.setattr(rag_pipeline, "choose_backends", lambda cfg: {
        "visual_backend": "none", "generation_backend": "extractive", "device": "cpu", "probe": {}})
    monkeypatch.setattr(rag_pipeline, "ingest_document", lambda *a, **k: (["p1.png", "p2.png"], None, None, {}))
    hits = [{"page_index": 0, "page_number": 1, "fused_score": .4, "evidence_confidence": .4,
             "text": "项目投资为350万元。", "page_path": "p1.png"},
            {"page_index": 1, "page_number": 2, "fused_score": .3, "evidence_confidence": .3,
             "text": "设备成本为240万元。", "page_path": "p2.png"}]
    monkeypatch.setattr(rag_pipeline, "_retrieve_hybrid", lambda q, p, b, e, cfg, v: ([], [], copy.deepcopy(hits[:cfg["retrieval"]["top_k"]])))
    cfg = {"retrieval": {"top_k": 1}, "grounding": {"enabled": True, "verify": True, "verify_mode": "lexical"},
           "refusal": {"enabled": True, "score_threshold": .1}}
    return path, cfg


def test_no_grounding_ablation_still_applies_retrieval_gate(monkeypatch, tmp_path):
    path, cfg = _stub_evidence(monkeypatch, tmp_path)
    cfg["refusal"]["score_threshold"] = .9
    ablated = apply_ablation_config(cfg, "no_grounding")
    result = rag_pipeline.run_rag(path, "项目投资？", ablated, method=ablated["_method"])
    assert result["refused"] and result["refusal_reason"] == "retrieval_miss"
    no_refusal = apply_ablation_config(cfg, "no_refusal")
    accepted = rag_pipeline.run_rag(path, "项目投资？", no_refusal, method=no_refusal["_method"])
    assert not accepted["refused"]


def test_real_grounding_failure_drives_retry_without_mutating_config(monkeypatch, tmp_path):
    path, cfg = _stub_evidence(monkeypatch, tmp_path)
    cfg["retry"] = {"max_retry": 2, "strategy": "evidence_budget", "top_k_increment": 1,
                    "max_top_k": 3, "max_total_pages": 6, "retryable_reasons": ["grounding_failed"]}
    original = copy.deepcopy(cfg)
    monkeypatch.setattr(rag_pipeline, "answer_extractive_with_citations", lambda q, hits:
        "设备成本为240万元。[p.2]" if len(hits) == 2 else "项目投资为999万元。[p.1]")
    result = run_rag_with_retry(path, "设备成本是多少？", cfg)
    trace = result["retry_metadata"]
    assert not result["refused"] and trace["retry_triggered"]
    assert "numeric_mismatch" in trace["history"][0]["feedback"]
    assert [h["top_k"] for h in trace["history"]] == [1, 2]
    assert trace["total_evidence_pages"] == 3 and cfg == original

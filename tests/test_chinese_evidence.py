"""Failure cases that motivated Chinese retrieval and claim verification."""
import pytest
from src.retrieval.bm25_retriever import BM25Retriever, tokenize
from src.utils.text import evidence_window, sentences
from src.grounding.claims import check_text_support, verify_claim_units, split_claims
from src.generation.extractive_generator import answer_extractive_with_citations
from src.retrieval.hybrid_fusion import fuse_retrieval
from src.refusal.score_gate import should_refuse_retrieval

pytestmark = pytest.mark.unit


def test_chinese_retrieval_without_spaces_and_single_page():
    index = BM25Retriever().build_index(["1.png"], [{"page": 1, "text": "项目总投资为350万元。"}])
    assert "投资" in tokenize("项目总投资为350万元")
    hits = index.retrieve("项目投资是多少万元？")
    assert hits[0]["page_index"] == 0
    assert hits[0]["score"] > 0
    assert index.retrieve("火星天气") == []


def test_query_window_can_find_evidence_beyond_first_300_characters():
    text = "这是与问题无关的章节内容。" * 100 + "项目总投资为350万元。"
    assert "350万元" in evidence_window(text, "项目总投资是多少？", max_chars=100)
    assert sentences("统计范围不包含办公\n楼。") == ["统计范围不包含办公楼。"]
    assert sentences("甲段\n\n乙段") == ["甲段", "乙段"]


@pytest.mark.parametrize("claim,evidence,reason", [
    ("项目投资350亿元", "项目投资350万元。", "numeric_mismatch"),
    ("召回率为0.84", "召回率为84。", "numeric_mismatch"),
    ("统计范围包含办公楼", "统计范围不包含办公楼。", "negation_mismatch"),
    ("ColPali training details", "Training details for ColPali are omitted.", "explicit_absence"),
])
def test_lexical_overlap_does_not_override_contradictions(claim, evidence, reason):
    result = check_text_support(claim, evidence)
    assert result["supported"] is False
    assert result["reason"] == reason


def test_claims_are_bound_to_their_own_citations_and_uncited_tail_fails():
    hits = [{"page_number": 1, "text": "投资350万元。"}, {"page_number": 2, "text": "样本500个。"}]
    assert verify_claim_units("投资350万元[p.1]。样本500个[p.2]", hits)["grounding_verified"]
    wrong = verify_claim_units("投资350万元[p.2]。样本500个[p.1]", hits)
    assert not wrong["grounding_verified"]
    tail = verify_claim_units("投资350万元[p.1]。样本500个。", hits)
    assert tail["claim_coverage"] == 0.5
    assert tail["claim_results"][-1]["reason"] == "uncited_claim"


def test_multiple_citations_pass_joint_evidence_to_verifier():
    observed = []
    def verify(claim, pages, question):
        observed.append([p["page_number"] for p in pages])
        return {"supported": True, "reason": "test_joint_verifier"}
    hits = [{"page_number": 1}, {"page_number": 3}]
    result = verify_claim_units("两阶段合计20万元[p.1][p.3]。", hits, verifier=verify)
    assert result["grounding_verified"] and observed == [[1, 3]]
    assert split_claims("准确率91.5% [p.1]")[0]["claim"] == "准确率91.5%"


def test_extractive_generator_rejects_explicit_missing_information():
    hits = [{"page_number": 1, "text": "本报告未披露项目的贷款利率。项目总投资350万元。"}]
    assert answer_extractive_with_citations("项目贷款利率是多少？", hits) == "UNANSWERABLE"
    english = [{"page_number": 1, "text": "Training details for ColPali are omitted."}]
    assert answer_extractive_with_citations("What training hyperparameters were used for ColPali?", english) == "UNANSWERABLE"


def test_confidence_does_not_renormalize_with_topk_or_weak_candidates():
    visual = [{"page_index": 0, "score": .05}, {"page_index": 1, "score": .001}]
    one = fuse_retrieval(visual, [], top_k=1, mode="reliability")
    two = fuse_retrieval(visual, [], top_k=2, mode="reliability")
    assert one[0]["evidence_confidence"] == two[0]["evidence_confidence"] == .05
    assert should_refuse_retrieval(one, .1)
    rrf = fuse_retrieval(visual, [], top_k=2, mode="rrf")
    assert rrf[0]["fused_score"] != rrf[0]["evidence_confidence"]


def test_query_reliability_downweights_dense_irrelevant_text():
    visual = [{"page_index": 0, "score": .3}, {"page_index": 1, "score": .3}]
    text = [{"page_index": 0, "score": 10}, {"page_index": 1, "score": 10}]
    features = {0: {"query_coverage": 0, "text_quality": 1}, 1: {"query_coverage": 1, "text_quality": 1}}
    hits = fuse_retrieval(visual, text, {0: 1000, 1: 1000}, mode="reliability", page_features=features)
    by_page = {h["page_index"]: h for h in hits}
    assert by_page[0]["text_weight"] < by_page[1]["text_weight"]

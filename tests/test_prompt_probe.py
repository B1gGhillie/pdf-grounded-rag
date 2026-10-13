import pytest
from PIL import Image

from src.generation.vlm_generator import answer_with_grounding
from src.grounding.pipeline import apply_grounding

pytestmark = pytest.mark.unit


def test_alternate_prompt_preserves_page_order_text_and_unknown_page_rejection(tmp_path, monkeypatch):
    paths = [tmp_path / "one.png", tmp_path / "two.png"]
    for path in paths:
        Image.new("RGB", (8, 8), "white").save(path)
    captured = []

    def generate(model, processor, messages, images, max_new_tokens):
        captured.append(messages[0]["content"])
        return "Claim [p.99]"

    monkeypatch.setattr("src.generation.vlm_generator._generate", generate)
    answer = answer_with_grounding(None, None, "Which entity?", paths,
        page_numbers=[17, 4], evidence_texts=["First evidence", "Second evidence"], prompt_language="en")
    content = captured[0]
    assert [entry["text"] for entry in content if entry["type"] == "text"][:2] == ["Document page 17.", "Document page 4."]
    assert "[Page 17] First evidence" in content[-1]["text"]
    assert "[Page 4] Second evidence" in content[-1]["text"]
    result = apply_grounding(None, None, answer, [{"page_number": 17}, {"page_number": 4}],
                             {"verify_mode": "vlm", "prompt_language": "en"})
    assert result["verification_reason"] == "citation_outside_retrieval"
    assert not result["grounding_verified"]


def test_alternate_verifier_still_requires_yes_and_preserves_short_answer_context(tmp_path, monkeypatch):
    path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), "white").save(path)
    hits = [{"page_number": 2, "page_path": str(path), "text": "The amount is 5 dollars."}]
    captured = []

    def generate(model, processor, messages, images, max_new_tokens):
        captured.append(messages[0]["content"][-1]["text"])
        return "NO"

    monkeypatch.setattr("src.generation.vlm_generator._generate", generate)
    result = apply_grounding(object(), object(), "5 dollars [p.2]", hits,
        {"verify_mode": "vlm", "prompt_language": "en", "granularity": "claim"}, question="What is the amount?")
    assert not result["grounding_verified"]
    assert "What is the amount?" in captured[0] and "5 dollars" in captured[0]
    assert result["verification_results"][0]["raw_response"] == "NO"

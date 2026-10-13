from src.generation.vlm_generator import answer_with_grounding, answer_with_text_context, answer_with_vlm


def test_text_generation_refuses_empty_retrieval():
    assert answer_with_text_context(None, None, "问题", "(no relevant text found)") == "UNANSWERABLE"


def test_visual_generation_refuses_without_pages():
    assert answer_with_vlm(None, None, "问题", []) == "UNANSWERABLE"
    assert answer_with_grounding(None, None, "问题", []) == "UNANSWERABLE"

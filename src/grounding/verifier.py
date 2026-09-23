"""Cite-then-Verify: check whether cited pages support the generated answer."""

from PIL import Image

from src.grounding.citation import page_path_by_number


def _vlm_yes_no(model, processor, page_path, page_number, prompt, max_new_tokens=8):
    """Run a single-page YES/NO verification call."""
    image = Image.open(page_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    reply = processor.batch_decode(generated, skip_special_tokens=True)[0].strip().upper()
    supported = reply.startswith("YES")
    return supported, reply


def verify_claim_on_page(
    model,
    processor,
    claim,
    page_path,
    page_number,
    max_new_tokens=8,
):
    """Ask the VLM whether a single page supports a factual claim."""
    prompt = (
        "You are verifying whether a document page supports a claim.\n"
        "Document page number: {0}\n\n"
        "Claim: {1}\n\n"
        "Does this page contain clear evidence that supports the claim? "
        "Reply with exactly YES or NO."
    ).format(page_number, claim)
    supported, raw = _vlm_yes_no(model, processor, page_path, page_number, prompt, max_new_tokens)
    return {"supported": supported, "raw_response": raw, "page": page_number}


def verify_grounded_answer(
    model,
    processor,
    answer,
    cited_pages,
    retrieved_hits,
    verification_threshold=0.5,
):
    """Verify each cited page against the answer (Cite-then-Verify loop).

    Returns:
        grounding_verified: bool
        confidence: fraction of cited pages that passed verification
        verification_results: per-page details
    """
    if not cited_pages:
        return {
            "grounding_verified": False,
            "confidence": 0.0,
            "verification_results": [],
            "verification_reason": "no_citations",
        }

    paths = page_path_by_number(retrieved_hits)
    results = []
    passed = 0

    for page in cited_pages:
        page_path = paths.get(page)
        if not page_path:
            results.append(
                {
                    "page": page,
                    "supported": False,
                    "raw_response": "PAGE_NOT_IN_RETRIEVAL",
                    "reason": "page_not_retrieved",
                }
            )
            continue

        result = verify_claim_on_page(model, processor, answer, page_path, page)
        result["reason"] = "verified" if result["supported"] else "evidence_not_found"
        results.append(result)
        if result["supported"]:
            passed += 1

    confidence = passed / len(cited_pages)
    grounding_verified = confidence >= verification_threshold and passed > 0

    reason = "verified" if grounding_verified else "verification_failed"
    if any(r.get("reason") == "page_not_retrieved" for r in results):
        reason = "citation_outside_retrieval"

    # Build structured feedback for retry
    failed_pages = [r["page"] for r in results if not r.get("supported", False)]
    feedback_parts = []
    if failed_pages:
        feedback_parts.append("Failed pages: {0}".format(", ".join("p.{0}".format(p) for p in failed_pages)))
        for r in results:
            if not r.get("supported", False):
                raw = r.get("raw_response", "")
                if r.get("reason") == "page_not_retrieved":
                    feedback_parts.append("Page {0}: not in retrieved set".format(r["page"]))
                elif "NO" in raw.upper():
                    feedback_parts.append("Page {0}: VLM determined no evidence".format(r["page"]))
    feedback = "; ".join(feedback_parts) if feedback_parts else None

    return {
        "grounding_verified": grounding_verified,
        "confidence": round(confidence, 4),
        "verification_results": results,
        "verification_reason": reason,
        "failed_citations": failed_pages,
        "feedback": feedback,
    }

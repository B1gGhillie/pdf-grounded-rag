"""VLM-based answer generation from retrieved document pages."""

from __future__ import annotations

import logging

from PIL import Image

logger = logging.getLogger(__name__)


def _build_page_messages(images, page_numbers, prompt):
    """Build multimodal chat messages with optional page-number labels."""
    content = []
    for image, page_num in zip(images, page_numbers):
        content.append({"type": "image", "image": image})
        if page_num is not None:
            content.append({"type": "text", "text": "Document page {0}.".format(page_num)})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _generate(model, processor, messages, images, max_new_tokens):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images or None, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def answer_with_vlm(model, processor, query, page_paths, max_new_tokens=512):
    """Generate an answer from page images and a question."""
    images = [Image.open(p).convert("RGB") for p in page_paths]
    prompt = (
        "Based on the document page images, answer the question concisely. "
        "If the pages do not contain enough information, reply exactly: UNANSWERABLE\n\n"
        "Question: {0}".format(query)
    )
    messages = _build_page_messages(images, [None] * len(images), prompt)
    return _generate(model, processor, messages, images, max_new_tokens)


def answer_with_grounding(
    model,
    processor,
    query,
    page_paths,
    page_numbers=None,
    max_new_tokens=512,
    cite_format="[p.{page}]",
):
    """Generate a grounded answer with mandatory page citations.

    Args:
        page_paths: retrieved page image paths (same order as page_numbers)
        page_numbers: 1-based document page numbers for each image
        cite_format: citation template shown in the prompt (default [p.{page}])
    """
    images = [Image.open(p).convert("RGB") for p in page_paths]
    if page_numbers is None:
        page_numbers = list(range(1, len(page_paths) + 1))

    example_cite = cite_format.format(page=page_numbers[0]) if page_numbers else "[p.1]"
    allowed = ", ".join(str(p) for p in page_numbers)
    prompt = (
        "You are a document QA assistant. Answer using ONLY the provided page images.\n"
        "Rules:\n"
        "1. Cite every factual claim with the document page number using format {0}.\n"
        "2. Only cite these page numbers: {1}.\n"
        "3. Put citations immediately after the supported claim, e.g. 'The accuracy is 87.3% {0}'.\n"
        "4. If the pages do not contain enough information, reply exactly: UNANSWERABLE\n\n"
        "Question: {2}"
    ).format(example_cite, allowed, query)

    messages = _build_page_messages(images, page_numbers, prompt)
    return _generate(model, processor, messages, images, max_new_tokens)


def answer_with_text_context(model, processor, query, text_context, max_new_tokens=512):
    """OCR-RAG baseline: answer from retrieved text snippets (no page images)."""
    prompt = (
        "Based on the following document excerpts, answer the question concisely. "
        "If the excerpts do not contain enough information, reply exactly: UNANSWERABLE\n\n"
        "Document excerpts:\n{0}\n\nQuestion: {1}".format(text_context, query)
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return _generate(model, processor, messages, None, max_new_tokens)


def load_vlm(model_name, load_in_4bit=False, device=None):
    """Load Qwen2-VL model and processor with memory-safe defaults."""
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_name)

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig

            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quant,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            return model, processor
        except Exception as exc:
            logger.warning("4-bit load failed (%s); falling back to bf16/fp32", exc)

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        if device == "cpu":
            model = model.to(device)
    except Exception as exc:
        logger.error("VLM load failed: %s", exc)
        raise
    return model, processor

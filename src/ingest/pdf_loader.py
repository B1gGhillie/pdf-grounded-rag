"""PDF to page images and text extraction."""

from pathlib import Path

import pymupdf


def pdf_to_page_images(pdf_path, output_dir, dpi=144, max_pages=None):
    doc = pymupdf.open(pdf_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = []
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)

    for i, page in enumerate(doc):
        if max_pages is not None and i >= max_pages:
            break
        pix = page.get_pixmap(matrix=matrix)
        img_path = out / ("page_{0:04d}.png".format(i + 1))
        pix.save(str(img_path))
        paths.append(str(img_path))

    doc.close()
    return paths


def pdf_to_text_chunks(pdf_path, max_pages=None, text_mode="native", ocr_language="chi_sim+eng",
                       ocr_dpi=200, tessdata=None, min_native_chars=30):
    """Extract native text or run real Tesseract OCR through MuPDF.

    auto OCRs sparse pages; ocr OCRs every page for a fair OCR-only baseline.
    Missing language data raises an actionable error instead of silently
    treating a scanned page as an unanswerable document.
    """
    if text_mode not in {"native", "auto", "ocr"}:
        raise ValueError("text_mode must be native, auto or ocr")
    doc = pymupdf.open(pdf_path)
    chunks = []

    for i, page in enumerate(doc):
        if max_pages is not None and i >= max_pages:
            break
        text = page.get_text("text", sort=True).strip()
        source = "native"
        if text_mode == "ocr" or (text_mode == "auto" and len(text) < min_native_chars):
            try:
                tp = page.get_textpage_ocr(language=ocr_language, dpi=ocr_dpi,
                                         full=True, tessdata=tessdata)
                text = page.get_text("text", textpage=tp, sort=True).strip()
                source = "ocr"
            except Exception as exc:
                doc.close()
                raise RuntimeError(
                    f"OCR failed on page {i+1}; install Tesseract language data ({ocr_language}) "
                    "and set pdf.tessdata or TESSDATA_PREFIX. " + str(exc)
                ) from exc
        if text:
            chunks.append({"page": i + 1, "text": text, "text_source": source})

    doc.close()
    return chunks

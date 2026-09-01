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


def pdf_to_text_chunks(pdf_path, max_pages=None):
    doc = pymupdf.open(pdf_path)
    chunks = []

    for i, page in enumerate(doc):
        if max_pages is not None and i >= max_pages:
            break
        text = page.get_text("text").strip()
        if text:
            chunks.append({"page": i + 1, "text": text})

    doc.close()
    return chunks

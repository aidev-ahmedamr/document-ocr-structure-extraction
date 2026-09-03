import time
import pytesseract


def run_tesseract(image, lang="eng"):
    """
    Run Tesseract on a single preprocessed PIL image.
    Returns raw text plus per-word confidence data.
    """

    start = time.time()

    text = pytesseract.image_to_string(image, lang=lang)

    data = pytesseract.image_to_data(
        image, lang=lang, output_type=pytesseract.Output.DICT
    )

    confidences = [int(c) for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) >= 0]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    elapsed = time.time() - start

    return {
        "text": text.strip(),
        "avg_confidence": round(avg_confidence, 2),
        "processing_time_seconds": elapsed,
    }


def run_tesseract_multi_page(images, lang="eng"):
    """Run Tesseract on every page and concatenate the text."""

    results = [run_tesseract(img, lang=lang) for img in images]

    return {
        "text": "\n\n".join(r["text"] for r in results),
        "avg_confidence": round(
            sum(r["avg_confidence"] for r in results) / len(results), 2
        ) if results else 0.0,
        "processing_time_seconds": sum(r["processing_time_seconds"] for r in results),
    }

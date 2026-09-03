"""
Orchestrates both pipelines end-to-end for one document and produces
a single comparison report. This is what the API and dashboard call.
"""

from src.preprocessing.preprocess import preprocess_document
from src.ocr.tesseract_ocr import run_tesseract_multi_page
from src.extraction.regex_extraction import extract_fields
from src.ocr.vision_llm_ocr import run_vision_llm_multi_page
from src.extraction.compare import build_result, compare_results


def run_traditional_pipeline(pages):
    ocr_result = run_tesseract_multi_page(pages)
    fields = extract_fields(ocr_result["text"])
    return build_result(
        "tesseract",
        ocr_result["text"],
        fields,
        ocr_result["processing_time_seconds"],
    )


def run_llm_pipeline(pages):
    llm_result = run_vision_llm_multi_page(pages)
    return build_result(
        "qwen2-vl",
        llm_result["raw_text"],
        llm_result["fields"],
        llm_result["processing_time_seconds"],
        document_type=llm_result["document_type"],
    )


def process_document(file_path, dpi=200):
    """
    Full end-to-end run: preprocess once, then run both pipelines on
    the same preprocessed pages and compare their extracted fields.
    """

    pages = preprocess_document(file_path, dpi=dpi)

    traditional_result = run_traditional_pipeline(pages)
    llm_result = run_llm_pipeline(pages)

    comparison = compare_results(traditional_result, llm_result)

    return {
        "tesseract": traditional_result,
        "qwen2_vl": llm_result,
        "comparison": comparison,
    }

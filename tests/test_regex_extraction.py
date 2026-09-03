from src.extraction.regex_extraction import extract_fields


SAMPLE_TEXT = """
INVOICE INV-2026-0913
Date: 09/03/2026
Email: john.smith@example.com
Phone: +20 100 123 4567
Total: $95.50
"""


def test_extracts_dates():
    result = extract_fields(SAMPLE_TEXT)
    assert "09/03/2026" in result["dates"]


def test_extracts_amounts():
    result = extract_fields(SAMPLE_TEXT)
    assert 95.5 in result["amounts"]


def test_extracts_emails():
    result = extract_fields(SAMPLE_TEXT)
    assert "john.smith@example.com" in result["emails"]


def test_extracts_document_ids():
    result = extract_fields(SAMPLE_TEXT)
    assert "2026-0913" in result["document_ids"]


def test_empty_text_returns_empty_lists():
    result = extract_fields("")
    assert result["dates"] == []
    assert result["amounts"] == []

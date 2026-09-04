# 📄 Multimodal Document OCR & Structure Extraction Pipeline

An end-to-end pipeline that extracts structured data (dates, amounts,
emails, phone numbers, document IDs) from any document — scanned PDF,
photo, or image — using **two independent, fully open-source pipelines**,
compared side-by-side. No paid APIs involved anywhere.

## Overview

Most OCR projects pick one approach and stop. This project deliberately
runs the **same document through two different pipelines** and measures
how much they agree on every extracted field, because that comparison is
the real signal of whether an extraction pipeline is trustworthy enough
for production.

| | Pipeline A: Traditional | Pipeline B: Vision-LLM |
|---|---|---|
| **Approach** | Tesseract OCR → regex-based field extraction | Qwen2-VL-2B-Instruct (open-source vision-language model) reads the image and returns structured JSON directly |
| **Strengths** | Fast (~1s/page), fully offline, no GPU needed | Understands layout/context, more resilient to messy scans |
| **Weaknesses** | Sensitive to OCR misreads (e.g. "Total" → "Tolat") | Slower, needs a GPU |
| **Cost** | Free | Free (open-source model, no paid API key) |
| **Measured speed** (900x400 sample image, Colab T4 GPU) | ~1s | ~7-8s |

## Pipeline

```
Document (PDF/image)
        |
        v
Preprocessing (src/preprocessing/preprocess.py)
  - PDF to images (pdf2image)
  - Deskew (principal-axis angle correction)
  - Binarize (Otsu adaptive thresholding)
        |
   +----+-----+
   v          v
Tesseract    Qwen2-VL-2B
OCR          (Vision-LLM)
   |          |
Regex field   Direct JSON
extraction    field extraction
   |          |
   +----+-----+
        v
Unified schema + field-by-field
comparison report
        |
        v
FastAPI (/extract) + Streamlit dashboard
```

## Key Features

- Dual-pipeline extraction with a field-level agreement score between them
- Custom image preprocessing: deskewing via principal-axis angle
  estimation, Otsu adaptive binarization (implemented from scratch with
  NumPy, no OpenCV dependency)
- Regex-based structured extraction (dates, amounts, emails, phones,
  document IDs) with context-aware amount detection - a plain digit
  sequence is only treated as a price if it's tied to a currency symbol
  or a Total/Subtotal/Amount label, so phone numbers and invoice IDs
  don't get misread as prices
- Vision-LLM pipeline using Qwen2-VL-2B-Instruct - fully open-source,
  runs on a free Colab GPU, no paid API calls
- REST API (FastAPI) with file upload
- Streamlit dashboard showing both pipelines' output side-by-side plus
  an agreement chart and a speed comparison
- Unit tests covering extraction, comparison, and preprocessing logic
- Verified, runnable end-to-end demo notebook (notebooks/demo_run.ipynb)

## Project Structure

```
src/preprocessing/   Deskew + binarize + PDF-to-image conversion
src/ocr/              Tesseract wrapper + Qwen2-VL wrapper
src/extraction/       Regex field extraction + pipeline comparison
src/pipeline.py       Orchestrates both pipelines end-to-end
api/                  FastAPI app (/extract endpoint)
dashboard/            Streamlit UI
notebooks/            Verified Colab notebook (full run + tests)
tests/                Unit tests
```

## API

POST /extract - upload a PDF or image (.pdf, .png, .jpg, .jpeg, .tiff,
.bmp). Returns a JSON object with three keys: `tesseract` (fields,
processing time, raw text), `qwen2_vl` (fields, processing time,
detected document type), and `comparison` (field agreement per field
plus an overall agreement score).

GET /health - health check.

## Sample Result

On a clean, high-resolution test invoice, both pipelines extracted
matching fields: 100% agreement on dates, amounts, emails, phones, and
document IDs. Tesseract completed in about 1.3 seconds; Qwen2-VL took
about 7.9 seconds on a Colab T4 GPU.

On a lower-resolution image (small bitmap font), Tesseract's raw text
had OCR misreads ("Total" read as "Tolat", $95.50 read as $9550), which
lowered field agreement - this is expected and is exactly the kind of
failure mode the dual-pipeline comparison is designed to surface:
Qwen2-VL, reading the image directly rather than through an intermediate
OCR text layer, stayed accurate where Tesseract degraded.

## Design Notes / Bugs Found & Fixed

- **Binarization was destroying thin text.** The first version used a
  median filter plus a fixed threshold, which erased thin character
  strokes and badly hurt OCR accuracy. Replaced it with Otsu's method
  (an adaptive threshold that maximizes between-class variance),
  verified against a real sample before and after.
- **The amount regex was too greedy.** It initially matched any digit
  sequence, pulling numbers out of phone numbers and invoice IDs. Fixed
  by requiring a currency symbol or a Total/Subtotal/Amount label
  nearby before treating a number as a monetary amount.
- **4+ digit amounts without a comma were truncated** (e.g. $9550 read
  as 955). Fixed the regex to also accept bare 4+ digit sequences next
  to a currency marker.
- **The Vision-LLM prompt's example value leaked into output.** An early
  prompt used a literal placeholder amount as a JSON example, and the
  model sometimes copied that placeholder instead of the real number.
  Rewrote the prompt to describe the field instead of showing a
  fill-in-the-blank example.

## Testing

Run with: `pytest tests/ -v`

9 tests covering regex field extraction, pipeline-comparison logic, and
the preprocessing pipeline (deskew + binarize) - all passing.

## How to Run (Google Colab)

See `notebooks/demo_run.ipynb` for the verified, runnable end-to-end
demo: clones the repo, installs dependencies (Tesseract, Poppler,
Python packages), generates a sample invoice image, runs both the
Tesseract and Qwen2-VL pipelines on it, prints the field-by-field
comparison report, and runs the full test suite.

Before running the Qwen2-VL step, set the Colab runtime to a GPU via
Runtime -> Change runtime type -> T4 GPU.

## How to Run (API + Dashboard, locally)

Install dependencies with `pip install -r requirements.txt`, then run
`uvicorn api.main:app --reload` in one terminal and
`streamlit run dashboard/app.py` in another.

## Future Improvements

- Add table-structure extraction (rows/columns) for invoices
- Fine-tune Qwen2-VL on a labeled document dataset for higher agreement
- Add a confidence-weighted merge strategy that picks the better field
  from each pipeline instead of just reporting raw agreement
- Add support for handwritten documents

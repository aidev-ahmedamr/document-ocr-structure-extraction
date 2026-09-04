"""
Vision-LLM OCR pipeline using Qwen2-VL-2B-Instruct (fully open-source,
runs on a free Colab GPU - no paid API calls).

Unlike the Tesseract pipeline (OCR -> regex), this pipeline asks the
model to read the document image and directly return structured JSON,
so it can use layout/context to disambiguate fields a regex can't.
"""

import json
import re
import time

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

_model = None
_processor = None

PROMPT = """You are a document-understanding assistant. Look at this
document image and extract information as a single JSON object with
exactly these keys:

- document_type: one of "invoice", "receipt", "id_card", "form", "contract", "other"
- dates: a list of every date string visible on the document
- amounts: a list of every monetary amount visible on the document, as numbers (e.g. 95.5) - read the exact value shown, never invent or default to 0
- emails: a list of every email address visible
- phones: a list of every phone number visible
- document_ids: a list of every invoice/reference/ID number visible
- summary: one sentence describing what this document is

Use an empty list [] for any field with nothing visible on the document - never guess or leave placeholder values.
Return ONLY the JSON object. No markdown code fences, no extra text before or after it.
"""
{
  "document_type": "invoice | receipt | id_card | form | contract | other",
  "dates": ["..."],
  "amounts": [0.0],
  "emails": ["..."],
  "phones": ["..."],
  "document_ids": ["..."],
  "summary": "one sentence describing what this document is"
}
"""


def load_model():
    """Load the model once and cache it (loading takes ~10-20s on a
    Colab GPU, so we don't want to repeat it per document)."""

    global _model, _processor

    if _model is None:
        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        _processor = AutoProcessor.from_pretrained(MODEL_ID)

    return _model, _processor


def _extract_json_block(raw_text):
    """The model sometimes wraps JSON in prose or markdown fences -
    pull out the first {...} block and parse it defensively."""

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def run_vision_llm(image):
    """Run the vision-LLM pipeline on a single preprocessed PIL image."""

    model, processor = load_model()

    start = time.time()

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }]

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_prompt], images=[image], return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512)

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    raw_output = processor.batch_decode(
        generated, skip_special_tokens=True
    )[0]

    parsed = _extract_json_block(raw_output)
    elapsed = time.time() - start

    fields = {
        "dates": parsed.get("dates", []),
        "amounts": parsed.get("amounts", []),
        "emails": parsed.get("emails", []),
        "phones": parsed.get("phones", []),
        "document_ids": parsed.get("document_ids", []),
    }

    return {
        "raw_text": raw_output,
        "fields": fields,
        "document_type": parsed.get("document_type", "unknown"),
        "summary": parsed.get("summary", ""),
        "processing_time_seconds": elapsed,
    }


def run_vision_llm_multi_page(images):
    """Run the vision-LLM on every page and merge the extracted fields."""

    results = [run_vision_llm(img) for img in images]

    merged_fields = {"dates": [], "amounts": [], "emails": [], "phones": [], "document_ids": []}
    for r in results:
        for key in merged_fields:
            merged_fields[key].extend(r["fields"][key])

    return {
        "raw_text": "\n\n".join(r["raw_text"] for r in results),
        "fields": merged_fields,
        "document_type": results[0]["document_type"] if results else "unknown",
        "processing_time_seconds": sum(r["processing_time_seconds"] for r in results),
    }

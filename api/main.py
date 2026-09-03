import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException

from src.pipeline import process_document


app = FastAPI(
    title="Document OCR & Structure Extraction Pipeline",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"message": "Document OCR & Structure Extraction Pipeline is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    """
    Upload a PDF or image. Runs both the Tesseract and Qwen2-VL
    pipelines and returns their extracted fields plus a comparison.
    """

    allowed_suffixes = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(allowed_suffixes)}",
        )

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        result = process_document(tmp_path)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        Path(tmp_path).unlink(missing_ok=True)

import streamlit as st
import requests
import pandas as pd


API_URL = "http://localhost:8000/extract"

st.set_page_config(page_title="Document OCR & Structure Extraction", layout="wide")
st.title("📄 Multimodal Document OCR & Structure Extraction")
st.caption("Compares a traditional OCR pipeline (Tesseract) against a Vision-LLM pipeline (Qwen2-VL) on the same document.")

uploaded_file = st.file_uploader(
    "Upload a document (PDF, PNG, JPG)",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
)

if uploaded_file is not None:
    if uploaded_file.type.startswith("image"):
        st.image(uploaded_file, caption="Uploaded document", width=350)

    if st.button("🔍 Extract & Compare"):
        with st.spinner("Running both pipelines..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                response = requests.post(API_URL, files=files, timeout=120)
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the API: {e}")
                st.stop()

        tesseract = result["tesseract"]
        qwen = result["qwen2_vl"]
        comparison = result["comparison"]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🅰️ Tesseract (Traditional OCR)")
            st.metric("Processing time", f"{tesseract['processing_time_seconds']:.2f}s")
            st.json(tesseract["fields"])
            with st.expander("Raw extracted text"):
                st.text(tesseract["raw_text"])

        with col2:
            st.subheader("🅱️ Qwen2-VL (Vision-LLM)")
            st.metric("Processing time", f"{qwen['processing_time_seconds']:.2f}s")
            st.caption(f"Detected type: **{qwen['document_type']}**")
            st.json(qwen["fields"])
            with st.expander("Raw model output"):
                st.text(qwen["raw_text"])

        st.subheader("📊 Comparison")
        st.metric("Overall field agreement", f"{comparison['overall_agreement'] * 100:.0f}%")

        agreement_df = pd.DataFrame(
            list(comparison["field_agreement"].items()),
            columns=["Field", "Agreement"],
        )
        st.bar_chart(agreement_df.set_index("Field"))

        speed_df = pd.DataFrame(
            list(comparison["speed"].items()),
            columns=["Pipeline", "Seconds"],
        )
        st.bar_chart(speed_df.set_index("Pipeline"))

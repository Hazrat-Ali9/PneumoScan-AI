import os
import tempfile

import cv2
import numpy as np
import streamlit as st

from project.components.inferance import Prediction_Pipeline


MODEL_PATH = "artifacts/training/best_chest_xray_model.keras"


def build_overlay(original_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create a red overlay image from a binary segmentation mask."""
    mask_binary = (mask > 0).astype(np.uint8)
    red_layer = np.zeros_like(original_img)
    red_layer[:, :, 0] = mask_binary * 255
    return cv2.addWeighted(original_img, 0.85, red_layer, 0.35, 0)


@st.cache_resource
def load_pipeline(model_path: str) -> Prediction_Pipeline:
    return Prediction_Pipeline(model_path=model_path)


def main() -> None:
    st.set_page_config(
        page_title="PneumoScan AI",
        page_icon="🫁",
        layout="wide",
    )

    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at 10% 10%, #dff4ff 0%, transparent 35%),
                    radial-gradient(circle at 90% 5%, #e8ffe8 0%, transparent 30%),
                    linear-gradient(180deg, #f8fbff 0%, #f1f6f7 100%);
            }
            .main-title {
                font-size: 2.1rem;
                font-weight: 800;
                color: #0d2f3b;
                margin-bottom: 0.1rem;
            }
            .sub-title {
                color: #35515e;
                margin-bottom: 1.2rem;
            }
            .metric-card {
                background: #ffffff;
                border: 1px solid #dbe7ef;
                border-radius: 14px;
                padding: 0.9rem 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">PneumoScan AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Upload a chest X-ray and run pneumothorax segmentation + detection.</div>',
        unsafe_allow_html=True,
    )

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at `{MODEL_PATH}`")
        st.stop()

    pipeline = load_pipeline(MODEL_PATH)

    left, right = st.columns([1.2, 2.8], gap="large")

    with left:
        st.subheader("Input")
        threshold = st.slider(
            "Detection threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.20,
            step=0.05,
            help="Lower values find more regions; higher values are stricter.",
        )
        uploaded_file = st.file_uploader(
            "Upload chest X-ray",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
        )
        run_inference = st.button("Run Analysis", type="primary", use_container_width=True)

    with right:
        if uploaded_file is None:
            st.info("Upload an image and click `Run Analysis`.")
            return

        if not run_inference:
            st.image(uploaded_file, caption="Preview", use_container_width=True)
            return

        file_suffix = os.path.splitext(uploaded_file.name)[1] or ".png"
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_path = tmp.name

            with st.spinner("Running model inference..."):
                original_img, mask, detection_img = pipeline.predict(temp_path, threshold=threshold)

            mask_binary = (mask > 0).astype(np.uint8)
            affected_area_pct = 100.0 * mask_binary.sum() / mask_binary.size
            overlay_img = build_overlay(original_img, mask_binary)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Estimated affected area", f"{affected_area_pct:.2f}%")
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                detected = "Yes" if affected_area_pct > 0 else "No"
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Suspicious region detected", detected)
                st.markdown("</div>", unsafe_allow_html=True)

            row1_col1, row1_col2 = st.columns(2, gap="medium")
            with row1_col1:
                st.image(original_img, caption="Original X-ray", use_container_width=True)
            with row1_col2:
                st.image((mask_binary * 255), caption="Segmentation Mask", use_container_width=True)

            row2_col1, row2_col2 = st.columns(2, gap="medium")
            with row2_col1:
                st.image(detection_img, caption="Detection (Bounding Box)", use_container_width=True)
            with row2_col2:
                st.image(overlay_img, caption="Highlighted Overlay", use_container_width=True)

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    main()


## streamlit run app.py

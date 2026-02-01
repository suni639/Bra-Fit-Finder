"""
Bra Fit Finder MVP
A Streamlit application for nursing bra size recommendation using MediaPipe pose detection.
"""

from pathlib import Path

import streamlit as st
from PIL import Image as PILImage
import numpy as np

# MediaPipe Tasks API (Pose Landmarker)
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

from logic import compute_bra_fit


# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bra Fit Finder",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# MediaPipe Initialisation (Tasks API - Pose Landmarker)
# -----------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "models" / "pose_landmarker.task"


@st.cache_resource
def get_pose_detector():
    """Initialise and cache the MediaPipe Pose Landmarker for efficiency."""
    return vision.PoseLandmarker.create_from_model_path(str(MODEL_PATH))


# -----------------------------------------------------------------------------
# UI Layout
# -----------------------------------------------------------------------------
def main():
    # Header
    st.title("Bra Fit Finder")
    st.markdown(
        "*Find your perfect nursing bra size with confidence. "
        "Upload your photos and we'll guide you.*"
    )
    st.divider()

    # Sidebar for additional context (optional, collapsed by default)
    with st.sidebar:
        st.caption("About")
        st.info(
            "This tool uses AI-powered pose detection to estimate your bra size "
            "based on front and side photos. Your privacy matters—images are "
            "processed locally and not stored."
        )

    # Input Section
    st.subheader("Your Details")

    col1, col2 = st.columns([1, 1])

    with col1:
        weeks_input = st.number_input(
            label="Weeks Postpartum / Pregnant",
            min_value=0,
            max_value=52,
            value=12,
            step=1,
            help="Enter 0 if pregnant, or weeks since birth if postpartum.",
        )

    with col2:
        st.caption("")
        st.caption("")  # Spacing alignment
        st.markdown(
            "Early postpartum (< 6 weeks)? We'll account for "
            "milk regulation and engorgement in your estimate."
        )

    st.divider()
    st.subheader("Upload Photos")

    front_col, side_col = st.columns(2)

    with front_col:
        front_image = st.file_uploader(
            "Front Image",
            type=["jpg", "jpeg", "png"],
            help="A clear front-facing photo in good lighting.",
        )
        if front_image:
            st.image(front_image, use_container_width=True)

    with side_col:
        side_image = st.file_uploader(
            "Side Image",
            type=["jpg", "jpeg", "png"],
            help="A clear side-profile photo in good lighting.",
        )
        if side_image:
            st.image(side_image, use_container_width=True)

    st.divider()

    # Action Button
    if st.button("Find My Size", type="primary", use_container_width=True):
        if not front_image or not side_image:
            st.warning("Please upload both front and side images to continue.")
        else:
            if not MODEL_PATH.exists():
                st.error(
                    "Pose model not found. Run `python download_model.py` to download it, "
                    "or place pose_landmarker.task in the models/ directory."
                )
                st.stop()

            with st.spinner("Analysing your photos..."):
                landmarker = get_pose_detector()

                # Convert uploads to RGB numpy arrays for MediaPipe
                front_img = PILImage.open(front_image).convert("RGB")
                side_img = PILImage.open(side_image).convert("RGB")
                front_arr = np.array(front_img)
                side_arr = np.array(side_img)

                # Create MediaPipe Image objects and run detection
                front_mp_image = Image(
                    image_format=ImageFormat.SRGB,
                    data=front_arr,
                )
                side_mp_image = Image(
                    image_format=ImageFormat.SRGB,
                    data=side_arr,
                )
                front_results = landmarker.detect(front_mp_image)
                side_results = landmarker.detect(side_mp_image)

                # Run full pipeline: extract landmarks, volume estimate, growth curve, bra size
                result = compute_bra_fit(
                    front_results,
                    side_results,
                    weeks_postpartum=weeks_input,
                )

                if result.landmarks_detected:
                    st.success("Landmarks detected successfully.")
                    st.info(f"Recommended size: **{result.recommended_size}**")
                    with st.expander("Details"):
                        st.caption(
                            f"Volume estimate: {result.volume_estimate:.1f} → "
                            f"Adjusted: {result.volume_adjusted:.1f}"
                        )
                else:
                    st.error(
                        "We couldn't detect a clear pose in one or both images. "
                        "Please try photos with better lighting and a clear view."
                    )

    # Footer
    st.divider()
    st.caption("Bra Fit Finder MVP · For support, please contact your retailer.")


if __name__ == "__main__":
    main()

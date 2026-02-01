"""
Download the MediaPipe Pose Landmarker model for Bra Fit Finder.
Run once before first use: python download_model.py
"""

import urllib.request
from pathlib import Path

# Lite model (smaller, faster) - suitable for MVP
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/"
    "pose_landmarker_lite.task"
)
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "pose_landmarker.task"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists():
        print(f"Model already exists at {MODEL_PATH}")
        return

    print(f"Downloading Pose Landmarker model to {MODEL_PATH}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")
        print(
            "Please download the model manually from:\n"
            "https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/index#models"
        )
        raise


if __name__ == "__main__":
    main()

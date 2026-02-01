<div align="center">

# Bra Fit Finder (MVP)

### Smart sizing for maternity and postpartum.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-red)
![MediaPipe](https://img.shields.io/badge/Computer%20Vision-MediaPipe-green)
![Privacy](https://img.shields.io/badge/Privacy-Local%20Processing-lightgrey)

<p align="center">
  The <b>Bra Fit Finder</b> is a prototype designed to solve the "sizing gamble" in maternity wear. <br>
  By combining computer vision with biological data, we move from reactive returns to predictive fitting.
</p>

</div>

---

## 📖 How It Works
You don't need to be a data scientist to understand the logic. Here is the user journey:

1.  **The Context** 🤰
    The user inputs how many weeks postpartum they are. This is crucial for our biological adjustments.
2.  **The Scan** 📸
    The user uploads a Front and Side photo. The app creates a "digital skeleton" of their torso.
3.  **The Math** 📐
    It calculates the total volume of breast tissue using geometric formulas, rather than just measuring circumference like a tape measure.
4.  **The Match** ✅
    It maps that volume to a specific cup size (e.g., 34D) and displays the result instantly.

---

## 🔬 The Science: Geometry & Biology

This engine combines **Computer Vision** with **Biological Data**.

### 1. The Geometry (The "How")
Standard bra fitting uses a 2D tape measure to guess 3D volume. We calculate 3D volume directly.

* **The Model:** We model the breast as a **Hemi-Ellipsoid** (half of a stretched sphere).
* **The Inputs:** From the *Front Photo*, we extract width relative to shoulders. From the *Side Photo*, we extract projection.
* **The Calculation:** By combining `Width × Height × Projection`, we get a cubic centimetre (cc) volume estimate.

### 2. The Maternity Multiplier (The "Why")
Most sizing tools fail because they assume a static body. A postpartum body is dynamic.

> **The Biological Reality:** In the first 6 weeks postpartum (Lactogenesis II), tissue swells due to milk regulation and lymphatic fluid.

* **Our Solution:** If the user is **< 6 weeks postpartum**, our algorithm applies a **1.15x multiplier** (15% increase) to the volume.
* **The Benefit:** We size the user for the bra they need *next week*, not just the size they are this second.

---

## 🔒 Privacy & Security

We built this with a **Privacy First** architecture. Security is the foundation of this product.

| Feature | Description |
| :--- | :--- |
| **Local Processing** | All image analysis happens directly on the user's laptop (client-side). |
| **No Cloud Storage** | Photos are **never** sent to a remote server or third-party API. |
| **Ephemeral Data** | Images exist in RAM only for the seconds required to calculate landmarks. |

---

## 🤖 Technical Deep Dive

**The Model: Google MediaPipe Pose Landmarker (Lite)**
We utilise the `pose_landmarker_lite.task` model—a lightweight convolutional neural network optimised for edge devices.

* **Why this model?** Low latency, no internet required.
* **The Topology:** Detects 33 3D landmarks. We isolate **Shoulders** (11/12) and **Hips** (23/24).
* **The Interpolation:** Since the model does not detect "underbust" (for privacy), we calculate it mathematically using anthropometric ratios (approx. 53% down the torso length).

---

## 🚀 Getting Started

Follow these steps to run the prototype on your own computer.

### 1. Prerequisites
Ensure you have **Python 3.9** or higher installed.

### 2. Create a "Clean Workspace"
It is best practice to run this in a virtual environment.

**On macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Installation
Install the required software tools (Streamlit, MediaPipe, etc.):
```bash
pip install -r requirements.txt
```

### 4. Download the AI Model
**Important:** You must run this once before starting the app. This script downloads the official Google MediaPipe task file and places it in the `models/` directory.
```bash
python download_model.py
```

### 5. Run the App
Launch the interface:
```bash
streamlit run app.py
```
A new tab should automatically open in your web browser, displaying the Bra Fit Finder.

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `app.py` | **The UI:** Handles the UI, file uploads, and user input. |
| `logic.py` | **The Brain:** Contains the geometry calculations (Hemi-Ellipsoid) and biological logic. |
| `download_model.py` | **AI Utility:** Fetches the AI model components. |
| `models/` | **Storage:** Directory storing the `pose_landmarker.task`. |
| `requirements.txt` | **Dependencies:** List of required libraries. |

---

*For support or to report a sizing discrepancy, please contact the product team.*

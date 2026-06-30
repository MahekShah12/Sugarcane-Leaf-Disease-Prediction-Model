"""
Sugarcane Leaf Disease Detection using Deep Learning
-----------------------------------------------------
A Streamlit web application that replaces the original Tkinter desktop GUI.
All machine learning logic (preprocessing, prediction, class labels,
disease descriptions, confidence calculation, and probability chart
generation) has been preserved EXACTLY as in the original Tkinter
application. Only the user interface layer has been rebuilt using
Streamlit.
"""
# IMPORTS
import io
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image


# ----------------------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------------------
# st.set_page_config() must be the first Streamlit command executed.
st.set_page_config(
    page_title="Sugarcane Leaf Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------
# CUSTOM CSS — GREEN AGRICULTURAL THEME
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background-color: #f4faf5;
    }

    /* Big page title styling */
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1b5e20;
        text-align: center;
        padding-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1b5e20;
        border-left: 6px solid #2E8B57;
        padding-left: 10px;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    /* Card-like container */
    .info-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e0e0e0;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: #555555;
        font-size: 0.9rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #cccccc;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #e8f5e9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# CONSTANTS — MODEL PATH, CLASS LABELS, DISEASE DESCRIPTIONS
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sugarcane_leaf_disease_model_v2_1.keras"
CLASS_LABELS = ['Bacterial Blight', 'Healthy', 'Mosaic', 'Red Rot', 'Rust', 'Yellow Leaf']

CLASS_DESCRIPTIONS = {
    'Bacterial Blight': "Bacterial blight causes water-soaked lesions that turn yellowish with age. "
                         "Treatment involves copper-based bactericides and removing infected plants.",
    'Healthy': "This leaf appears healthy with no visible signs of disease.",
    'Mosaic': "Mosaic disease shows yellow-green mottling patterns. Control by using resistant "
              "varieties and removing infected plants.",
    'Red Rot': "Red rot causes reddish discoloration inside stems. Treat with fungicides and use "
               "disease-free planting material.",
    'Rust': "Rust appears as orange-brown pustules. Control with fungicides and resistant varieties.",
    'Yellow Leaf': "Yellow leaf syndrome causes midrib and leaf yellowing. Manage with virus-free "
                   "seedcane and resistant varieties."
}

INPUT_SIZE = (224, 224) 


# ----------------------------------------------------------------------
# MODEL LOADING 
# ----------------------------------------------------------------------
@st.cache_resource
def load_disease_model(model_path: Path):
    """
    Load the trained Keras model from disk.

    Cached with st.cache_resource so the (potentially large) model is
    loaded into memory only once per app session, instead of being
    reloaded on every Streamlit rerun.
    """
    if not model_path.exists():
        return None
    try:
        model = load_model(model_path)
        return model
    except Exception as exc:
        st.session_state["model_load_error"] = str(exc)
        return None


# ----------------------------------------------------------------------
# PREDICTION LOGIC
# ----------------------------------------------------------------------
def predict_disease(model, img_path_or_buffer):
    """
    Run the model on a given image and return prediction results.

    Preprocessing steps (UNCHANGED from the original app):
      1. Load image resized to (224, 224)
      2. Convert to array
      3. Expand dims to create a batch of size 1
      4. Normalize pixel values to [0, 1] by dividing by 255.0

    Returns a dict with:
      - predicted_class: str
      - confidence: float (percentage)
      - probabilities: dict[str, float] (percentage per class)
    """
    img = keras_image.load_img(img_path_or_buffer, target_size=INPUT_SIZE)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  
    
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_class_idx]) * 100

    return {
        'predicted_class': CLASS_LABELS[predicted_class_idx],
        'confidence': confidence,
        'probabilities': {
            CLASS_LABELS[i]: float(prediction[0][i]) * 100 for i in range(len(CLASS_LABELS))
        }
    }


# ----------------------------------------------------------------------
# PROBABILITY BAR CHART
# ----------------------------------------------------------------------
def create_bar_chart(prediction_results):
    """
    Build a horizontal bar chart of class probabilities, sorted in
    descending order, with the predicted class highlighted in green.
    This mirrors the original Tkinter `create_bar_chart` method exactly.
    """
    probs = prediction_results['probabilities']
    labels = list(probs.keys())
    values = list(probs.values())

    sorted_indices = np.argsort(values)[::-1]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]

    colors = ['#2E8B57' if label == prediction_results['predicted_class'] else '#CCCCCC'
              for label in sorted_labels]

    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=100)
    ax.barh(sorted_labels, sorted_values, color=colors)
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Disease Probability')
    ax.set_xlim(0, 100)
    ax.invert_yaxis() 

    for i, v in enumerate(sorted_values):
        ax.text(v + 1, i, f"{v:.1f}%", va='center')

    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------
# SIDEBAR — PROJECT INFORMATION
# ----------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🌾 Project Information")
        st.markdown("---")

        st.markdown("**🧠 Model Name**")
        st.write("Sugarcane Leaf Disease Model v2.1")

        st.markdown("**🏷️ Number of Classes**")
        st.write(f"{len(CLASS_LABELS)} classes")
        st.caption(", ".join(CLASS_LABELS))

        st.markdown("---")
        st.info(
            "Upload a sugarcane leaf image on the main page to automatically "
            "detect the disease and view detailed information."
        )


# ----------------------------------------------------------------------
# MAIN APPLICATION
# ----------------------------------------------------------------------
def main():
    render_sidebar()
    st.markdown(
        '<div class="main-title">🌱 Sugarcane Leaf Disease Detection</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Upload a sugarcane leaf image to automatically detect '
        'disease type, confidence score, and treatment guidance.</div>',
        unsafe_allow_html=True,
    )
    model = load_disease_model(MODEL_PATH)

    if model is None:
        if "model_load_error" in st.session_state:
            st.error(f"Failed to load model: {st.session_state['model_load_error']}")
        else:
            st.error(
                f"Model file not found at `{MODEL_PATH}`. "
                "Please place the `.keras` model file in the project folder."
            )
        st.stop()

    st.success("Model loaded successfully and ready for predictions.")

    st.markdown("---")

    st.markdown('<div class="section-header">Upload Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a sugarcane leaf image (JPG, JPEG, PNG, BMP)",
        type=["jpg", "jpeg", "png", "bmp"],
    )

    st.markdown("---")
    left_col, right_col = st.columns(2, gap="large")

    prediction_results = None

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

        with left_col:
            st.markdown('<div class="section-header"> Uploaded Image</div>', unsafe_allow_html=True)
            with st.container():
                st.image(pil_image, caption="Uploaded Leaf Image", use_container_width=True)

        with st.spinner("🔍 Analyzing leaf image..."):
            prediction_results = predict_disease(model, io.BytesIO(file_bytes))
        with right_col:
            st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)
            predicted_class = prediction_results['predicted_class']
            confidence = prediction_results['confidence']
            
            metric_col1, metric_col2 = st.columns(2, gap="small")
            metric_col1.metric(label="Predicted Disease", value=predicted_class)
            metric_col2.metric(label="Confidence", value=f"{confidence:.1f}%")

            st.progress(min(int(confidence), 100) / 100)

            if predicted_class == "Healthy":
                st.success(f"The leaf appears **Healthy** ({confidence:.2f}% confidence).")
            elif confidence < 60:
                st.warning(
                    f"Detected **{predicted_class}** with low confidence ({confidence:.2f}%). "
                    "Consider uploading a clearer image."
                )
            else:
                st.error(f"Disease Detected: **{predicted_class}** ({confidence:.2f}% confidence)")

            # Disease description
            st.markdown("**Disease Description**")
            description = CLASS_DESCRIPTIONS.get(predicted_class, "No information available.")
            st.markdown(f'<div class="info-card">{description}</div>', unsafe_allow_html=True)

    else:
        with left_col:
            st.markdown('<div class="section-header">Uploaded Image</div>', unsafe_allow_html=True)
            st.info(" Upload an image to see it displayed here.")

        with right_col:
            st.markdown('<div class="section-header">🔬 Prediction Results</div>', unsafe_allow_html=True)
            st.info(" Prediction results will appear here after you upload an image.")
            
    st.markdown("---")
    st.markdown('<div class="section-header">Disease Probability Graph</div>', unsafe_allow_html=True)

    if prediction_results is not None:
        fig = create_bar_chart(prediction_results)
        st.pyplot(fig)
    else:
        st.info("The probability chart will be generated automatically once an image is analyzed.")

    st.markdown("---")
    st.markdown('<div class="section-header"> Detailed Disease Information</div>', unsafe_allow_html=True)

    info_cols = st.columns(3)
    for idx, label in enumerate(CLASS_LABELS):
        with info_cols[idx % 3]:
            highlight = (
                prediction_results is not None
                and prediction_results['predicted_class'] == label
            )
            border_color = "#2E8B57" if highlight else "#e0e0e0"
            st.markdown(
                f"""
                <div style="background-color:#ffffff; border:2px solid {border_color};
                            border-radius:10px; padding:0.9rem; margin-bottom:0.8rem;">
                    <b>🌿 {label}</b><br>
                    <span style="font-size:0.88rem; color:#444444;">
                        {CLASS_DESCRIPTIONS[label]}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    with st.expander("❓ Help — How to use this application"):
        st.markdown(
            """
            ### Sugarcane Leaf Disease Detector — Help

            This application helps identify diseases in sugarcane leaves using
            machine learning.

            **How to use:**
            1. Use the **Upload Image** section to select a leaf image from your computer.
            2. The image is displayed automatically and the model runs a prediction right away.
            3. View the **Prediction**, **Confidence**, and **Disease Description** in the right column.
            4. Check the **Disease Probability Graph** for a full breakdown across all classes.
            5. Scroll down to **Detailed Disease Information** for treatment guidance on every class.
            6. To analyze another image, simply upload a new file — the results update automatically.

            **Detectable Diseases:**
            - Bacterial Blight
            - Healthy (no disease)
            - Mosaic
            - Red Rot
            - Rust
            - Yellow Leaf

            **Tip:** For best results, use clear, well-lit images of individual leaves.
            """
        )

if __name__ == "__main__":
    main()
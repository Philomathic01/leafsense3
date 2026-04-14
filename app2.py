"""
🥔 Potato Disease Detection App
Streamlit frontend — CNN Feature Extractor + SVM classifier
UI kept close to the older app layout/style.
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import io
import json
import time
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from PIL import Image

# ── Try importing TensorFlow / Keras ─────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from disease_info import get_disease_info, DISEASE_INFO


# ═══════════════════════════════════════════════════════════════════════════════
#  Page Config
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Potato Disease Detector",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  Custom CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3, h4, h5 {
    font-family: 'Syne', sans-serif;
}

.main { background: #0d1f0e; color: #e8f5e9; }
.stApp { background: linear-gradient(135deg, #0d1f0e 0%, #1a3320 50%, #0d1f0e 100%); }

.hero-banner {
    background: linear-gradient(135deg, #1b5e20, #2e7d32, #388e3c);
    border-radius: 18px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    border: 1px solid #4caf50;
    box-shadow: 0 8px 40px rgba(76,175,80,0.25);
}
.hero-banner h1 { color: #fff; font-size: 2.6rem; margin: 0; }
.hero-banner p  { color: #a5d6a7; margin: 0.5rem 0 0; font-size: 1.05rem; }

.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(76,175,80,0.3);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
}
.metric-card h3 {
    color: #81c784;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 0 0 0.3rem;
}
.metric-card .value {
    color: #fff;
    font-size: 2rem;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
}

.result-healthy  { background: rgba(39,174,96,0.15);  border: 2px solid #27ae60; }
.result-early    { background: rgba(243,156,18,0.15); border: 2px solid #f39c12; }
.result-late     { background: rgba(231,76,60,0.15);  border: 2px solid #e74c3c; }

.result-box {
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem 0;
}
.result-box h2 {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    margin: 0 0 0.5rem;
    color: #fff;
}
.result-box p {
    color: #cfd8dc;
    margin: 0.15rem 0;
    font-size: 1rem;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    border: 2px dashed rgba(76,175,80,0.5);
    border-radius: 14px;
    padding: 1rem;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    color: #a5d6a7;
    font-family: 'Syne', sans-serif;
}
.stTabs [aria-selected="true"] {
    background: #2e7d32 !important;
    color: #fff !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1a0b 0%, #1a2e1b 100%);
    border-right: 1px solid rgba(76,175,80,0.2);
}
section[data-testid="stSidebar"] * { color: #c8e6c9 !important; }

div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(76,175,80,0.25);
    border-radius: 12px;
    padding: 1rem;
}

.stProgress > div > div { background-color: #4caf50; }

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #81c784;
    border-left: 4px solid #4caf50;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def hex_to_rgba(hex_color, alpha=0.15):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def clean_display_name(name: str) -> str:
    return name.replace("Potato___", "").replace("Potato__", "").replace("_", " ").title()


def build_feature_extractor_architecture(input_shape=(224, 224, 3)):
    """
    Same architecture used in training for the CNN feature extractor.
    This is mainly for summary display fallback.
    """
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.30)(x)

    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.30)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.50)(x)

    features = layers.Dense(256, activation="relu", name="feature_vector")(x)

    return keras.Model(inputs, features, name="PotatoCNN_FeatureExtractor")


@st.cache_resource(show_spinner=False)
def load_hybrid_assets(feature_model_path, svm_path, scaler_path, metadata_path):
    if not TF_AVAILABLE:
        return None

    try:
        feature_extractor = keras.models.load_model(feature_model_path, compile=False)
        svm_model = joblib.load(svm_path)
        scaler = joblib.load(scaler_path)

        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

        return {
            "feature_extractor": feature_extractor,
            "svm_model": svm_model,
            "scaler": scaler,
            "metadata": metadata,
        }
    except Exception as e:
        st.error(f"Could not load hybrid assets: {e}")
        return None


def preprocess_image(img: Image.Image, img_size=(224, 224)):
    img = img.convert("RGB").resize(img_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0), img


def predict_hybrid(feature_extractor, scaler, svm_model, img_array):
    features = feature_extractor.predict(img_array, verbose=0)
    features_scaled = scaler.transform(features)
    probs = svm_model.predict_proba(features_scaled)[0]
    pred_idx = int(np.argmax(probs))
    return probs, pred_idx, features[0]


def get_model_summary_text(model_obj):
    stream = io.StringIO()

    def capture(text, line_break=False):
        stream.write(str(text))
        if line_break:
            stream.write("\n")

    model_obj.summary(print_fn=capture)
    return stream.getvalue()


def resolve_class_names(metadata):
    class_names = metadata.get("class_names", [])
    if class_names:
        return class_names

    idx_to_class = metadata.get("idx_to_class", {})
    if idx_to_class:
        ordered = []
        for i in range(len(idx_to_class)):
            ordered.append(idx_to_class[str(i)])
        return ordered

    return ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🥔 Potato Disease Detector")
    st.markdown("---")

    st.markdown("### ⚙️ Model Settings")
    feature_model_path = st.text_input(
        "Feature extractor (.keras)",
        value="best_cnn_feature_extractor.keras",
        help="Path to the saved CNN feature extractor model.",
    )
    svm_model_path = st.text_input(
        "SVM model (.joblib)",
        value="svm_on_cnn_features.joblib",
        help="Path to the trained SVM model.",
    )
    scaler_path = st.text_input(
        "Scaler (.joblib)",
        value="feature_scaler.joblib",
        help="Path to the saved feature scaler.",
    )
    metadata_path = st.text_input(
        "Metadata (.json)",
        value="metadata.json",
        help="Path to the metadata JSON generated during training.",
    )

    img_size_option = st.selectbox("Input image size", [224, 128, 256], index=0)
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)

    st.markdown("---")
    st.markdown("### 📋 About")
    st.markdown(
        """
        This app detects **potato leaf diseases** using a **CNN + SVM hybrid model**.

        **Pipeline:**
        - CNN extracts deep visual features
        - StandardScaler normalizes features
        - SVM predicts the final class

        **Classes:**
        - 🍂 Early Blight (*Alternaria solani*)
        - ⚠️ Late Blight (*Phytophthora infestans*)
        - ✅ Healthy
        """
    )

    st.markdown("---")
    if TF_AVAILABLE:
        st.caption(f"TensorFlow: {tf.__version__}")
        st.caption(f"Keras: {keras.__version__}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Hero Banner
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <h1>🥔 Potato Disease Detection System</h1>
    <p>AI-powered CNN + SVM hybrid model — Upload a leaf image to detect Early Blight, Late Blight, or Healthy status</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Load hybrid assets
# ═══════════════════════════════════════════════════════════════════════════════
assets = None
feature_extractor = None
svm_model = None
scaler = None
metadata = {}
model_loaded = False

if TF_AVAILABLE and os.path.exists(feature_model_path) and os.path.exists(svm_model_path) and os.path.exists(scaler_path):
    with st.spinner("Loading CNN feature extractor + SVM assets…"):
        assets = load_hybrid_assets(feature_model_path, svm_model_path, scaler_path, metadata_path)

    if assets is not None:
        feature_extractor = assets["feature_extractor"]
        svm_model = assets["svm_model"]
        scaler = assets["scaler"]
        metadata = assets.get("metadata", {})
        model_loaded = True
        st.success("✅ Hybrid model assets loaded successfully.")
elif not TF_AVAILABLE:
    st.warning("⚠️ TensorFlow/Keras not installed correctly.")
else:
    st.info(
        "📌 Hybrid model files not found. Demo mode is active — sample predictions will be shown."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Class names
# ═══════════════════════════════════════════════════════════════════════════════
CLASS_NAMES = resolve_class_names(metadata)
CLEAN_NAMES = [clean_display_name(name) for name in CLASS_NAMES]

# fallback for disease_info mapping consistency
DEFAULT_CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
if len(CLASS_NAMES) != 3:
    CLASS_NAMES = DEFAULT_CLASS_NAMES
    CLEAN_NAMES = [clean_display_name(name) for name in CLASS_NAMES]


# ═══════════════════════════════════════════════════════════════════════════════
#  Tabs
# ═══════════════════════════════════════════════════════════════════════════════
tab_predict, tab_dashboard, tab_info, tab_model = st.tabs([
    "🔍 Predict Disease",
    "📊 Model Performance",
    "📖 Disease Information",
    "🧠 Model Architecture",
])


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — Predict
# ─────────────────────────────────────────────────────────────────────────────
with tab_predict:
    col_upload, col_result = st.columns([1, 1.5], gap="large")

    with col_upload:
        st.markdown('<div class="section-header">Upload Leaf Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a potato leaf image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Upload a clear image of the potato leaf for analysis.",
        )

        if uploaded_file:
            img_pil = Image.open(uploaded_file)
            st.image(img_pil, caption="Uploaded Image", use_container_width=True)

            st.markdown('<div class="section-header">Image Info</div>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Width",  f"{img_pil.width}px")
            m2.metric("Height", f"{img_pil.height}px")
            m3.metric("Mode",   img_pil.mode)
            st.metric("File size", f"{uploaded_file.size / 1024:.1f} KB")

    with col_result:
        st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)

        if uploaded_file:
            with st.spinner("🔬 Analysing image…"):
                time.sleep(0.4)
                img_array, _ = preprocess_image(img_pil, (img_size_option, img_size_option))

                if model_loaded:
                    probs, pred_idx, feature_vec = predict_hybrid(
                        feature_extractor, scaler, svm_model, img_array
                    )
                else:
                    np.random.seed(int(sum(uploaded_file.getvalue()[:50])) % 10000)
                    probs = np.random.dirichlet(np.ones(3) * 0.5)
                    pred_idx = int(np.argmax(probs))
                    feature_vec = np.zeros(256)
                    st.info("🎭 Demo mode — predictions are illustrative only.")

            confidence = float(probs[pred_idx])
            pred_class = CLASS_NAMES[pred_idx]
            pred_clean = CLEAN_NAMES[pred_idx]
            disease = get_disease_info(pred_class)

            if confidence < confidence_threshold:
                st.warning(
                    f"⚠️ Low confidence ({confidence:.1%}). "
                    "The image may be unclear or not a potato leaf."
                )

            severity = disease["severity"]
            css_class = (
                "result-healthy" if "healthy" in pred_class.lower()
                else "result-late" if "late" in pred_class.lower()
                else "result-early"
            )

            st.markdown(f"""
            <div class="result-box {css_class}">
                <h2>{disease['icon']} {pred_clean}</h2>
                <p>Scientific name: <em>{disease['scientific_name']}</em></p>
                <p>Confidence: <strong style="font-size:1.3rem;">{confidence:.2%}</strong></p>
                <p>Severity: <strong style="color:{disease['severity_color']}">{severity}</strong></p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-header">Prediction Confidence</div>', unsafe_allow_html=True)
            bar_colors = ["#f39c12", "#e74c3c", "#27ae60"]
            fig_bar = go.Figure(go.Bar(
                x=CLEAN_NAMES,
                y=[float(p) for p in probs],
                marker_color=bar_colors,
                marker_line_color="rgba(255,255,255,0.2)",
                marker_line_width=1.5,
                text=[f"{float(p):.2%}" for p in probs],
                textposition="outside",
                textfont=dict(color="white", size=13),
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", family="DM Sans"),
                yaxis=dict(title="Probability", range=[0, 1.1], gridcolor="rgba(255,255,255,0.1)"),
                xaxis=dict(tickfont=dict(size=13)),
                height=320,
                margin=dict(t=20, b=10),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                number={"suffix": "%", "font": {"color": "white", "size": 28}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "white"},
                    "bar": {"color": disease["severity_color"]},
                    "steps": [
                        {"range": [0, 50], "color": "rgba(231,76,60,0.2)"},
                        {"range": [50, 75], "color": "rgba(243,156,18,0.2)"},
                        {"range": [75, 100], "color": "rgba(39,174,96,0.2)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.75,
                        "value": confidence_threshold * 100
                    },
                },
                title={"text": "Model Confidence", "font": {"color": "white"}},
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", family="DM Sans"),
                height=270,
                margin=dict(t=20, b=10),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            with st.expander("📋 Full Disease Report", expanded=True):
                st.markdown(f"**Description:** {disease['description']}")
                st.markdown(f"**Affected Parts:** `{disease['affected_part']}`")
                st.markdown(f"**Spread:** `{disease['spread']}`")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**🔬 Symptoms:**")
                    for s in disease["symptoms"]:
                        st.markdown(f"- {s}")
                with c2:
                    st.markdown("**🌿 Causes:**")
                    for c in disease["causes"]:
                        st.markdown(f"- {c}")

                st.markdown("**💊 Solutions & Treatments:**")
                for sol in disease["solutions"]:
                    st.success(f"✔ {sol}")

                st.markdown("**🛡️ Prevention Measures:**")
                for prev in disease["prevention"]:
                    st.info(f"🔒 {prev}")

            report_text = f"""POTATO DISEASE DETECTION REPORT
================================
Predicted Disease : {pred_clean}
Scientific Name   : {disease['scientific_name']}
Confidence        : {confidence:.2%}
Severity          : {severity}

DESCRIPTION
-----------
{disease['description']}

SYMPTOMS
--------
""" + "\n".join(f"• {s}" for s in disease["symptoms"]) + """

CAUSES
------
""" + "\n".join(f"• {c}" for c in disease["causes"]) + """

SOLUTIONS
---------
""" + "\n".join(f"• {s}" for s in disease["solutions"]) + """

PREVENTION
----------
""" + "\n".join(f"• {p}" for p in disease["prevention"])

            st.download_button(
                "📥 Download Report (.txt)",
                data=report_text,
                file_name="potato_disease_report.txt",
                mime="text/plain",
            )

            with st.expander("🧩 Hybrid Model Internals", expanded=False):
                st.write("**Pipeline:** Image → CNN Feature Extractor → StandardScaler → SVM")
                st.write(f"**Feature vector length:** {len(feature_vec)}")
                st.write(f"**Top confidence class:** {pred_clean}")
        else:
            st.markdown("""
            <div style="text-align:center; padding: 4rem 2rem; color: #81c784; 
                        border: 2px dashed rgba(76,175,80,0.3); border-radius: 16px;">
                <div style="font-size:4rem;">🌿</div>
                <div style="font-size:1.1rem; margin-top:1rem;">
                    Upload a potato leaf image on the left to begin analysis
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — Dashboard
# ─────────────────────────────────────────────────────────────────────────────
with tab_dashboard:
    st.markdown('<div class="section-header">Model Performance Overview</div>', unsafe_allow_html=True)

    model_comparison_csv = "model_comparison.csv"
    history_json = "training_history.json"
    hybrid_cm_path = "cnn_svm_confusion_matrix.png"
    hybrid_roc_path = "cnn_svm_roc_curves.png"
    svm_pred_csv = "svm_test_predictions.csv"
    svm_grid_csv = "svm_gridsearch_results.csv"

    m1, m2, m3, m4 = st.columns(4)
    if os.path.exists(model_comparison_csv):
        df_cmp = pd.read_csv(model_comparison_csv)
        hybrid_row = df_cmp[df_cmp["Model"] == "CNN_SVM"]
        if len(hybrid_row) > 0:
            row = hybrid_row.iloc[0]
            m1.metric("Test Accuracy", f"{float(row['Accuracy']):.2%}")
            m2.metric("Precision", f"{float(row['Macro_Precision']):.2%}")
            m3.metric("Recall", f"{float(row['Macro_Recall']):.2%}")
            m4.metric("F1-Score", f"{float(row['Macro_F1']):.2%}")
        else:
            m1.metric("Test Accuracy", "—")
            m2.metric("Precision", "—")
            m3.metric("Recall", "—")
            m4.metric("F1-Score", "—")
    else:
        m1.metric("Test Accuracy", "—")
        m2.metric("Precision", "—")
        m3.metric("Recall", "—")
        m4.metric("F1-Score", "—")

    st.markdown("---")
    st.markdown('<div class="section-header">Training & Validation Curves</div>', unsafe_allow_html=True)

    if os.path.exists(history_json):
        with open(history_json) as f:
            hist = json.load(f)

        if "accuracy" in hist and "val_accuracy" in hist and "loss" in hist and "val_loss" in hist:
            epochs = list(range(1, len(hist["accuracy"]) + 1))

            fig_curves = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"))
            fig_curves.add_trace(go.Scatter(
                x=epochs, y=hist["accuracy"], name="Train Acc",
                line=dict(color="#4caf50", width=2.5)), row=1, col=1)
            fig_curves.add_trace(go.Scatter(
                x=epochs, y=hist["val_accuracy"], name="Val Acc",
                line=dict(color="#81d4fa", width=2.5, dash="dash")), row=1, col=1)
            fig_curves.add_trace(go.Scatter(
                x=epochs, y=hist["loss"], name="Train Loss",
                line=dict(color="#ef9a9a", width=2.5)), row=1, col=2)
            fig_curves.add_trace(go.Scatter(
                x=epochs, y=hist["val_loss"], name="Val Loss",
                line=dict(color="#f06292", width=2.5, dash="dash")), row=1, col=2)

            fig_curves.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", family="DM Sans"),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                height=380,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_curves, use_container_width=True)

    st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
    if os.path.exists(hybrid_cm_path):
        st.image(hybrid_cm_path, use_container_width=True)
    else:
        st.info("Confusion matrix image not found.")

    st.markdown('<div class="section-header">Model Comparison Table</div>', unsafe_allow_html=True)
    if os.path.exists(model_comparison_csv):
        df_cmp = pd.read_csv(model_comparison_csv)
        st.dataframe(df_cmp, use_container_width=True)

    st.markdown('<div class="section-header">ROC Curves</div>', unsafe_allow_html=True)
    if os.path.exists(hybrid_roc_path):
        st.image(hybrid_roc_path, use_container_width=True)
    else:
        st.info("ROC curve image not found.")

    st.markdown('<div class="section-header">Prediction Sample Table</div>', unsafe_allow_html=True)
    if os.path.exists(svm_pred_csv):
        df_pred = pd.read_csv(svm_pred_csv)
        st.dataframe(df_pred.head(20), use_container_width=True)

    st.markdown('<div class="section-header">SVM Grid Search Results</div>', unsafe_allow_html=True)
    if os.path.exists(svm_grid_csv):
        df_grid = pd.read_csv(svm_grid_csv)
        show_cols = [c for c in ["params", "mean_test_score", "rank_test_score"] if c in df_grid.columns]
        st.dataframe(df_grid[show_cols].head(10), use_container_width=True)

    st.markdown('<div class="section-header">Class Performance Radar</div>', unsafe_allow_html=True)

    if os.path.exists(svm_pred_csv):
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

            df_pred = pd.read_csv(svm_pred_csv)

            true_idx = df_pred["true_label_index"].values
            pred_idx = df_pred["pred_label_index"].values

            cm = confusion_matrix(true_idx, pred_idx)
            per_acc = []
            for i in range(len(CLASS_NAMES)):
                denom = cm[i].sum()
                per_acc.append((cm[i, i] / denom) if denom > 0 else 0.0)

            prec_arr = precision_score(true_idx, pred_idx, average=None, zero_division=0)
            rec_arr = recall_score(true_idx, pred_idx, average=None, zero_division=0)
            f1_arr = f1_score(true_idx, pred_idx, average=None, zero_division=0)

            categories = ["Precision", "Recall", "F1-Score", "Accuracy"]
            fig_radar = go.Figure()
            rdr_colors = ["#f39c12", "#e74c3c", "#27ae60"]

            for i, (cls_name, color) in enumerate(zip(CLEAN_NAMES, rdr_colors)):
                vals = [float(prec_arr[i]), float(rec_arr[i]), float(f1_arr[i]), float(per_acc[i])]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=cls_name,
                    line=dict(color=color, width=2),
                    fillcolor=hex_to_rgba(color, 0.15),
                ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], color="white"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", family="DM Sans"),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                height=450,
                title="Per-Class Performance Radar",
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        except Exception as e:
            st.warning(f"Radar chart could not be rendered: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — Disease Info
# ─────────────────────────────────────────────────────────────────────────────
with tab_info:
    st.markdown('<div class="section-header">Potato Disease Encyclopedia</div>', unsafe_allow_html=True)

    for _, info in DISEASE_INFO.items():
        with st.expander(f"{info['icon']}  {info['display_name']} — {info['scientific_name']}", expanded=False):
            col_desc, col_meta = st.columns([2, 1])
            with col_desc:
                st.markdown(f"**Description:** {info['description']}")
            with col_meta:
                st.markdown(f"""
                | Property | Value |
                |---|---|
                | **Severity** | {info['severity']} |
                | **Pathogen** | *{info['scientific_name']}* |
                | **Affected Parts** | {info['affected_part']} |
                | **Spread** | {info['spread']} |
                """)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**🔬 Symptoms**")
                for s in info["symptoms"]:
                    st.markdown(f"- {s}")
            with c2:
                st.markdown("**🌿 Causes**")
                for c in info["causes"]:
                    st.markdown(f"- {c}")
            with c3:
                st.markdown("**💊 Solutions**")
                for s in info["solutions"][:4]:
                    st.markdown(f"- {s}")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — Model Architecture
# ─────────────────────────────────────────────────────────────────────────────
with tab_model:
    st.markdown('<div class="section-header">Hybrid Architecture Summary</div>', unsafe_allow_html=True)

    st.markdown("""
    **Pipeline**
    1. Input image is resized and normalized  
    2. CNN extracts a 256-dimensional deep feature vector  
    3. StandardScaler normalizes the feature vector  
    4. SVM predicts class probabilities and final disease label  
    """)

    if model_loaded and feature_extractor is not None:
        summary_str = get_model_summary_text(feature_extractor)
        st.code(summary_str, language="text")

        params = feature_extractor.count_params()
        trainable = int(sum(np.prod(v.shape) for v in feature_extractor.trainable_variables))
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("CNN Parameters", f"{params:,}")
        mc2.metric("Trainable Params", f"{trainable:,}")
        mc3.metric("Feature Dimension", "256")

        st.markdown("### SVM Details")
        try:
            svm_kernel = getattr(svm_model, "kernel", "rbf")
            svm_c = getattr(svm_model, "C", "—")
            svm_gamma = getattr(svm_model, "gamma", "—")

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Kernel", str(svm_kernel))
            sc2.metric("C", str(svm_c))
            sc3.metric("Gamma", str(svm_gamma))
        except Exception:
            st.info("SVM details could not be displayed.")

        if metadata:
            st.markdown("### Metadata")
            st.json(metadata)
    elif TF_AVAILABLE:
        model_for_summary = build_feature_extractor_architecture()
        summary_str = get_model_summary_text(model_for_summary)
        st.code(summary_str, language="text")


# ═══════════════════════════════════════════════════════════════════════════════
#  Footer
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#81c784; font-size:0.85rem; padding:1rem 0;">
    🥔 Potato Disease Detection System &nbsp;|&nbsp; 
    Built with TensorFlow, Scikit-learn &amp; Streamlit &nbsp;|&nbsp;
    CNN Feature Extractor + SVM Classifier
</div>
""", unsafe_allow_html=True)

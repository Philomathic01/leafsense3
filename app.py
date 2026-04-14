"""
🥔 Potato Disease Detection App
Streamlit frontend — CNN Feature Extractor + SVM classifier
UI kept close to the older app layout/style.
Added:
1. Hindi / English toggle
2. "Not a Leaf Detected" rejection layer before CNN+SVM prediction
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
.result-nonleaf  { background: rgba(149,165,166,0.15); border: 2px solid #95a5a6; }

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
#  Language Strings
# ═══════════════════════════════════════════════════════════════════════════════
TEXT = {
    "en": {
        "app_name": "Potato Disease Detector",
        "model_settings": "⚙️ Model Settings",
        "language": "Language",
        "feature_model": "Feature extractor (.keras)",
        "svm_model": "SVM model (.joblib)",
        "scaler": "Scaler (.joblib)",
        "metadata": "Metadata (.json)",
        "input_size": "Input image size",
        "confidence_threshold": "Confidence threshold",
        "leaf_threshold": "Non-leaf sensitivity",
        "about": "📋 About",
        "about_text": """
This app detects **potato leaf diseases** using a **CNN + SVM hybrid model**.

**Pipeline:**
- CNN extracts deep visual features
- StandardScaler normalizes features
- SVM predicts the final class
- A non-leaf rejection layer filters unrelated uploads

**Classes:**
- 🍂 Early Blight
- ⚠️ Late Blight
- ✅ Healthy
- 🚫 Not a Leaf Detected
""",
        "hero_title": "🥔 Potato Disease Detection System",
        "hero_subtitle": "AI-powered CNN + SVM hybrid model — Upload a leaf image to detect Early Blight, Late Blight, Healthy status, or non-leaf input",
        "tabs": ["🔍 Predict Disease", "📊 Model Performance", "📖 Disease Information", "🧠 Model Architecture"],
        "upload_header": "Upload Leaf Image",
        "upload_label": "Choose a potato leaf image",
        "upload_help": "Upload a clear image of the potato leaf for analysis.",
        "uploaded_image": "Uploaded Image",
        "image_info": "Image Info",
        "analysis_results": "Analysis Results",
        "analysing": "🔬 Analysing image…",
        "demo_mode": "🎭 Demo mode — predictions are illustrative only.",
        "low_conf": "⚠️ Low confidence. The image may be unclear or not a potato leaf.",
        "prediction_confidence": "Prediction Confidence",
        "full_report": "📋 Full Disease Report",
        "description": "Description",
        "affected_parts": "Affected Parts",
        "spread": "Spread",
        "symptoms": "🔬 Symptoms",
        "causes": "🌿 Causes",
        "solutions": "💊 Solutions & Treatments",
        "prevention": "🛡️ Prevention Measures",
        "download_report": "📥 Download Report (.txt)",
        "placeholder": "Upload a potato leaf image on the left to begin analysis",
        "performance_overview": "Model Performance Overview",
        "training_curves": "Training & Validation Curves",
        "confusion_matrix": "Confusion Matrix",
        "model_comparison": "Model Comparison Table",
        "roc_curves": "ROC Curves",
        "prediction_sample_table": "Prediction Sample Table",
        "svm_results": "SVM Grid Search Results",
        "radar": "Class Performance Radar",
        "encyclopedia": "Potato Disease Encyclopedia",
        "architecture": "Hybrid Architecture Summary",
        "pipeline_title": "Pipeline",
        "internals": "🧩 Hybrid Model Internals",
        "feature_vector": "Feature vector length",
        "top_class": "Top confidence class",
        "non_leaf_note": "If the uploaded image does not look like a leaf, the app returns a non-leaf result before disease classification.",
        "footer": "🥔 Potato Disease Detection System | Built with TensorFlow, Scikit-learn & Streamlit | CNN Feature Extractor + SVM Classifier",
        "success_loaded": "✅ Hybrid model assets loaded successfully.",
        "missing_assets": "📌 Hybrid model files not found. Demo mode is active — sample predictions will be shown.",
        "not_installed": "⚠️ TensorFlow/Keras not installed correctly.",
        "non_leaf_detected": "Non-leaf detected",
        "non_leaf_score": "Leaf-likelihood score",
        "test_accuracy": "Test Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-Score",
        "cnn_params": "CNN Parameters",
        "trainable_params": "Trainable Params",
        "feature_dim": "Feature Dimension",
        "kernel": "Kernel",
        "c_value": "C",
        "gamma": "Gamma",
        "metadata_title": "Metadata",
        "pipeline_text": """
1. Input image is resized and normalized  
2. A leaf-likelihood gate checks whether the image is a leaf  
3. CNN extracts a 256-dimensional deep feature vector  
4. StandardScaler normalizes the feature vector  
5. SVM predicts class probabilities and final label  
"""
    },
    "hi": {
        "app_name": "आलू रोग पहचान ऐप",
        "model_settings": "⚙️ मॉडल सेटिंग्स",
        "language": "भाषा",
        "feature_model": "फ़ीचर एक्सट्रैक्टर (.keras)",
        "svm_model": "SVM मॉडल (.joblib)",
        "scaler": "स्केलर (.joblib)",
        "metadata": "मेटाडेटा (.json)",
        "input_size": "इनपुट इमेज साइज़",
        "confidence_threshold": "कॉन्फिडेंस थ्रेशहोल्ड",
        "leaf_threshold": "नॉन-लीफ सेंसिटिविटी",
        "about": "📋 जानकारी",
        "about_text": """
यह ऐप **CNN + SVM हाइब्रिड मॉडल** का उपयोग करके **आलू पत्ती रोग** पहचानता है।

**पाइपलाइन:**
- CNN गहरे विज़ुअल फीचर्स निकालता है
- StandardScaler फीचर्स को नॉर्मलाइज़ करता है
- SVM अंतिम क्लास बताता है
- एक non-leaf rejection layer गलत अपलोड को पहले ही रोक देती है

**क्लासेस:**
- 🍂 अर्ली ब्लाइट
- ⚠️ लेट ब्लाइट
- ✅ स्वस्थ
- 🚫 पत्ती नहीं मिली
""",
        "hero_title": "🥔 आलू रोग पहचान प्रणाली",
        "hero_subtitle": "AI-आधारित CNN + SVM हाइब्रिड मॉडल — पत्ती की इमेज अपलोड करें और Early Blight, Late Blight, Healthy या non-leaf input पहचानें",
        "tabs": ["🔍 रोग पहचान", "📊 मॉडल प्रदर्शन", "📖 रोग जानकारी", "🧠 मॉडल संरचना"],
        "upload_header": "पत्ती की इमेज अपलोड करें",
        "upload_label": "आलू पत्ती की इमेज चुनें",
        "upload_help": "विश्लेषण के लिए आलू पत्ती की साफ़ इमेज अपलोड करें।",
        "uploaded_image": "अपलोड की गई इमेज",
        "image_info": "इमेज जानकारी",
        "analysis_results": "विश्लेषण परिणाम",
        "analysing": "🔬 इमेज का विश्लेषण हो रहा है…",
        "demo_mode": "🎭 डेमो मोड — ये परिणाम केवल प्रदर्शन के लिए हैं।",
        "low_conf": "⚠️ कॉन्फिडेंस कम है। इमेज अस्पष्ट हो सकती है या यह आलू की पत्ती नहीं है।",
        "prediction_confidence": "प्रेडिक्शन कॉन्फिडेंस",
        "full_report": "📋 पूरी रोग रिपोर्ट",
        "description": "विवरण",
        "affected_parts": "प्रभावित भाग",
        "spread": "फैलाव",
        "symptoms": "🔬 लक्षण",
        "causes": "🌿 कारण",
        "solutions": "💊 उपचार और समाधान",
        "prevention": "🛡️ बचाव के उपाय",
        "download_report": "📥 रिपोर्ट डाउनलोड करें (.txt)",
        "placeholder": "विश्लेषण शुरू करने के लिए बाईं ओर आलू पत्ती की इमेज अपलोड करें",
        "performance_overview": "मॉडल प्रदर्शन अवलोकन",
        "training_curves": "ट्रेनिंग और वैलिडेशन कर्व्स",
        "confusion_matrix": "कन्फ्यूज़न मैट्रिक्स",
        "model_comparison": "मॉडल तुलना तालिका",
        "roc_curves": "ROC कर्व्स",
        "prediction_sample_table": "प्रेडिक्शन सैंपल तालिका",
        "svm_results": "SVM ग्रिड सर्च परिणाम",
        "radar": "क्लास परफॉर्मेंस रडार",
        "encyclopedia": "आलू रोग ज्ञानकोश",
        "architecture": "हाइब्रिड आर्किटेक्चर सारांश",
        "pipeline_title": "पाइपलाइन",
        "internals": "🧩 हाइब्रिड मॉडल विवरण",
        "feature_vector": "फ़ीचर वेक्टर लंबाई",
        "top_class": "सर्वाधिक कॉन्फिडेंस क्लास",
        "non_leaf_note": "यदि अपलोड की गई इमेज पत्ती जैसी नहीं लगती है, तो रोग पहचान से पहले ऐप non-leaf परिणाम देगा।",
        "footer": "🥔 आलू रोग पहचान प्रणाली | TensorFlow, Scikit-learn और Streamlit से निर्मित | CNN Feature Extractor + SVM Classifier",
        "success_loaded": "✅ हाइब्रिड मॉडल फ़ाइलें सफलतापूर्वक लोड हो गईं।",
        "missing_assets": "📌 हाइब्रिड मॉडल फ़ाइलें नहीं मिलीं। डेमो मोड सक्रिय है।",
        "not_installed": "⚠️ TensorFlow/Keras सही तरह इंस्टॉल नहीं है।",
        "non_leaf_detected": "पत्ती नहीं मिली",
        "non_leaf_score": "लीफ-लाइकलीहुड स्कोर",
        "test_accuracy": "टेस्ट एक्यूरेसी",
        "precision": "प्रिसीजन",
        "recall": "रिकॉल",
        "f1": "F1-स्कोर",
        "cnn_params": "CNN पैरामीटर्स",
        "trainable_params": "ट्रेन करने योग्य पैरामीटर्स",
        "feature_dim": "फ़ीचर डायमेंशन",
        "kernel": "कर्नेल",
        "c_value": "C",
        "gamma": "गामा",
        "metadata_title": "मेटाडेटा",
        "pipeline_text": """
1. इनपुट इमेज को resize और normalize किया जाता है  
2. leaf-likelihood gate जाँचता है कि इमेज पत्ती है या नहीं  
3. CNN 256-डायमेंशनल deep feature vector बनाता है  
4. StandardScaler फीचर vector को normalize करता है  
5. SVM class probabilities और final label देता है  
"""
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def tr(key, lang):
    return TEXT.get(lang, TEXT["en"]).get(key, key)


def hex_to_rgba(hex_color, alpha=0.15):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def clean_display_name(name: str) -> str:
    return name.replace("Potato___", "").replace("Potato__", "").replace("_", " ").title()


def build_feature_extractor_architecture(input_shape=(224, 224, 3)):
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


def canonicalize_class_name(label: str) -> str:
    if not isinstance(label, str):
        return ""

    x = label.strip().lower()
    x = x.replace("-", "_").replace(" ", "_")
    x = x.replace("potato___", "")
    x = x.replace("potato__", "")
    x = x.replace("potato_", "")

    mapping = {
        "early_blight": "Potato___Early_blight",
        "late_blight": "Potato___Late_blight",
        "healthy": "Potato___healthy",
        "not_leaf_detected": "Not_Leaf_Detected",
        "not_a_leaf_detected": "Not_Leaf_Detected",
        "not_leaf": "Not_Leaf_Detected",
        "non_leaf": "Not_Leaf_Detected",
    }

    return mapping.get(x, label.strip())


def resolve_class_names(metadata):
    raw = metadata.get("class_names", [])

    if isinstance(raw, str):
        raw = [raw]
    elif isinstance(raw, dict):
        raw = list(raw.values())
    elif not isinstance(raw, list):
        raw = []

    if not raw:
        idx_to_class = metadata.get("idx_to_class", {})
        if isinstance(idx_to_class, dict) and idx_to_class:
            try:
                raw = [idx_to_class[str(i)] for i in sorted(int(k) for k in idx_to_class.keys())]
            except Exception:
                raw = list(idx_to_class.values())

    raw = [canonicalize_class_name(x) for x in raw]
    raw = list(dict.fromkeys(raw))  # remove duplicates, keep order

    valid = {"Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"}
    raw = [x for x in raw if x in valid]

    if len(raw) != 3:
        raw = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

    return raw


def analyze_leaf_likelihood(img: Image.Image):
    """
    App-level non-leaf rejection.
    This is not a trained 4th class.
    It is a practical image gate before CNN+SVM prediction.
    """
    img_small = img.convert("RGB").resize((224, 224))
    arr = np.array(img_small, dtype=np.uint8)

    hsv = np.array(img_small.convert("HSV"), dtype=np.uint8)
    h = hsv[:, :, 0].astype(np.float32) * (360.0 / 255.0)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    # Green vegetation
    green_mask = (h >= 35) & (h <= 110) & (s >= 0.18) & (v >= 0.15)

    # Wider plant-like mask to also capture yellow/brown diseased leaves
    plant_mask = (h >= 15) & (h <= 120) & (s >= 0.12) & (v >= 0.12)

    # Brown/dry leaf-like mask
    brown_mask = (h >= 10) & (h <= 35) & (s >= 0.20) & (v >= 0.10)

    green_ratio = float(np.mean(green_mask))
    plant_ratio = float(np.mean(plant_mask | brown_mask))
    sat_mean = float(np.mean(s))
    std_rgb = float(np.std(arr / 255.0))

    # heuristic combined score
    leaf_score = 0.50 * plant_ratio + 0.25 * green_ratio + 0.15 * sat_mean + 0.10 * std_rgb
    leaf_score = max(0.0, min(1.0, leaf_score))

    return {
        "leaf_score": leaf_score,
        "green_ratio": green_ratio,
        "plant_ratio": plant_ratio,
        "sat_mean": sat_mean,
        "std_rgb": std_rgb,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"## 🥔 {TEXT['en']['app_name']}")
    st.markdown("---")

    language_label = TEXT["en"]["language"]
    language_option = st.selectbox(language_label, ["English", "हिंदी"], index=0)
    LANG = "hi" if language_option == "हिंदी" else "en"

    st.markdown(f"### {tr('model_settings', LANG)}")
    feature_model_path = st.text_input(
        tr("feature_model", LANG),
        value="best_cnn_feature_extractor.keras",
        help="Path to the saved CNN feature extractor model.",
    )
    svm_model_path = st.text_input(
        tr("svm_model", LANG),
        value="svm_on_cnn_features.joblib",
        help="Path to the trained SVM model.",
    )
    scaler_path = st.text_input(
        tr("scaler", LANG),
        value="feature_scaler.joblib",
        help="Path to the saved feature scaler.",
    )
    metadata_path = st.text_input(
        tr("metadata", LANG),
        value="metadata.json",
        help="Path to the metadata JSON generated during training.",
    )

    img_size_option = st.selectbox(tr("input_size", LANG), [224, 128, 256], index=0)
    confidence_threshold = st.slider(tr("confidence_threshold", LANG), 0.0, 1.0, 0.5, 0.05)
    non_leaf_threshold = st.slider(tr("leaf_threshold", LANG), 0.0, 1.0, 0.16, 0.01)

    st.markdown("---")
    st.markdown(f"### {tr('about', LANG)}")
    st.markdown(tr("about_text", LANG))

    st.markdown("---")
    if TF_AVAILABLE:
        st.caption(f"TensorFlow: {tf.__version__}")
        st.caption(f"Keras: {keras.__version__}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Hero Banner
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero-banner">
    <h1>{tr('hero_title', LANG)}</h1>
    <p>{tr('hero_subtitle', LANG)}</p>
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
        st.success(tr("success_loaded", LANG))
elif not TF_AVAILABLE:
    st.warning(tr("not_installed", LANG))
else:
    st.info(tr("missing_assets", LANG))


# ═══════════════════════════════════════════════════════════════════════════════
#  Class names
# ═══════════════════════════════════════════════════════════════════════════════
BASE_CLASS_NAMES = resolve_class_names(metadata)
if len(BASE_CLASS_NAMES) != 3:
    BASE_CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

CLASS_NAMES = BASE_CLASS_NAMES + ["Not_Leaf_Detected"]

DISPLAY_NAMES = []
for name in CLASS_NAMES:
    info = get_disease_info(name, LANG)
    DISPLAY_NAMES.append(info["display_name"])


# ═══════════════════════════════════════════════════════════════════════════════
#  Tabs
# ═══════════════════════════════════════════════════════════════════════════════
tab_predict, tab_dashboard, tab_info, tab_model = st.tabs(tr("tabs", LANG))


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — Predict
# ─────────────────────────────────────────────────────────────────────────────
with tab_predict:
    col_upload, col_result = st.columns([1, 1.5], gap="large")

    with col_upload:
        st.markdown(f'<div class="section-header">{tr("upload_header", LANG)}</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            tr("upload_label", LANG),
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help=tr("upload_help", LANG),
        )

        if uploaded_file:
            img_pil = Image.open(uploaded_file)
            st.image(img_pil, caption=tr("uploaded_image", LANG), use_container_width=True)

            st.markdown(f'<div class="section-header">{tr("image_info", LANG)}</div>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Width",  f"{img_pil.width}px")
            m2.metric("Height", f"{img_pil.height}px")
            m3.metric("Mode",   img_pil.mode)
            st.metric("File size", f"{uploaded_file.size / 1024:.1f} KB")

    with col_result:
        st.markdown(f'<div class="section-header">{tr("analysis_results", LANG)}</div>', unsafe_allow_html=True)

        if uploaded_file:
            with st.spinner(tr("analysing", LANG)):
                time.sleep(0.4)
                img_array, _ = preprocess_image(img_pil, (img_size_option, img_size_option))

                leaf_stats = analyze_leaf_likelihood(img_pil)
                leaf_score = leaf_stats["leaf_score"]
                non_leaf_detected = leaf_score < non_leaf_threshold

                if model_loaded and not non_leaf_detected:
                    probs3, pred_idx, feature_vec = predict_hybrid(
                        feature_extractor, scaler, svm_model, img_array
                    )
                    probs = list(probs3) + [0.0]
                else:
                    feature_vec = np.zeros(256)
                    if not model_loaded:
                        np.random.seed(int(sum(uploaded_file.getvalue()[:50])) % 10000)
                        probs = list(np.random.dirichlet(np.ones(3) * 0.5)) + [0.0]
                        pred_idx = int(np.argmax(probs[:3]))
                        st.info(tr("demo_mode", LANG))
                    else:
                        probs = [0.0, 0.0, 0.0, 1.0]
                        pred_idx = 3

            confidence = float(probs[pred_idx])
            pred_class = CLASS_NAMES[pred_idx]
            disease = get_disease_info(pred_class, LANG)

            if pred_class != "Not_Leaf_Detected" and confidence < confidence_threshold:
                st.warning(tr("low_conf", LANG))

            severity = disease["severity"]

            if pred_class == "Not_Leaf_Detected":
                css_class = "result-nonleaf"
            elif "healthy" in pred_class.lower():
                css_class = "result-healthy"
            elif "late" in pred_class.lower():
                css_class = "result-late"
            else:
                css_class = "result-early"

            st.markdown(f"""
            <div class="result-box {css_class}">
                <h2>{disease['icon']} {disease['display_name']}</h2>
                <p>{'Scientific name' if LANG == 'en' else 'वैज्ञानिक नाम'}: <em>{disease['scientific_name']}</em></p>
                <p>{'Confidence' if LANG == 'en' else 'कॉन्फिडेंस'}: <strong style="font-size:1.3rem;">{confidence:.2%}</strong></p>
                <p>{'Severity' if LANG == 'en' else 'गंभीरता'}: <strong style="color:{disease['severity_color']}">{severity}</strong></p>
            </div>
            """, unsafe_allow_html=True)

            c_a, c_b = st.columns(2)
            c_a.metric(tr("non_leaf_score", LANG), f"{leaf_score:.2f}")
            c_b.metric(tr("confidence_threshold", LANG), f"{confidence_threshold:.2f}")

            if pred_class == "Not_Leaf_Detected":
                st.info(tr("non_leaf_note", LANG))

            st.markdown(f'<div class="section-header">{tr("prediction_confidence", LANG)}</div>', unsafe_allow_html=True)
            bar_colors = ["#f39c12", "#e74c3c", "#27ae60", "#95a5a6"]
            fig_bar = go.Figure(go.Bar(
                x=DISPLAY_NAMES,
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
                height=340,
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
                title={"text": "Model Confidence" if LANG == "en" else "मॉडल कॉन्फिडेंस",
                       "font": {"color": "white"}},
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", family="DM Sans"),
                height=270,
                margin=dict(t=20, b=10),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            with st.expander(tr("full_report", LANG), expanded=True):
                st.markdown(f"**{tr('description', LANG)}:** {disease['description']}")
                st.markdown(f"**{tr('affected_parts', LANG)}:** `{disease['affected_part']}`")
                st.markdown(f"**{tr('spread', LANG)}:** `{disease['spread']}`")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**{tr('symptoms', LANG)}**")
                    for s in disease["symptoms"]:
                        st.markdown(f"- {s}")
                with c2:
                    st.markdown(f"**{tr('causes', LANG)}**")
                    for c in disease["causes"]:
                        st.markdown(f"- {c}")

                st.markdown(f"**{tr('solutions', LANG)}**")
                for sol in disease["solutions"]:
                    st.success(f"✔ {sol}")

                st.markdown(f"**{tr('prevention', LANG)}**")
                for prev in disease["prevention"]:
                    st.info(f"🔒 {prev}")

            report_text = f"""POTATO DISEASE DETECTION REPORT
================================
Predicted Label   : {disease['display_name']}
Scientific Name   : {disease['scientific_name']}
Confidence        : {confidence:.2%}
Severity          : {severity}
Leaf Score        : {leaf_score:.2f}

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
                tr("download_report", LANG),
                data=report_text,
                file_name="potato_disease_report.txt",
                mime="text/plain",
            )

            with st.expander(tr("internals", LANG), expanded=False):
                st.write("**Pipeline:** Image → Leaf Gate → CNN Feature Extractor → StandardScaler → SVM")
                st.write(f"**{tr('feature_vector', LANG)}:** {len(feature_vec)}")
                st.write(f"**{tr('top_class', LANG)}:** {disease['display_name']}")
                st.write(f"**Plant ratio:** {leaf_stats['plant_ratio']:.3f}")
                st.write(f"**Green ratio:** {leaf_stats['green_ratio']:.3f}")
        else:
            st.markdown(f"""
            <div style="text-align:center; padding: 4rem 2rem; color: #81c784; 
                        border: 2px dashed rgba(76,175,80,0.3); border-radius: 16px;">
                <div style="font-size:4rem;">🌿</div>
                <div style="font-size:1.1rem; margin-top:1rem;">
                    {tr("placeholder", LANG)}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — Dashboard
# ─────────────────────────────────────────────────────────────────────────────
with tab_dashboard:
    st.markdown(f'<div class="section-header">{tr("performance_overview", LANG)}</div>', unsafe_allow_html=True)

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
            m1.metric(tr("test_accuracy", LANG), f"{float(row['Accuracy']):.2%}")
            m2.metric(tr("precision", LANG), f"{float(row['Macro_Precision']):.2%}")
            m3.metric(tr("recall", LANG), f"{float(row['Macro_Recall']):.2%}")
            m4.metric(tr("f1", LANG), f"{float(row['Macro_F1']):.2%}")
        else:
            m1.metric(tr("test_accuracy", LANG), "—")
            m2.metric(tr("precision", LANG), "—")
            m3.metric(tr("recall", LANG), "—")
            m4.metric(tr("f1", LANG), "—")
    else:
        m1.metric(tr("test_accuracy", LANG), "—")
        m2.metric(tr("precision", LANG), "—")
        m3.metric(tr("recall", LANG), "—")
        m4.metric(tr("f1", LANG), "—")

    st.markdown("---")
    st.markdown(f'<div class="section-header">{tr("training_curves", LANG)}</div>', unsafe_allow_html=True)

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

    st.markdown(f'<div class="section-header">{tr("confusion_matrix", LANG)}</div>', unsafe_allow_html=True)
    if os.path.exists(hybrid_cm_path):
        st.image(hybrid_cm_path, use_container_width=True)

    st.markdown(f'<div class="section-header">{tr("model_comparison", LANG)}</div>', unsafe_allow_html=True)
    if os.path.exists(model_comparison_csv):
        df_cmp = pd.read_csv(model_comparison_csv)
        st.dataframe(df_cmp, use_container_width=True)

    st.markdown(f'<div class="section-header">{tr("roc_curves", LANG)}</div>', unsafe_allow_html=True)
    if os.path.exists(hybrid_roc_path):
        st.image(hybrid_roc_path, use_container_width=True)

    st.markdown(f'<div class="section-header">{tr("prediction_sample_table", LANG)}</div>', unsafe_allow_html=True)
    if os.path.exists(svm_pred_csv):
        df_pred = pd.read_csv(svm_pred_csv)
        st.dataframe(df_pred.head(20), use_container_width=True)

    st.markdown(f'<div class="section-header">{tr("svm_results", LANG)}</div>', unsafe_allow_html=True)
    if os.path.exists(svm_grid_csv):
        df_grid = pd.read_csv(svm_grid_csv)
        show_cols = [c for c in ["params", "mean_test_score", "rank_test_score"] if c in df_grid.columns]
        st.dataframe(df_grid[show_cols].head(10), use_container_width=True)

    st.markdown(f'<div class="section-header">{tr("radar", LANG)}</div>', unsafe_allow_html=True)
    if os.path.exists(svm_pred_csv):
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

            df_pred = pd.read_csv(svm_pred_csv)

            true_idx = df_pred["true_label_index"].values
            pred_idx = df_pred["pred_label_index"].values

            cm = confusion_matrix(true_idx, pred_idx)
            per_acc = []
            for i in range(len(BASE_CLASS_NAMES)):
                denom = cm[i].sum()
                per_acc.append((cm[i, i] / denom) if denom > 0 else 0.0)

            prec_arr = precision_score(true_idx, pred_idx, average=None, zero_division=0)
            rec_arr = recall_score(true_idx, pred_idx, average=None, zero_division=0)
            f1_arr = f1_score(true_idx, pred_idx, average=None, zero_division=0)

            base_display_names = [get_disease_info(name, LANG)["display_name"] for name in BASE_CLASS_NAMES]
            categories = [tr("precision", LANG), tr("recall", LANG), tr("f1", LANG), "Accuracy"]
            fig_radar = go.Figure()
            rdr_colors = ["#f39c12", "#e74c3c", "#27ae60"]

            for i, (cls_name, color) in enumerate(zip(base_display_names, rdr_colors)):
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
                title=tr("radar", LANG),
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        except Exception as e:
            st.warning(f"Radar chart could not be rendered: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — Disease Info
# ─────────────────────────────────────────────────────────────────────────────
with tab_info:
    st.markdown(f'<div class="section-header">{tr("encyclopedia", LANG)}</div>', unsafe_allow_html=True)

    for key, info_pack in DISEASE_INFO.items():
        info = info_pack.get(LANG, info_pack["en"])
        with st.expander(f"{info['icon']}  {info['display_name']} — {info['scientific_name']}", expanded=False):
            col_desc, col_meta = st.columns([2, 1])
            with col_desc:
                st.markdown(f"**{tr('description', LANG)}:** {info['description']}")
            with col_meta:
                st.markdown(f"""
                | {"Property" if LANG == "en" else "गुण"} | {"Value" if LANG == "en" else "मान"} |
                |---|---|
                | **{"Severity" if LANG == "en" else "गंभीरता"}** | {info['severity']} |
                | **{"Pathogen" if LANG == "en" else "रोगकारक"}** | *{info['scientific_name']}* |
                | **{"Affected Parts" if LANG == "en" else "प्रभावित भाग"}** | {info['affected_part']} |
                | **{"Spread" if LANG == "en" else "फैलाव"}** | {info['spread']} |
                """)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"**{tr('symptoms', LANG)}**")
                for s in info["symptoms"]:
                    st.markdown(f"- {s}")
            with c2:
                st.markdown(f"**{tr('causes', LANG)}**")
                for c in info["causes"]:
                    st.markdown(f"- {c}")
            with c3:
                st.markdown(f"**{tr('solutions', LANG)}**")
                for s in info["solutions"][:4]:
                    st.markdown(f"- {s}")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — Model Architecture
# ─────────────────────────────────────────────────────────────────────────────
with tab_model:
    st.markdown(f'<div class="section-header">{tr("architecture", LANG)}</div>', unsafe_allow_html=True)

    st.markdown(f"""
**{tr("pipeline_title", LANG)}**
{tr("pipeline_text", LANG)}
""")

    if model_loaded and feature_extractor is not None:
        summary_str = get_model_summary_text(feature_extractor)
        st.code(summary_str, language="text")

        params = feature_extractor.count_params()
        trainable = int(sum(np.prod(v.shape) for v in feature_extractor.trainable_variables))
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric(tr("cnn_params", LANG), f"{params:,}")
        mc2.metric(tr("trainable_params", LANG), f"{trainable:,}")
        mc3.metric(tr("feature_dim", LANG), "256")

        st.markdown("### SVM Details")
        try:
            svm_kernel = getattr(svm_model, "kernel", "rbf")
            svm_c = getattr(svm_model, "C", "—")
            svm_gamma = getattr(svm_model, "gamma", "—")

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric(tr("kernel", LANG), str(svm_kernel))
            sc2.metric(tr("c_value", LANG), str(svm_c))
            sc3.metric(tr("gamma", LANG), str(svm_gamma))
        except Exception:
            st.info("SVM details could not be displayed.")

        if metadata:
            st.markdown(f"### {tr('metadata_title', LANG)}")
            st.json(metadata)
    elif TF_AVAILABLE:
        model_for_summary = build_feature_extractor_architecture()
        summary_str = get_model_summary_text(model_for_summary)
        st.code(summary_str, language="text")


# ═══════════════════════════════════════════════════════════════════════════════
#  Footer
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(f"""
<div style="text-align:center; color:#81c784; font-size:0.85rem; padding:1rem 0;">
    {tr("footer", LANG)}
</div>
""", unsafe_allow_html=True)

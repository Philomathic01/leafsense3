# ============================================================
# POTATO DISEASE CLASSIFICATION
# Hybrid Pipeline: Custom CNN + SVM on Deep Features
# Colab-ready complete code
# ============================================================

# ---------------------------
# 1. Install required libs
# ---------------------------
!pip install -q tensorflow scikit-learn seaborn pandas matplotlib pillow joblib

# ---------------------------
# 2. Imports
# ---------------------------
import os
import json
import random
import warnings
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_curve, auc
)
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ---------------------------
# 3. Reproducibility
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
keras.utils.set_random_seed(SEED)

print("TensorFlow version:", tf.__version__)
print("GPU available:", len(tf.config.list_physical_devices("GPU")) > 0)

# ---------------------------
# 4. Mount Google Drive
# ---------------------------
from google.colab import drive
drive.mount('/content/drive')

# ============================================================
# 5. PATHS
# ============================================================
# Change BASE_DIR only if your dataset path is different
BASE_DIR = "/content/drive/MyDrive/dataset/Potato"

TRAIN_DIR = os.path.join(BASE_DIR, "Train")
VAL_DIR   = os.path.join(BASE_DIR, "Valid")
TEST_DIR  = os.path.join(BASE_DIR, "Test")

OUTPUT_DIR = "/content/drive/MyDrive/Potato_CNN_SVM_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CNN_MODEL_PATH        = os.path.join(OUTPUT_DIR, "best_cnn_classifier.keras")
FEATURE_MODEL_PATH    = os.path.join(OUTPUT_DIR, "best_cnn_feature_extractor.keras")
SVM_MODEL_PATH        = os.path.join(OUTPUT_DIR, "svm_on_cnn_features.joblib")
SCALER_PATH           = os.path.join(OUTPUT_DIR, "feature_scaler.joblib")
HISTORY_JSON_PATH     = os.path.join(OUTPUT_DIR, "training_history.json")
METADATA_JSON_PATH    = os.path.join(OUTPUT_DIR, "metadata.json")
COUNTS_CSV_PATH       = os.path.join(OUTPUT_DIR, "dataset_counts.csv")
COMPARISON_CSV_PATH   = os.path.join(OUTPUT_DIR, "model_comparison.csv")
SVM_GRID_CSV_PATH     = os.path.join(OUTPUT_DIR, "svm_gridsearch_results.csv")
TEST_PRED_CSV_PATH    = os.path.join(OUTPUT_DIR, "svm_test_predictions.csv")

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

for p in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Directory not found: {p}")

# ============================================================
# 6. DATASET INFO
# ============================================================
def list_classes(split_dir):
    return sorted([
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    ])

CLASS_NAMES = list_classes(TRAIN_DIR)
NUM_CLASSES = len(CLASS_NAMES)

if NUM_CLASSES < 2:
    raise ValueError("Need at least 2 classes for classification.")

CLEAN_CLASS_NAMES = [c.replace("Potato__", "").replace("_", " ") for c in CLASS_NAMES]

print("Classes found:", CLASS_NAMES)
print("Number of classes:", NUM_CLASSES)

def count_images(split_dir, class_names):
    counts = {}
    for cls in class_names:
        cls_path = os.path.join(split_dir, cls)
        if os.path.isdir(cls_path):
            counts[cls] = len([
                f for f in os.listdir(cls_path)
                if f.lower().endswith(VALID_EXTS)
            ])
        else:
            counts[cls] = 0
    return counts

train_counts = count_images(TRAIN_DIR, CLASS_NAMES)
val_counts   = count_images(VAL_DIR, CLASS_NAMES)
test_counts  = count_images(TEST_DIR, CLASS_NAMES)

counts_df = pd.DataFrame({
    "Class": CLASS_NAMES,
    "Train": [train_counts[c] for c in CLASS_NAMES],
    "Validation": [val_counts[c] for c in CLASS_NAMES],
    "Test": [test_counts[c] for c in CLASS_NAMES]
})
counts_df.to_csv(COUNTS_CSV_PATH, index=False)

print("\nDataset counts:")
print(counts_df)

# ============================================================
# 7. EDA PLOTS
# ============================================================
# 7.1 Distribution bar chart
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
split_info = [
    ("Train", train_counts, "#2ecc71"),
    ("Validation", val_counts, "#e74c3c"),
    ("Test", test_counts, "#3498db"),
]

for ax, (split_name, split_counts, color) in zip(axes, split_info):
    labels = [c.replace("Potato__", "") for c in CLASS_NAMES]
    values = [split_counts[c] for c in CLASS_NAMES]
    bars = ax.bar(labels, values, color=color, edgecolor="black", alpha=0.85)
    ax.set_title(f"{split_name} Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    ax.tick_params(axis='x', rotation=20)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val),
                ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_distribution.png"), dpi=200, bbox_inches="tight")
plt.show()

# 7.2 Pie chart for train split
plt.figure(figsize=(7, 7))
plt.pie(
    [train_counts[c] for c in CLASS_NAMES],
    labels=CLEAN_CLASS_NAMES,
    autopct="%1.1f%%",
    startangle=140,
    explode=[0.04] * NUM_CLASSES
)
plt.title("Training Set Class Distribution", fontsize=14, fontweight="bold")
plt.savefig(os.path.join(OUTPUT_DIR, "eda_train_pie_chart.png"), dpi=200, bbox_inches="tight")
plt.show()

# 7.3 Sample images grid
n_cols = 5
fig, axes = plt.subplots(NUM_CLASSES, n_cols, figsize=(4*n_cols, 3*NUM_CLASSES))

if NUM_CLASSES == 1:
    axes = np.expand_dims(axes, axis=0)

for r, cls in enumerate(CLASS_NAMES):
    cls_path = os.path.join(TRAIN_DIR, cls)
    images = [
        f for f in os.listdir(cls_path)
        if f.lower().endswith(VALID_EXTS)
    ][:n_cols]

    for c in range(n_cols):
        ax = axes[r, c]
        if c < len(images):
            img_path = os.path.join(cls_path, images[c])
            img = Image.open(img_path).convert("RGB").resize((224, 224))
            ax.imshow(img)
            ax.set_title(CLEAN_CLASS_NAMES[r] if c == 0 else "", fontsize=12, fontweight="bold")
        ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_sample_images.png"), dpi=200, bbox_inches="tight")
plt.show()

# ============================================================
# 8. IMAGE GENERATORS
# ============================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.10,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.85, 1.15],
    fill_mode="nearest"
)

eval_aug = ImageDataGenerator(rescale=1./255)

# For CNN training
train_gen = train_aug.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED
)

# For feature extraction (NO augmentation, fixed order)
train_eval_gen = eval_aug.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

val_gen = eval_aug.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_gen = eval_aug.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("\nClass indices from generator:", train_gen.class_indices)

# Optional sanity check
assert list(train_gen.class_indices.keys()) == CLASS_NAMES, "Class order mismatch between folder listing and generator."

# Augmented sample visualization
x_batch, y_batch = next(train_gen)
fig, axes = plt.subplots(2, 6, figsize=(18, 6))
for i, ax in enumerate(axes.flatten()):
    idx = i % len(x_batch)
    ax.imshow(x_batch[idx])
    ax.set_title(CLEAN_CLASS_NAMES[np.argmax(y_batch[idx])], fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "augmented_samples.png"), dpi=200, bbox_inches="tight")
plt.show()

# ============================================================
# 9. BUILD CUSTOM CNN
# ============================================================
def build_cnn_classifier(num_classes, input_shape=(224, 224, 3)):
    inputs = keras.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3,3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3,3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3,3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.Dropout(0.30)(x)

    # Block 4
    x = layers.Conv2D(256, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.Dropout(0.30)(x)

    # Dense feature block
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.50)(x)

    # This layer will be used as deep feature vector for SVM
    feature_vector = layers.Dense(256, activation="relu", name="feature_vector")(x)

    x = layers.Dropout(0.30)(feature_vector)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    model = keras.Model(inputs, outputs, name="PotatoCNN_Hybrid")
    return model

cnn_model = build_cnn_classifier(NUM_CLASSES)
cnn_model.summary()

cnn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall")
    ]
)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        CNN_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

# ============================================================
# 10. TRAIN CNN
# ============================================================
EPOCHS = 25

history = cnn_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Save training history
history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(HISTORY_JSON_PATH, "w") as f:
    json.dump(history_dict, f, indent=2)

# Plot training curves
hist = history.history
epochs_ran = range(1, len(hist["accuracy"]) + 1)

fig, axes = plt.subplots(1, 3, figsize=(22, 5))

# Accuracy
axes[0].plot(epochs_ran, hist["accuracy"], marker="o", label="Train Accuracy")
axes[0].plot(epochs_ran, hist["val_accuracy"], marker="o", label="Val Accuracy")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(epochs_ran, hist["loss"], marker="o", label="Train Loss")
axes[1].plot(epochs_ran, hist["val_loss"], marker="o", label="Val Loss")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Precision and Recall
axes[2].plot(epochs_ran, hist["precision"], marker="o", label="Train Precision")
axes[2].plot(epochs_ran, hist["val_precision"], marker="o", label="Val Precision")
axes[2].plot(epochs_ran, hist["recall"], marker="o", label="Train Recall")
axes[2].plot(epochs_ran, hist["val_recall"], marker="o", label="Val Recall")
axes[2].set_title("Precision / Recall")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Score")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=200, bbox_inches="tight")
plt.show()

# Load best saved CNN
best_cnn = keras.models.load_model(CNN_MODEL_PATH)

# Build feature extractor from trained CNN
feature_extractor = keras.Model(
    inputs=best_cnn.input,
    outputs=best_cnn.get_layer("feature_vector").output,
    name="BestCNN_FeatureExtractor"
)
feature_extractor.save(FEATURE_MODEL_PATH)

# ============================================================
# 11. HELPERS FOR EVALUATION
# ============================================================
def save_confusion_matrices(y_true, y_pred, class_names, prefix, out_dir):
    labels = [c.replace("Potato__", "").replace("_", " ") for c in class_names]
    cm = confusion_matrix(y_true, y_pred)

    # Raw confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"{prefix} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix.lower()}_confusion_matrix.png"),
                dpi=200, bbox_inches="tight")
    plt.show()

    # Normalized confusion matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=labels, yticklabels=labels, vmin=0, vmax=1)
    plt.title(f"{prefix} - Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix.lower()}_confusion_matrix_normalized.png"),
                dpi=200, bbox_inches="tight")
    plt.show()

def save_roc_curves(y_true, y_prob, class_names, prefix, out_dir):
    labels = [c.replace("Potato__", "").replace("_", " ") for c in class_names]
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))

    plt.figure(figsize=(8, 7))
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", lw=1.5, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix} - ROC Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix.lower()}_roc_curves.png"),
                dpi=200, bbox_inches="tight")
    plt.show()

def evaluate_and_save(y_true, y_pred, y_prob, class_names, prefix, out_dir):
    labels = [c.replace("Potato__", "").replace("_", " ") for c in class_names]

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec  = recall_score(y_true, y_pred, average="macro")
    f1   = f1_score(y_true, y_pred, average="macro")

    print(f"\n{prefix} Accuracy: {acc:.4f}")
    print(f"{prefix} Macro Precision: {prec:.4f}")
    print(f"{prefix} Macro Recall: {rec:.4f}")
    print(f"{prefix} Macro F1: {f1:.4f}")

    report_dict = classification_report(
        y_true, y_pred,
        target_names=labels,
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose().round(4)
    report_df.to_csv(os.path.join(out_dir, f"{prefix.lower()}_classification_report.csv"))

    print(f"\n{prefix} Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

    save_confusion_matrices(y_true, y_pred, class_names, prefix, out_dir)

    if y_prob is not None:
        save_roc_curves(y_true, y_prob, class_names, prefix, out_dir)

        # confidence distribution
        confidence = np.max(y_prob, axis=1)
        correct = (y_true == y_pred)

        plt.figure(figsize=(10, 5))
        plt.hist(confidence[correct], bins=20, alpha=0.7, label="Correct")
        plt.hist(confidence[~correct], bins=20, alpha=0.7, label="Wrong")
        plt.title(f"{prefix} - Confidence Distribution")
        plt.xlabel("Confidence")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix.lower()}_confidence_distribution.png"),
                    dpi=200, bbox_inches="tight")
        plt.show()

    return {
        "Model": prefix,
        "Accuracy": round(acc, 4),
        "Macro_Precision": round(prec, 4),
        "Macro_Recall": round(rec, 4),
        "Macro_F1": round(f1, 4)
    }

# ============================================================
# 12. CNN DIRECT EVALUATION
# ============================================================
test_gen.reset()
cnn_test_prob = best_cnn.predict(test_gen, verbose=1)
cnn_test_pred = np.argmax(cnn_test_prob, axis=1)
y_test = test_gen.classes.copy()

cnn_metrics = evaluate_and_save(
    y_true=y_test,
    y_pred=cnn_test_pred,
    y_prob=cnn_test_prob,
    class_names=CLASS_NAMES,
    prefix="CNN",
    out_dir=OUTPUT_DIR
)

# ============================================================
# 13. EXTRACT CNN FEATURES FOR ML ALGORITHM
# ============================================================
def extract_features(generator, feature_model):
    generator.reset()
    features = feature_model.predict(generator, verbose=1)
    labels = generator.classes.copy()
    filenames = list(generator.filenames)
    return features, labels, filenames

X_train, y_train, train_files = extract_features(train_eval_gen, feature_extractor)
X_val, y_val, val_files       = extract_features(val_gen, feature_extractor)
X_test, y_test, test_files    = extract_features(test_gen, feature_extractor)

print("\nFeature shapes:")
print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)

# ============================================================
# 14. SCALE FEATURES
# ============================================================
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

joblib.dump(scaler, SCALER_PATH)

# ============================================================
# 15. TRAIN SVM (ML ALGORITHM)
# ============================================================
print("\nTraining SVM on CNN deep features...")

min_class_count = min(Counter(y_train).values())
cv_splits = 3 if min_class_count >= 3 else 2

param_grid = {
    "C": [0.1, 1, 10, 50],
    "gamma": ["scale", 0.01, 0.001],
    "kernel": ["rbf"]
}

cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)

grid = GridSearchCV(
    estimator=SVC(probability=True, random_state=SEED),
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_sc, y_train)

best_svm = grid.best_estimator_
print("Best SVM parameters:", grid.best_params_)
print("Best CV score:", grid.best_score_)

grid_results_df = pd.DataFrame(grid.cv_results_).sort_values("rank_test_score")
grid_results_df.to_csv(SVM_GRID_CSV_PATH, index=False)

joblib.dump(best_svm, SVM_MODEL_PATH)

# ============================================================
# 16. EVALUATE SVM
# ============================================================
svm_val_pred  = best_svm.predict(X_val_sc)
svm_test_pred = best_svm.predict(X_test_sc)

svm_val_prob  = best_svm.predict_proba(X_val_sc)
svm_test_prob = best_svm.predict_proba(X_test_sc)

print("\nValidation Accuracy (SVM):", accuracy_score(y_val, svm_val_pred))
print("Test Accuracy (SVM):", accuracy_score(y_test, svm_test_pred))

svm_metrics = evaluate_and_save(
    y_true=y_test,
    y_pred=svm_test_pred,
    y_prob=svm_test_prob,
    class_names=CLASS_NAMES,
    prefix="CNN_SVM",
    out_dir=OUTPUT_DIR
)

# Save test predictions
pred_df = pd.DataFrame({
    "filename": test_files,
    "true_label_index": y_test,
    "pred_label_index": svm_test_pred,
    "true_label": [CLEAN_CLASS_NAMES[i] for i in y_test],
    "pred_label": [CLEAN_CLASS_NAMES[i] for i in svm_test_pred],
    "confidence": np.max(svm_test_prob, axis=1),
    "correct": (y_test == svm_test_pred)
})
pred_df.to_csv(TEST_PRED_CSV_PATH, index=False)

# ============================================================
# 17. PCA VISUALIZATION OF TEST FEATURES
# ============================================================
pca = PCA(n_components=2, random_state=SEED)
X_test_2d = pca.fit_transform(X_test_sc)

plt.figure(figsize=(9, 7))
for class_idx, class_name in enumerate(CLEAN_CLASS_NAMES):
    mask = (y_test == class_idx)
    plt.scatter(
        X_test_2d[mask, 0],
        X_test_2d[mask, 1],
        alpha=0.7,
        label=class_name
    )

plt.title("PCA of CNN Deep Features (Test Set)")
plt.xlabel("PC-1")
plt.ylabel("PC-2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_test_features.png"), dpi=200, bbox_inches="tight")
plt.show()

# ============================================================
# 18. COMPARE CNN vs CNN+SVM
# ============================================================
comparison_df = pd.DataFrame([cnn_metrics, svm_metrics])
comparison_df.to_csv(COMPARISON_CSV_PATH, index=False)

print("\nModel Comparison:")
print(comparison_df)

# ============================================================
# 19. SAVE METADATA
# ============================================================
metadata = {
    "base_dir": BASE_DIR,
    "train_dir": TRAIN_DIR,
    "val_dir": VAL_DIR,
    "test_dir": TEST_DIR,
    "output_dir": OUTPUT_DIR,
    "class_names": CLASS_NAMES,
    "clean_class_names": CLEAN_CLASS_NAMES,
    "num_classes": NUM_CLASSES,
    "img_size": list(IMG_SIZE),
    "batch_size": BATCH_SIZE,
    "seed": SEED,
    "best_svm_params": grid.best_params_,
    "cnn_model_path": CNN_MODEL_PATH,
    "feature_model_path": FEATURE_MODEL_PATH,
    "svm_model_path": SVM_MODEL_PATH,
    "scaler_path": SCALER_PATH
}

with open(METADATA_JSON_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

# ============================================================
# 20. SINGLE IMAGE PREDICTION USING CNN+SVM
# ============================================================
def predict_single_image_hybrid(image_path, feature_model, scaler, svm_model, class_names, img_size=(224, 224)):
    clean_names = [c.replace("Potato__", "").replace("_", " ") for c in class_names]

    img = Image.open(image_path).convert("RGB").resize(img_size)
    img_arr = np.array(img, dtype=np.float32) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    feat = feature_model.predict(img_arr, verbose=0)
    feat_sc = scaler.transform(feat)

    probs = svm_model.predict_proba(feat_sc)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = clean_names[pred_idx]
    confidence = float(probs[pred_idx])

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Prediction: {pred_label}\nConfidence: {confidence:.2%}")

    plt.subplot(1, 2, 2)
    plt.barh(clean_names, probs)
    plt.xlim(0, 1)
    plt.xlabel("Probability")
    plt.title("Class Probabilities")

    plt.tight_layout()
    plt.show()

    return pred_label, confidence, probs

# Example prediction on one test image
sample_image_path = os.path.join(TEST_DIR, CLASS_NAMES[0], os.listdir(os.path.join(TEST_DIR, CLASS_NAMES[0]))[0])
pred_label, confidence, probs = predict_single_image_hybrid(
    sample_image_path,
    feature_extractor,
    scaler,
    best_svm,
    CLASS_NAMES,
    img_size=IMG_SIZE
)

print("\nSample prediction done.")
print("Predicted:", pred_label)
print("Confidence:", confidence)

# ============================================================
# 21. FINAL SUMMARY
# ============================================================
print("\nAll files saved in:", OUTPUT_DIR)
print("\nImportant saved outputs:")
important_files = [
    "eda_distribution.png",
    "eda_train_pie_chart.png",
    "eda_sample_images.png",
    "augmented_samples.png",
    "training_curves.png",
    "cnn_confusion_matrix.png",
    "cnn_roc_curves.png",
    "cnn_svm_confusion_matrix.png",
    "cnn_svm_roc_curves.png",
    "pca_test_features.png",
    "dataset_counts.csv",
    "model_comparison.csv",
    "svm_gridsearch_results.csv",
    "svm_test_predictions.csv",
    "best_cnn_classifier.keras",
    "best_cnn_feature_extractor.keras",
    "svm_on_cnn_features.joblib",
    "feature_scaler.joblib",
    "metadata.json"
]

for f in important_files:
    print(os.path.join(OUTPUT_DIR, f))

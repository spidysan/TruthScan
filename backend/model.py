"""
model.py — SVM + TF-IDF Fake News Detection Pipeline
Handles: preprocessing, training, prediction, and model persistence.
"""

import re
import os
import json
import joblib
import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.calibration import CalibratedClassifierCV

# ─────────────────────────────────────────────
# NLTK resource bootstrap
# ─────────────────────────────────────────────
def _ensure_nltk():
    for resource in ["stopwords", "punkt"]:
        try:
            nltk.data.find(f"corpora/{resource}" if resource == "stopwords" else f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

_ensure_nltk()

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
STATS_PATH = os.path.join(os.path.dirname(__file__), "model_stats.json")

_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))

# ─────────────────────────────────────────────
# Text Preprocessing
# ─────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline:
    1. Lowercase
    2. Remove URLs
    3. Remove special characters & digits
    4. Tokenise
    5. Remove stopwords
    6. Porter stemming
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # Remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)               # Keep letters only
    tokens = text.split()
    tokens = [_stemmer.stem(t) for t in tokens if t not in _stop_words and len(t) > 2]
    return " ".join(tokens)


# ─────────────────────────────────────────────
# Build Pipeline
# ─────────────────────────────────────────────
def _build_pipeline() -> Pipeline:
    """
    TF-IDF (bigrams, 10k features) → LinearSVC wrapped in CalibratedClassifierCV
    so we get probability scores.
    """
    base_svm = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    calibrated_svm = CalibratedClassifierCV(base_svm, cv=3)

    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=preprocess_text,
            max_features=10_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )),
        ("svm", calibrated_svm),
    ])


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def train(df: pd.DataFrame) -> dict:
    """
    Train the SVM pipeline on the provided DataFrame.
    DataFrame must have columns: 'text' (str), 'label' (0=Fake, 1=Real).
    Returns a metrics dict and saves model + stats to disk.
    """
    # Combine title + text if both columns present
    if "title" in df.columns and "text" in df.columns:
        df = df.copy()
        df["content"] = df["title"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str)
    elif "text" in df.columns:
        df = df.copy()
        df["content"] = df["text"].fillna("").astype(str)
    else:
        raise ValueError("DataFrame must contain a 'text' column.")

    df = df.dropna(subset=["content", "label"])
    df["label"] = df["label"].astype(int)

    X = df["content"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=["Fake", "Real"], output_dict=True)

    stats = {
        "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred, average="weighted") * 100, 2),
        "recall":    round(recall_score(y_test, y_pred, average="weighted") * 100, 2),
        "f1_score":  round(f1_score(y_test, y_pred, average="weighted") * 100, 2),
        "confusion_matrix": cm,
        "classification_report": report,
        "train_samples": len(X_train),
        "test_samples":  len(X_test),
        "total_samples": len(df),
    }

    joblib.dump(pipeline, MODEL_PATH)
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
def predict(text: str) -> dict:
    """
    Predict whether the given news text is Real (1) or Fake (0).
    Returns label name, confidence %, and raw probabilities.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not trained yet. Call /train first.")

    pipeline = joblib.load(MODEL_PATH)
    proba = pipeline.predict_proba([text])[0]   # [P(fake), P(real)]
    label_idx = int(np.argmax(proba))
    label_name = "Real" if label_idx == 1 else "Fake"
    confidence = round(float(proba[label_idx]) * 100, 2)

    return {
        "label": label_name,
        "label_id": label_idx,
        "confidence": confidence,
        "prob_fake": round(float(proba[0]) * 100, 2),
        "prob_real": round(float(proba[1]) * 100, 2),
    }


# ─────────────────────────────────────────────
# Model Stats
# ─────────────────────────────────────────────
def get_model_stats() -> dict:
    """Load persisted model stats from disk."""
    if not os.path.exists(STATS_PATH):
        return None
    with open(STATS_PATH, "r") as f:
        return json.load(f)


def is_trained() -> bool:
    return os.path.exists(MODEL_PATH)

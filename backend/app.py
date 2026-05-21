"""
app.py — FastAPI REST API for Fake News Detection
Endpoints: /train, /predict, /stats, /health
"""

import os
import sys
import time

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import feedparser
from bs4 import BeautifulSoup

# Add parent dir so `backend.model` resolves from project root
sys.path.insert(0, os.path.dirname(__file__))
import model as ml_model

# ─────────────────────────────────────────────
# Dataset paths — Kaggle True/Fake CSVs
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE_CSV   = os.path.join(BASE_DIR, "Fake.csv")
TRUE_CSV   = os.path.join(BASE_DIR, "True.csv")

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Fake News Detector API",
    description="SVM + TF-IDF based fake news classification",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    label_id: int
    confidence: float
    prob_fake: float
    prob_real: float
    processing_time_ms: float

class TrainResponse(BaseModel):
    message: str
    dataset_used: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    train_samples: int
    test_samples: int
    total_samples: int
    training_time_s: float

class LiveNewsItem(BaseModel):
    title: str
    link: str
    published: str
    source: str
    label: str
    confidence: float

class LiveNewsResponse(BaseModel):
    articles: list[LiveNewsItem]
    count: int


# ─────────────────────────────────────────────
# Helper: load dataset
# ─────────────────────────────────────────────
def _load_dataset():
    """
    Load and merge Fake.csv (label=0) and True.csv (label=1)
    from the project root directory.
    Columns expected: title, text, subject, date
    """
    if not os.path.exists(FAKE_CSV):
        raise FileNotFoundError(f"Fake.csv not found at: {FAKE_CSV}")
    if not os.path.exists(TRUE_CSV):
        raise FileNotFoundError(f"True.csv not found at: {TRUE_CSV}")

    fake_df = pd.read_csv(FAKE_CSV)
    true_df = pd.read_csv(TRUE_CSV)

    # Assign labels: 0 = Fake, 1 = Real
    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    dataset_name = f"Fake.csv + True.csv ({len(fake_df):,} fake · {len(true_df):,} real)"
    return df, dataset_name


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_trained": ml_model.is_trained(),
    }


@app.post("/train", response_model=TrainResponse)
def train():
    """Load dataset and train the SVM model."""
    try:
        t0 = time.time()
        df, dataset_name = _load_dataset()
        stats = ml_model.train(df)
        elapsed = round(time.time() - t0, 2)
        return TrainResponse(
            message="Model trained successfully!",
            dataset_used=dataset_name,
            accuracy=stats["accuracy"],
            precision=stats["precision"],
            recall=stats["recall"],
            f1_score=stats["f1_score"],
            train_samples=stats["train_samples"],
            test_samples=stats["test_samples"],
            total_samples=stats["total_samples"],
            training_time_s=elapsed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Classify a news article as Real or Fake."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if not ml_model.is_trained():
        raise HTTPException(status_code=503, detail="Model not trained. Call POST /train first.")

    try:
        t0 = time.time()
        result = ml_model.predict(req.text)
        elapsed_ms = round((time.time() - t0) * 1000, 2)
        return PredictResponse(**result, processing_time_ms=elapsed_ms)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def stats():
    """Return model performance statistics."""
    s = ml_model.get_model_stats()
    if s is None:
        raise HTTPException(status_code=404, detail="No stats found. Train the model first.")
    return s


@app.get("/live-news", response_model=LiveNewsResponse)
def live_news():
    """Fetch latest news from RSS feed and predict fake/real."""
    if not ml_model.is_trained():
        raise HTTPException(status_code=503, detail="Model not trained. Call POST /train first.")

    rss_url = "http://feeds.bbci.co.uk/news/rss.xml"
    try:
        feed = feedparser.parse(rss_url)
        articles = []
        for entry in feed.entries[:10]:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            link = entry.get("link", "")
            published = entry.get("published", "")
            
            # Clean HTML from summary
            soup = BeautifulSoup(summary, "html.parser")
            clean_summary = soup.get_text()
            
            text_to_predict = f"{title}. {clean_summary}"
            
            # Predict
            pred = ml_model.predict(text_to_predict)
            
            articles.append(LiveNewsItem(
                title=title,
                link=link,
                published=published,
                source="BBC News",
                label=pred["label"],
                confidence=pred["confidence"]
            ))
            
        return LiveNewsResponse(articles=articles, count=len(articles))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch live news: {str(e)}")
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

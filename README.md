# TruthScan AI — Fake News Detection System
## SVM + TF-IDF | Python + FastAPI + Vanilla JS

---

## 📁 Project Structure

```
svm fake news detector/
├── backend/
│   ├── app.py              ← FastAPI REST server
│   ├── model.py            ← SVM + TF-IDF pipeline
│   ├── requirements.txt    ← Python dependencies
│   ├── model.pkl           ← Saved model (auto-generated after training)
│   └── model_stats.json    ← Performance metrics (auto-generated)
├── data/
│   └── generate_dataset.py ← (not needed — real data is used)
├── Fake.csv                ← Kaggle fake news dataset (23,481 articles)
├── True.csv                ← Kaggle real news dataset (21,417 articles)
├── frontend/
│   ├── index.html          ← Single-page application
│   ├── style.css           ← Premium dark theme
│   └── app.js              ← Frontend logic
├── README.md               ← This file
└── changes.txt             ← Project build log
```

---

## 🚀 Setup & Run

### Step 1: Install Python dependencies
```bash
cd "svm fake news detector"
pip install -r backend/requirements.txt
```

### Step 2: Start the backend API server
```bash
c
python -m uvicorn backend.app:app --reload --port 8000
```
The API will be available at `http://127.0.0.1:8000`

### Step 3: Open the frontend
Open `frontend/index.html` in your browser.

### Step 4: Train the model
Click **"Train Model"** in the web UI (or POST to `/train`).
The model trains automatically using the synthetic dataset.

### Step 5: Analyze news
Paste any news article and click **"Analyze Article"**.

---

## 📊 Expected Performance

| Dataset                    | Samples | Accuracy | F1-Score |
|----------------------------|---------|----------|----------|
| Fake.csv + True.csv (Kaggle) | ~44,898 | ~98–99%  | ~0.98    |

---

## 📰 A Note on Live News & "Truth"

The Live News Feed fetches real-time articles from respected news outlets (like the BBC) to demonstrate the model's capabilities on completely unseen data.

**Important context:** The AI model is "blind" to the source of the news. When it labels an article as "Real," it is not verifying the objective truth of the event. Instead, it is detecting that the *linguistic patterns, formal grammar, and objective tone* of the article match the journalistic standards of the "Real" news articles it was trained on. 

Fake news often uses sensationalist phrasing, excessive capitalization, and informal grammar. The Live News feed is a fun, live test to see if our model can recognize professional journalistic writing in the wild!

---

## 🔌 API Endpoints

| Method | Endpoint   | Description                    |
|--------|------------|--------------------------------|
| GET    | /health    | Server health + training status |
| POST   | /train     | Train the SVM model            |
| POST   | /predict   | Classify a news article        |
| GET    | /stats     | Get model performance metrics  |

### Example: Predict
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists at MIT published new research on climate change."}'
```

---

## 🧠 How It Works

1. **Text Preprocessing**: lowercase → remove URLs/punctuation → remove stopwords → Porter stemming
2. **TF-IDF Vectorization**: 10,000 features, bigrams (1,2), sublinear TF scaling
3. **LinearSVC**: Linear kernel SVM, wrapped in CalibratedClassifierCV for probability output
4. **Prediction**: Returns label (Real/Fake), confidence %, probabilities, and processing time

---

## 📦 Dependencies

- `fastapi` — REST API framework
- `uvicorn` — ASGI server
- `scikit-learn` — SVM, TF-IDF, metrics
- `pandas` — data handling
- `nltk` — stopwords, stemming
- `joblib` — model persistence

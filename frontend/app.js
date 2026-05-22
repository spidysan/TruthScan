/**
 * app.js — TruthScan AI Frontend Logic
 * Connects to FastAPI backend for train/predict/stats and Live News.
 */

const API = "";

// ─── DOM References ───────────────────────────────────────────
const newsInput      = document.getElementById("newsInput");
const analyzeBtn     = document.getElementById("analyzeBtn");
const clearBtn       = document.getElementById("clearBtn");
const exampleBtn     = document.getElementById("exampleBtn");
const charCount      = document.getElementById("charCount");
const resultCard     = document.getElementById("resultCard");
const verdictBadge   = document.getElementById("verdictBadge");
const verdictLabel   = document.getElementById("verdictLabel");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceVal  = document.getElementById("confidenceValue");
const probFake       = document.getElementById("probFake");
const probReal       = document.getElementById("probReal");
const procTime       = document.getElementById("procTime");
const trainBtn       = document.getElementById("trainBtn");
const trainBtnText   = document.getElementById("trainBtnText");
const statusDot      = document.getElementById("statusDot");
const statsNotTrained= document.getElementById("statsNotTrained");
const statsContent   = document.getElementById("statsContent");
const toast          = document.getElementById("toast");

const fetchLiveBtn   = document.getElementById("fetchLiveBtn");
const liveNewsGrid   = document.getElementById("liveNewsGrid");

// Stats elements
const valAccuracy  = document.getElementById("valAccuracy");
const valPrecision = document.getElementById("valPrecision");
const valRecall    = document.getElementById("valRecall");
const valF1        = document.getElementById("valF1");
const barAccuracy  = document.getElementById("barAccuracy");
const barPrecision = document.getElementById("barPrecision");
const barRecall    = document.getElementById("barRecall");
const barF1        = document.getElementById("barF1");
const datasetInfo  = document.getElementById("datasetInfo");
const cmTN         = document.getElementById("cmTN");
const cmFP         = document.getElementById("cmFP");
const cmFN         = document.getElementById("cmFN");
const cmTP         = document.getElementById("cmTP");

// ─── State ────────────────────────────────────────────────────
let isModelTrained = false;

// ─── Example Articles ─────────────────────────────────────────
const EXAMPLES = [
  {
    label: "real",
    text: `The World Health Organization published updated guidelines on Monday recommending that adults aged 18 to 64 engage in at least 150 minutes of moderate-intensity aerobic physical activity throughout the week. The guidelines, which were developed after reviewing over 400 scientific studies, represent the most comprehensive update to the organization's physical activity recommendations in a decade. Health officials emphasized that even small amounts of physical activity can have significant benefits for cardiovascular health, mental well-being, and longevity.`
  },
  {
    label: "fake",
    text: `BREAKING!!! GOVERNMENT SCIENTIST ADMITS 5G TOWERS ARE SECRETLY CONTROLLING YOUR MIND!!! A whistleblower who worked for a top-secret government lab has come forward to EXPOSE the TRUTH about 5G towers. According to this brave insider, the electromagnetic signals emitted by these towers contain hidden frequencies designed to suppress critical thinking and make citizens more obedient. BIG PHARMA doesn't want you to know this!!! Share this before it gets DELETED. DO YOUR OWN RESEARCH!!!`
  }
];
let exampleIndex = 0;

// ─── Toast ────────────────────────────────────────────────────
let toastTimer;
function showToast(msg, type = "info") {
  toast.textContent = msg;
  toast.className = `toast ${type} show`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.className = "toast", 3000);
}

// ─── Health Check ─────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API}/health`, { signal: AbortSignal.timeout(3000) });
    if (!res.ok) throw new Error();
    const data = await res.json();
    statusDot.className = "status-dot online";
    isModelTrained = data.model_trained;
    if (isModelTrained) {
      analyzeBtn.disabled = newsInput.value.trim().length < 10;
      loadStats();
    }
  } catch {
    statusDot.className = "status-dot offline";
    showToast("⚠️ Cannot reach backend. Start the server first.", "error");
  }
}

// ─── Training ─────────────────────────────────────────────────
async function trainModel() {
  trainBtnText.innerHTML = `Training…`;
  trainBtn.disabled = true;
  showToast("🔄 Training model, please wait…", "info");

  try {
    const res = await fetch(`${API}/train`, { method: "POST", signal: AbortSignal.timeout(120000) });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Training failed");
    }
    const data = await res.json();
    isModelTrained = true;
    analyzeBtn.disabled = newsInput.value.trim().length < 10;
    showToast(`✅ Model trained! Accuracy: ${data.accuracy}%`, "success");
    showStats(data);
  } catch (e) {
    showToast(`❌ ${e.message}`, "error");
  } finally {
    trainBtnText.textContent = "TRAIN MODEL";
    trainBtn.disabled = false;
  }
}

// ─── Load Stats ───────────────────────────────────────────────
async function loadStats() {
  try {
    const res = await fetch(`${API}/stats`);
    if (!res.ok) return;
    const data = await res.json();
    showStats(data);
  } catch {}
}

function showStats(data) {
  statsNotTrained.classList.add("hidden");
  statsContent.classList.remove("hidden");

  animateValue(valAccuracy,  barAccuracy,  data.accuracy);
  animateValue(valPrecision, barPrecision, data.precision);
  animateValue(valRecall,    barRecall,    data.recall);
  animateValue(valF1,        barF1,        data.f1_score);

  if (data.dataset_used) {
    datasetInfo.innerHTML = `
      📂 Dataset: <strong>${data.dataset_used || "N/A"}</strong><br>
      🔢 Total:   ${(data.total_samples || 0).toLocaleString()} samples<br>
      🏋️ Train:  ${(data.train_samples || 0).toLocaleString()} · 
      🧪 Test:   ${(data.test_samples || 0).toLocaleString()}
    `;
  }

  if (data.confusion_matrix) {
    const [[tn, fp], [fn, tp]] = data.confusion_matrix;
    cmTN.textContent = tn.toLocaleString();
    cmFP.textContent = fp.toLocaleString();
    cmFN.textContent = fn.toLocaleString();
    cmTP.textContent = tp.toLocaleString();
  }
}

function animateValue(el, barEl, value) {
  const target = parseFloat(value);
  const start = 0;
  const duration = 1000;
  const step = (timestamp, startTime) => {
    const progress = Math.min((timestamp - startTime) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = start + (target - start) * eased;
    el.textContent = current.toFixed(1) + "%";
    barEl.style.width = current + "%";
    if (progress < 1) requestAnimationFrame(ts => step(ts, startTime));
  };
  requestAnimationFrame(ts => step(ts, ts));
}

// ─── Prediction ───────────────────────────────────────────────
async function analyzeArticle() {
  const text = newsInput.value.trim();
  if (text.length < 10) {
    showToast("Please enter a longer article (at least 10 characters).", "error");
    return;
  }
  if (!isModelTrained) {
    showToast("Train the model first!", "error");
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = `<span class="btn-text">Analyzing…</span>`;
  resultCard.classList.add("hidden");

  try {
    const res = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Prediction failed");
    }

    const data = await res.json();
    displayResult(data, text);
  } catch (e) {
    showToast(`❌ ${e.message}`, "error");
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = `<span class="btn-text">Analyze Article</span>`;
  }
}

function displayResult(data, text) {
  const isFake = data.label === "Fake";
  const cls    = isFake ? "fake" : "real";

  verdictBadge.textContent = isFake ? "FAKE" : "REAL";
  verdictBadge.className   = `verdict-badge ${cls}`;
  verdictLabel.textContent = isFake ? "Fake News Detected" : "Real News";
  verdictLabel.className   = `verdict-label ${cls}`;

  confidenceFill.className = `confidence-bar-fill ${cls}`;
  confidenceFill.style.width = "0%";
  setTimeout(() => confidenceFill.style.width = data.confidence + "%", 50);
  confidenceVal.textContent = data.confidence + "%";

  probFake.textContent = data.prob_fake + "%";
  probReal.textContent = data.prob_real + "%";
  procTime.textContent = data.processing_time_ms + "ms";

  resultCard.classList.remove("hidden");
  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ─── Live News Feed ───────────────────────────────────────────
async function fetchLiveNews() {
  if (!isModelTrained) {
    showToast("Train the model first to fetch live news!", "error");
    return;
  }

  fetchLiveBtn.disabled = true;
  fetchLiveBtn.textContent = "FETCHING...";
  liveNewsGrid.innerHTML = `<p class="history-empty">Fetching latest news from BBC...</p>`;

  try {
    const res = await fetch(`${API}/live-news`);
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to fetch live news");
    }
    const data = await res.json();
    
    if (data.articles.length === 0) {
      liveNewsGrid.innerHTML = `<p class="history-empty">No news found right now.</p>`;
      return;
    }

    liveNewsGrid.innerHTML = data.articles.map(item => {
      const isFake = item.label === "Fake";
      const cls = isFake ? "fake" : "real";
      const date = new Date(item.published).toLocaleString();
      
      return `
        <div class="news-card">
          <div class="news-card-meta">
            <span class="news-card-source">${item.source}</span>
            <span>${date}</span>
          </div>
          <h3 class="news-card-title">
            <a href="${item.link}" target="_blank" rel="noopener noreferrer">${item.title}</a>
          </h3>
          <div class="news-card-footer">
            <span class="news-verdict ${cls}">${item.label.toUpperCase()}</span>
            <span class="news-conf">${item.confidence.toFixed(1)}% confidence</span>
          </div>
        </div>
      `;
    }).join("");
    
  } catch (e) {
    liveNewsGrid.innerHTML = `<p class="history-empty" style="color:var(--fake-color)">Error: ${e.message}</p>`;
  } finally {
    fetchLiveBtn.disabled = false;
    fetchLiveBtn.textContent = "REFRESH";
  }
}

// ─── Textarea helpers ─────────────────────────────────────────
newsInput.addEventListener("input", () => {
  const len = newsInput.value.trim().length;
  charCount.textContent = newsInput.value.length;
  analyzeBtn.disabled = !isModelTrained || len < 10;
});

clearBtn.addEventListener("click", () => {
  newsInput.value = "";
  charCount.textContent = "0";
  resultCard.classList.add("hidden");
  analyzeBtn.disabled = true;
});

exampleBtn.addEventListener("click", () => {
  const ex = EXAMPLES[exampleIndex % EXAMPLES.length];
  exampleIndex++;
  newsInput.value = ex.text;
  charCount.textContent = ex.text.length;
  analyzeBtn.disabled = !isModelTrained;
  showToast(`📄 Example loaded (${ex.label} news sample)`, "info");
});

analyzeBtn.addEventListener("click", analyzeArticle);
trainBtn.addEventListener("click", trainModel);
fetchLiveBtn.addEventListener("click", fetchLiveNews);

// ─── Bootstrap ───────────────────────────────────────────────
checkHealth();
setInterval(checkHealth, 30000);

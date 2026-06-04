const manualForm = document.getElementById("manual-form");
const predictBtn = document.getElementById("predict-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const cardsEl = document.getElementById("cards");
const useLstmEl = document.getElementById("use-lstm");
const csvFileEl = document.getElementById("csv-file");

let featureNames = [];
let activeTab = "manual";

function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("error", isError);
}

function showResults(data) {
  const items = [
    ["Random Forest", data.random_forest],
    ["XGBoost", data.xgboost],
  ];
  if (data.lstm != null) items.push(["LSTM", data.lstm]);
  items.push(["Ensemble", data.ensemble, true]);

  cardsEl.innerHTML = items
    .map(([label, value, hero]) => {
      const cls = hero ? "card hero" : "card";
      return `<div class="${cls}"><div class="label">${label}</div><div class="value">${Number(value).toFixed(6)}</div></div>`;
    })
    .join("");

  resultsEl.classList.remove("hidden");
}

async function loadFeatures() {
  const res = await fetch("/api/features");
  const data = await res.json();
  featureNames = data.features;

  manualForm.innerHTML = featureNames
    .map(
      (name) =>
        `<label>${name.replace(/_/g, " ")}<input type="number" step="any" name="${name}" value="0" /></label>`
    )
    .join("");
}

function getManualFeatures() {
  const features = {};
  featureNames.forEach((name) => {
    features[name] = parseFloat(manualForm.querySelector(`[name="${name}"]`).value) || 0;
  });
  return features;
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",").map((h) => h.trim());
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const vals = lines[i].split(",").map((v) => v.trim());
    const row = {};
    headers.forEach((h, j) => {
      row[h] = parseFloat(vals[j]);
    });
    rows.push(row);
  }
  return rows;
}

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    activeTab = btn.dataset.tab;
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    document.getElementById(`panel-${activeTab}`).classList.add("active");
  });
});

predictBtn.addEventListener("click", async () => {
  predictBtn.disabled = true;
  resultsEl.classList.add("hidden");
  setStatus("Running models…");

  const useLstm = useLstmEl.checked;

  try {
    let res;

    if (activeTab === "csv") {
      const file = csvFileEl.files[0];
      if (!file) {
        setStatus("Choose a CSV file first.", true);
        return;
      }
      const form = new FormData();
      form.append("file", file);
      form.append("use_lstm", useLstm);
      res = await fetch("/api/predict/csv", { method: "POST", body: form });
    } else {
      if (useLstm) {
        setStatus("LSTM needs CSV upload with 30+ rows.", true);
        return;
      }
      const body = { features: getManualFeatures(), use_lstm: false };
      res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    }

    const data = await res.json();
    if (!res.ok) {
      const msg = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
      setStatus(msg || "Prediction failed.", true);
      return;
    }

    setStatus(data.rows_loaded ? `${data.rows_loaded} rows loaded.` : "Done.");
    showResults(data);
  } catch (err) {
    setStatus(err.message || "Network error.", true);
  } finally {
    predictBtn.disabled = false;
  }
});

loadFeatures().catch(() => setStatus("Could not load features from server.", true));

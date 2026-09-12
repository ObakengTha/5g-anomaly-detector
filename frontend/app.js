const API = ""; // same origin - backend serves this frontend directly

// ---------- tab switching ----------
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll("[data-panel]").forEach((p) => p.classList.add("hidden"));
    tab.classList.add("active");
    document.getElementById(`panel-${tab.dataset.tab}`).classList.remove("hidden");
  });
});

// ---------- health check ----------
async function checkHealth() {
  const dot = document.getElementById("statusDot");
  const text = document.getElementById("statusText");
  try {
    const res = await fetch(`${API}/api/health`);
    const data = await res.json();
    if (data.model_loaded) {
      dot.classList.add("ok");
      text.textContent = "Model loaded";
    } else {
      throw new Error("model not loaded");
    }
  } catch (e) {
    dot.classList.add("error");
    text.textContent = "Backend unreachable";
  }
}

// ---------- build the quick-test form from /api/schema ----------
async function buildQuickForm() {
  const form = document.getElementById("quickForm");
  try {
    const res = await fetch(`${API}/api/schema`);
    const schema = await res.json();
    const fields = schema.key_fields.length ? schema.key_fields
      : schema.numeric_cols.slice(0, 6).concat(schema.categorical_cols.slice(0, 2));

    form.innerHTML = "";
    fields.forEach((name) => {
      const wrap = document.createElement("div");
      wrap.className = "field";
      const label = document.createElement("label");
      label.textContent = name;
      label.setAttribute("for", `f_${name}`);
      wrap.appendChild(label);

      if (name === "Protocol") {
        const select = document.createElement("select");
        select.id = `f_${name}`;
        select.name = name;
        ["TCP", "UDP", "ICMP"].forEach((opt) => {
          const o = document.createElement("option");
          o.value = opt;
          o.textContent = opt;
          select.appendChild(o);
        });
        wrap.appendChild(select);
      } else {
        const input = document.createElement("input");
        input.type = "number";
        input.step = "any";
        input.id = `f_${name}`;
        input.name = name;
        input.placeholder = "0";
        wrap.appendChild(input);
      }
      form.appendChild(wrap);
    });
  } catch (e) {
    form.innerHTML = `<p style="color:var(--anomaly)">Could not load form fields - is the backend running?</p>`;
  }
}

// ---------- render one explanation method's bar list into a container ----------
function renderMethodBlock(container, title, items) {
  const template = document.getElementById("methodBlockTemplate");
  const block = template.content.cloneNode(true);
  block.querySelector(".exp-title").textContent = title;
  const rowsContainer = block.querySelector(".method-rows");

  if (!items || !items.length) {
    const empty = document.createElement("p");
    empty.className = "method-empty";
    empty.textContent = "No data.";
    rowsContainer.appendChild(empty);
  } else {
    const maxAbs = Math.max(...items.map((it) => Math.abs(it.weight)), 0.0001);
    const barTemplate = document.getElementById("explanationBarTemplate");
    items.forEach((item) => {
      const row = barTemplate.content.cloneNode(true);
      row.querySelector(".exp-token").textContent = item.token;
      row.querySelector(".exp-token").title = item.token;
      const pct = (Math.abs(item.weight) / maxAbs) * 100;
      const fill = row.querySelector(".exp-bar-fill");
      fill.style.width = `${pct}%`;
      if (item.weight < 0) fill.classList.add("negative");
      row.querySelector(".exp-weight").textContent = item.weight.toFixed(4);
      rowsContainer.appendChild(row);
    });
  }
  container.appendChild(block);
}

// ---------- render a prediction result (verdict + attack type + explanations) ----------
function renderResult(container, result, { showFullExplainButton = false, onFullExplain = null } = {}) {
  const isAnomaly = result.prediction === "ANOMALY";
  container.innerHTML = "";

  const verdict = document.createElement("div");
  verdict.className = `verdict ${isAnomaly ? "anomaly" : "benign"}`;
  let verdictText = result.prediction;
  if (isAnomaly && result.attack_type) verdictText += ` — ${result.attack_type}`;
  verdict.innerHTML = `<span>${isAnomaly ? "⚠" : "✓"}</span><span>${verdictText}</span>`;
  container.appendChild(verdict);

  const metrics = document.createElement("div");
  metrics.className = "metric-row";
  metrics.innerHTML = `
    <div class="metric">
      <span class="metric-value">${(result.confidence * 100).toFixed(1)}%</span>
      <span class="metric-label">confidence</span>
    </div>
    <div class="metric">
      <span class="metric-value">${(result.anomaly_probability * 100).toFixed(1)}%</span>
      <span class="metric-label">anomaly probability</span>
    </div>
  `;
  container.appendChild(metrics);

  if (isAnomaly && result.attack_type_probabilities) {
    const atTitle = document.createElement("div");
    atTitle.className = "exp-title";
    atTitle.textContent = "Attack type probabilities";
    container.appendChild(atTitle);

    const sorted = Object.entries(result.attack_type_probabilities).sort((a, b) => b[1] - a[1]);
    const maxP = Math.max(...sorted.map(([, p]) => p), 0.0001);
    const barTemplate = document.getElementById("explanationBarTemplate");
    sorted.forEach(([name, p]) => {
      const row = barTemplate.content.cloneNode(true);
      row.querySelector(".exp-token").textContent = name;
      row.querySelector(".exp-bar-fill").style.width = `${(p / maxP) * 100}%`;
      row.querySelector(".exp-weight").textContent = `${(p * 100).toFixed(1)}%`;
      container.appendChild(row);
    });
  }

  const detailsTitle = document.createElement("div");
  detailsTitle.className = "exp-title";
  detailsTitle.style.marginTop = "16px";
  detailsTitle.textContent = "Why - explanation methods";
  container.appendChild(detailsTitle);

  const methodsWrap = document.createElement("div");
  methodsWrap.className = "methods-grid";
  container.appendChild(methodsWrap);

  renderMethodBlock(methodsWrap, "Attention", result.explanation.attention);

  if (result.explanation.shap) {
    renderMethodBlock(methodsWrap, "SHAP", result.explanation.shap);
  }
  if (result.explanation.lime) {
    renderMethodBlock(methodsWrap, "LIME", result.explanation.lime);
  }

  if (showFullExplainButton && !result.explanation.shap) {
    const btn = document.createElement("button");
    btn.className = "btn btn-secondary";
    btn.textContent = "Full explain (SHAP + LIME)";
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      btn.textContent = "Computing…";
      await onFullExplain(btn);
    });
    container.appendChild(btn);
  }
}

// ---------- quick test submit ----------
document.getElementById("quickForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = document.getElementById("analyzeBtn");
  const resultBox = document.getElementById("quickResult");
  btn.disabled = true;
  btn.textContent = "Analyzing…";

  const formData = new FormData(e.target);
  const fields = {};
  for (const [key, value] of formData.entries()) {
    fields[key] = value === "" ? 0 : (isNaN(value) ? value : Number(value));
  }

  try {
    const res = await fetch(`${API}/api/predict/manual`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fields }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || "Prediction failed");
    const result = await res.json();
    renderResult(resultBox, result);
  } catch (err) {
    resultBox.innerHTML = `<div class="empty-state"><span class="empty-glyph">✕</span><p style="color:var(--anomaly)">${err.message}</p></div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze log";
  }
});

// ---------- batch CSV ----------
const dropzone = document.getElementById("dropzone");
const csvInput = document.getElementById("csvInput");
const batchBtn = document.getElementById("batchAnalyzeBtn");
let selectedFile = null;

dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
csvInput.addEventListener("change", () => {
  if (csvInput.files.length) handleFile(csvInput.files[0]);
});

function handleFile(file) {
  selectedFile = file;
  document.getElementById("dropzoneLabel").textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  batchBtn.disabled = false;
}

batchBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  batchBtn.disabled = true;
  batchBtn.textContent = "Analyzing…";

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch(`${API}/api/predict/csv`, { method: "POST", body: formData });
    if (!res.ok) throw new Error((await res.json()).detail || "Batch prediction failed");
    const data = await res.json();
    renderBatch(data);
  } catch (err) {
    alert(err.message);
  } finally {
    batchBtn.disabled = false;
    batchBtn.textContent = "Analyze batch";
  }
});

function renderBatch(data) {
  document.getElementById("batchSummaryCard").classList.remove("hidden");
  document.getElementById("summaryTotal").textContent = data.n_rows;
  document.getElementById("summaryAnomalies").textContent = data.n_anomalies;
  document.getElementById("summaryRate").textContent =
    data.n_rows ? `${((data.n_anomalies / data.n_rows) * 100).toFixed(1)}%` : "0%";

  const tbody = document.getElementById("resultsBody");
  tbody.innerHTML = "";

  data.results.forEach((r) => {
    const tr = document.createElement("tr");
    if (r.error) {
      tr.innerHTML = `<td>${r.row}</td><td colspan="5" style="color:var(--anomaly)">${r.error}</td>`;
      tbody.appendChild(tr);
      return;
    }
    const isAnomaly = r.prediction === "ANOMALY";
    const topToken = r.explanation.attention[0] ? r.explanation.attention[0].token : "-";
    tr.innerHTML = `
      <td>${r.row}</td>
      <td><span class="badge ${isAnomaly ? "anomaly" : "benign"}">${r.prediction}</span></td>
      <td>${r.attack_type || "-"}</td>
      <td>${(r.confidence * 100).toFixed(1)}%</td>
      <td>${topToken}</td>
      <td><button class="row-detail-btn" data-row="${r.row}">Details</button></td>
    `;
    tbody.appendChild(tr);

    const detailTr = document.createElement("tr");
    detailTr.classList.add("hidden");
    detailTr.id = `detail-${r.row}`;
    const detailTd = document.createElement("td");
    detailTd.colSpan = 6;
    detailTd.style.background = "var(--panel-raised)";
    detailTr.appendChild(detailTd);
    tbody.appendChild(detailTr);

    tr.querySelector(".row-detail-btn").addEventListener("click", () => {
      const dRow = document.getElementById(`detail-${r.row}`);
      dRow.classList.toggle("hidden");
      if (!dRow.classList.contains("hidden") && !detailTd.dataset.rendered) {
        renderResult(detailTd, r, {
          showFullExplainButton: true,
          onFullExplain: async (btn) => {
            try {
              const res = await fetch(`${API}/api/explain/full`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ fields: r.fields }),
              });
              if (!res.ok) throw new Error((await res.json()).detail || "Full explain failed");
              const fullResult = await res.json();
              renderResult(detailTd, fullResult);
              detailTd.dataset.rendered = "1";
            } catch (err) {
              btn.textContent = "Failed - retry";
              btn.disabled = false;
            }
          },
        });
        detailTd.dataset.rendered = "1";
      }
    });
  });
}

// ---------- init ----------
checkHealth();
buildQuickForm();

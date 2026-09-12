# 5G Anomaly Detector — Web Interface (v2: attack typing + SHAP/LIME)

A local web app that loads your trained models and classifies network log
records as BENIGN or ANOMALY. If anomalous, it also identifies **which kind
of attack** it most resembles, and explains the decision using **attention,
SHAP, and LIME** together.

```
anomaly_web_app/
├── backend/
│   ├── main.py              FastAPI app: endpoints + startup
│   ├── model.py              Model architecture (must match the notebook exactly)
│   ├── preprocessing.py      Tokenisation, mirrors the notebook's pipeline
│   ├── requirements.txt
│   └── artifacts/            Put your exported model files here (see below)
└── frontend/
    ├── index.html
    ├── style.css
    └── app.js
```

## 1. Get the model artifacts

In Colab, run `Honours_Research.ipynb` (or `Honours_Research_Synthetic.ipynb`)
through **Section 13: Model Export** (this comes right after the new
**Section 12: Multi-Class Attack Type Classification**, which trains the
attack-typing model on top of the binary detector you already had).

That produces **six** files in your Colab working directory:

- `transformer_model.pt` — the binary BENIGN/ANOMALY detector
- `multiclass_model.pt` — the attack-type classifier (only used when the
  binary model says ANOMALY)
- `vocab.json`
- `inference_config.json`
- `attack_label_map.json` — maps the multiclass model's output index to an
  attack type name
- `background_samples.json` — a small sample of training sequences that
  SHAP's `KernelExplainer` needs as a reference distribution; the web app
  has no other way to get this once it's running standalone

Download all six and place them in `backend/artifacts/`.

## 2. Set up the backend (VS Code)

```bash
cd backend
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

(Windows PowerShell only, if `venv\Scripts\activate` is blocked: run
`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once,
or use `venv\Scripts\activate.bat` instead.)

## 3. Run it

```bash
uvicorn main:app --reload --port 8000
```

Open **http://127.0.0.1:8000**. The status dot should turn green ("Model
loaded"). If it's red, check the terminal — almost always a missing/misnamed
file in `backend/artifacts/`.

## Using it

**Quick Test tab** — fill in the fields the model's attention analysis
identified as informative. Submitting computes **all three** explanation
methods immediately (a few seconds — acceptable for a one-off demo, see
the performance note below). If flagged ANOMALY, you'll also see the
predicted attack type and the full probability breakdown across all known
attack types.

**Batch CSV tab** — upload up to 500 rows at once. For speed, batch results
only compute attention (one cheap forward pass per row) plus the attack
type. Click **"Details"** on any row, then **"Full explain (SHAP + LIME)"**
to compute the slower explanations for that specific row on demand.

## Why attention is fast but SHAP/LIME are on-demand only

Attention needs exactly one forward pass per prediction. SHAP's
`KernelExplainer` needs ~50 extra forward passes per explanation, and LIME
needs ~200. For a single Quick Test that's a few seconds — fine. For a
500-row batch, computing SHAP+LIME for every row up front would mean over
100,000 extra forward passes, which would make the app unusably slow. That's
why batch mode computes them lazily, one row at a time, only when you
actually click for the detail.

## What "attack type" means here, and its limits

The attack-type model is trained **only on the anomalous rows** from your
dataset — it never sees BENIGN examples, and it's a completely separate
model from the binary detector (same architecture, different classification
head). Two things worth knowing:

- **It only runs at all if the binary model already said ANOMALY.** A
  BENIGN verdict never gets an attack-type label — there isn't one to give.
- **Rare attack types will have unreliable attack-type predictions**, even
  if the binary BENIGN/ANOMALY call is solid. Some attack categories (e.g.
  Heartbleed, SQL Injection) may have very few training examples; the
  notebook's Section 12 warns you at training time if a class had too few
  rows to stratify properly. Don't over-interpret a specific attack-type
  label for a rare category without checking that model's classification
  report in the notebook first.

## Known limitations to keep in mind

- **Bin edges are frozen at export time.** If your training data's
  distribution shifts significantly later, re-run Section 13 and replace
  all six files in `artifacts/` together — don't mix files from different
  export runs.
- **`model.py` must stay in sync with the notebook's architecture.** If you
  change `TransformerAnomalyDetector` in the notebook, mirror the exact same
  change here.
- **CSV batch caps at 500 rows per request.** Raise the limit in
  `main.py`'s `predict_csv` if needed.
- **SHAP/LIME use only 30 background samples** (exported from training) to
  keep explanations fast enough to be usable interactively. This is a
  smaller reference set than the notebook itself uses for its own SHAP/LIME
  analysis - fine for a live demo, but don't treat the app's SHAP/LIME
  output as a substitute for the more thorough comparison already in the
  notebook's Section 11.

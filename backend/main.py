"""
FastAPI backend for the 5G network log anomaly detector - now with
attack-type classification and SHAP/LIME explanations alongside attention.

Run with:
    uvicorn main:app --reload --port 8000

Then open http://127.0.0.1:8000 in a browser.
"""

import io
import json
import os

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from model import load_model
from preprocessing import (dataframe_to_records, load_config, load_vocab,
                            tokenize_and_encode)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

REQUIRED_FILES = [
    "transformer_model.pt", "multiclass_model.pt", "vocab.json",
    "inference_config.json", "attack_label_map.json", "background_samples.json",
]

app = FastAPI(title="5G Anomaly Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state = {}


@app.on_event("startup")
def load_artifacts():
    paths = {name: os.path.join(ARTIFACTS_DIR, name) for name in REQUIRED_FILES}
    missing = [name for name, path in paths.items() if not os.path.exists(path)]
    if missing:
        raise RuntimeError(
            f"Missing {', '.join(missing)} in backend/artifacts/. Run the "
            f"notebook's Section 13 'Model Export' and copy all six exported "
            f"files into backend/artifacts/ before starting this server."
        )

    vocab = load_vocab(paths["vocab.json"])
    config = load_config(paths["inference_config.json"])

    with open(paths["attack_label_map.json"]) as f:
        attack_label_map_raw = json.load(f)
    attack_label_map = {int(k): v for k, v in attack_label_map_raw.items()}

    with open(paths["background_samples.json"]) as f:
        background_samples = json.load(f)

    binary_model = load_model(paths["transformer_model.pt"], vocab_size=len(vocab),
                               max_len=config["max_len"], n_classes=2)
    multiclass_model = load_model(paths["multiclass_model.pt"], vocab_size=len(vocab),
                                   max_len=config["max_len"],
                                   n_classes=config["n_attack_classes"])

    _state["vocab"] = vocab
    _state["config"] = config
    _state["binary_model"] = binary_model
    _state["multiclass_model"] = multiclass_model
    _state["attack_label_map"] = attack_label_map
    _state["background_samples"] = torch.tensor(background_samples, dtype=torch.long)
    _state["id_to_token"] = {v: k for k, v in vocab.items()}

    print(f"Loaded: vocab_size={len(vocab)}, max_len={config['max_len']}, "
          f"{config['n_attack_classes']} attack types: {list(attack_label_map.values())}")


def _filter_real_tokens(tokens: list) -> np.ndarray:
    return np.array([t not in ("<CLS>", "<PAD>", "<UNK>") for t in tokens])


def explain_attention(model, input_ids: torch.Tensor, tokens: list, top_k: int = 5) -> list:
    with torch.no_grad():
        _, attentions = model(input_ids.unsqueeze(0), return_attention=True)
    last_layer_attn = attentions[-1][0]
    cls_attention = last_layer_attn[0].numpy()

    tokens_arr = np.array(tokens)
    keep_mask = _filter_real_tokens(tokens)
    kept_tokens, kept_attn = tokens_arr[keep_mask], cls_attention[keep_mask]

    top_idx = np.argsort(kept_attn)[::-1][:top_k]
    return [{"token": str(kept_tokens[i]), "weight": float(kept_attn[i])} for i in top_idx]


def explain_shap(model, input_ids: torch.Tensor, tokens: list, background: torch.Tensor,
                  top_k: int = 5) -> list:
    import shap

    def predict_fn(batch):
        x = torch.tensor(batch, dtype=torch.long)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=-1)
        return probs.numpy()

    sample = input_ids.numpy().reshape(1, -1)
    explainer = shap.KernelExplainer(predict_fn, background.numpy())
    shap_values = explainer.shap_values(sample, nsamples=50, silent=True)
    shap_for_anomaly = (np.array(shap_values)[1, 0, :] if isinstance(shap_values, list)
                        else shap_values[0, :, 1])

    real_mask = _filter_real_tokens(tokens)
    real_positions = np.where(real_mask)[0]
    top_idx = real_positions[np.argsort(np.abs(shap_for_anomaly[real_positions]))[::-1][:top_k]]
    return [{"token": tokens[i], "weight": float(shap_for_anomaly[i])} for i in top_idx]


def explain_lime(model, input_ids: torch.Tensor, tokens: list, background: torch.Tensor,
                  top_k: int = 5) -> list:
    from lime.lime_tabular import LimeTabularExplainer

    def predict_fn(batch):
        x = torch.tensor(batch, dtype=torch.long)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=-1)
        return probs.numpy()

    explainer = LimeTabularExplainer(
        training_data=background.numpy(),
        mode="classification",
        categorical_features=list(range(background.shape[1])),
        discretize_continuous=False,
    )
    sample = input_ids.numpy()
    lime_exp = explainer.explain_instance(sample, predict_fn, num_features=20,
                                           num_samples=200, labels=(0, 1))

    results = []
    for position, weight in lime_exp.as_map()[1]:
        if tokens[position] in ("<CLS>", "<PAD>", "<UNK>"):
            continue
        results.append({"token": tokens[position], "weight": float(weight)})
        if len(results) >= top_k:
            break
    return results


def predict_one(record: dict, full_explain: bool = False) -> dict:
    config = _state["config"]
    vocab = _state["vocab"]
    binary_model = _state["binary_model"]
    multiclass_model = _state["multiclass_model"]
    attack_label_map = _state["attack_label_map"]

    tokens, encoded = tokenize_and_encode(record, config, vocab)
    input_ids = torch.tensor(encoded, dtype=torch.long)

    with torch.no_grad():
        binary_logits = binary_model(input_ids.unsqueeze(0))
        binary_probs = torch.softmax(binary_logits, dim=-1)[0]

    predicted_class = int(binary_probs.argmax().item())
    is_anomaly = predicted_class == 1

    attack_type = None
    attack_type_probs = None
    if is_anomaly:
        with torch.no_grad():
            attack_logits = multiclass_model(input_ids.unsqueeze(0))
            attack_probs = torch.softmax(attack_logits, dim=-1)[0]
        attack_idx = int(attack_probs.argmax().item())
        attack_type = attack_label_map.get(attack_idx, "Unknown")
        attack_type_probs = {
            attack_label_map.get(i, str(i)): round(float(p), 4)
            for i, p in enumerate(attack_probs.tolist())
        }

    result = {
        "prediction": "ANOMALY" if is_anomaly else "BENIGN",
        "attack_type": attack_type,
        "attack_type_probabilities": attack_type_probs,
        "confidence": round(float(binary_probs[predicted_class].item()), 4),
        "anomaly_probability": round(float(binary_probs[1].item()), 4),
        "explanation": {
            "attention": explain_attention(binary_model, input_ids, tokens),
        },
    }

    if full_explain:
        background = _state["background_samples"]
        result["explanation"]["shap"] = explain_shap(binary_model, input_ids, tokens, background)
        result["explanation"]["lime"] = explain_lime(binary_model, input_ids, tokens, background)

    return result


class ManualRecord(BaseModel):
    fields: dict


class ExplainRequest(BaseModel):
    fields: dict


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": "binary_model" in _state}


@app.get("/api/schema")
def schema():
    config = _state["config"]
    known_informative = [
        "Flow Duration", "Flow Bytes/s", "Flow Packets/s",
        "Fwd Packet Length Mean", "Protocol", "Destination Port",
        "SYN Flag Count", "ACK Flag Count", "PSH Flag Count",
    ]
    return {
        "categorical_cols": config["categorical_cols"],
        "numeric_cols": config["numeric_cols"],
        "key_fields": [c for c in known_informative
                        if c in config["categorical_cols"] + config["numeric_cols"]],
        "attack_types": list(_state["attack_label_map"].values()),
    }


@app.post("/api/predict/manual")
def predict_manual(payload: ManualRecord):
    try:
        return predict_one(payload.fields, full_explain=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/explain/full")
def explain_full(payload: ExplainRequest):
    try:
        return predict_one(payload.fields, full_explain=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if len(df) > 500:
        raise HTTPException(status_code=400,
                             detail=f"{len(df)} rows is a lot for one request - "
                                    f"try a batch of 500 or fewer at a time.")

    records = dataframe_to_records(df)
    results = []
    for i, record in enumerate(records):
        try:
            result = predict_one(record, full_explain=False)
            result["row"] = i
            result["fields"] = record
        except Exception as e:
            result = {"row": i, "error": str(e)}
        results.append(result)

    n_anomalies = sum(1 for r in results if r.get("prediction") == "ANOMALY")
    return {
        "n_rows": len(results),
        "n_anomalies": n_anomalies,
        "results": results,
    }


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

"""
Preprocessing - mirrors Section 3 of the notebook (hybrid tokenisation),
but adapted for inference on ONE NEW record at a time rather than fitting
bins from a whole training dataframe. Uses the bin edges saved during
training (see apply_saved_bins) instead of recomputing quantiles, since
quantiles from a single row would be meaningless.
"""

import json

import numpy as np
import pandas as pd


def load_vocab(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_config(path: str) -> dict:
    with open(path) as f:
        config = json.load(f)
    # JSON can't store numpy arrays - bin_edges were saved as plain lists
    config["bin_edges"] = {col: np.array(edges) for col, edges in config["bin_edges"].items()}
    return config


def apply_saved_bins(value, edges: np.ndarray) -> int:
    """Slots a new numeric value into the bin edges learned at training time.
    Mirrors the notebook's apply_saved_bins exactly."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return -1
    try:
        value = float(value)
    except (TypeError, ValueError):
        return -1
    result = pd.cut([value], bins=edges, labels=False, include_lowest=True)[0]
    if pd.isna(result):
        result = 0 if value <= edges[0] else len(edges) - 2
    return int(result)


def tokenize_record(record: dict, categorical_cols: list, numeric_cols: list,
                     bin_edges: dict) -> list:
    """Converts one raw record (dict of column -> value) into the same
    'column=value' / 'column=binN' token sequence used during training.
    Missing categorical fields fall back to a placeholder (naturally maps to
    <UNK> in the vocab, which is the correct behaviour for genuinely unknown
    values). Missing numeric fields default to 0, matching the base rate of
    the sparse zeek.udp_conns flag family (~99.6% zero in the real data)."""
    tokens = []
    for col in categorical_cols:
        value = record.get(col, "MISSING")
        tokens.append(f"{col}={value}")
    for col in numeric_cols:
        raw_value = record.get(col, 0)
        edges = bin_edges.get(col)
        if edges is None:
            bin_id = 0
        else:
            bin_id = apply_saved_bins(raw_value, edges)
        tokens.append(f"{col}=bin{bin_id}")
    return tokens


def encode_sequence(tokens: list, vocab: dict, max_len: int) -> list:
    """Identical to the notebook's encode_sequence."""
    ids = [vocab["<CLS>"]]
    for tok in tokens[: max_len - 1]:
        ids.append(vocab.get(tok, vocab["<UNK>"]))
    ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids[:max_len]


def tokenize_and_encode(record: dict, config: dict, vocab: dict) -> tuple:
    """Full pipeline: raw record -> token list -> encoded id sequence.
    Returns (token_list, encoded_ids) - the token list is kept around so the
    explanation step can map attention weights back to human-readable tokens."""
    tokens = tokenize_record(record, config["categorical_cols"],
                              config["numeric_cols"], config["bin_edges"])
    # Prepend the <CLS> placeholder token so token_list and encoded_ids line up
    # position-for-position (encode_sequence itself prepends the <CLS> id).
    full_tokens = ["<CLS>"] + tokens
    encoded = encode_sequence(tokens, vocab, config["max_len"])
    padded_tokens = (full_tokens[: config["max_len"]]
                      + ["<PAD>"] * max(0, config["max_len"] - len(full_tokens)))
    return padded_tokens, encoded


def dataframe_to_records(df: pd.DataFrame) -> list:
    """Converts an uploaded CSV into a list of record dicts, one per row.
    Replaces inf/-inf/NaN with None first - raw JSON can't serialize inf or
    NaN, and these get included in the API response so the frontend can
    request a full explanation for a specific row later.

    Note: pandas silently turns None back into NaN on float64 columns via
    .where()/.replace(), so the NaN->None substitution has to happen on the
    plain dicts after to_dict(), not on the dataframe itself."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan)
    records = df.to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                record[key] = None
    return records

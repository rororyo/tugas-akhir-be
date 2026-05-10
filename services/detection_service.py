import time
import uuid
from datetime import datetime

import numpy as np
import pandas as pd

from models.model_loader import get_artifacts
from preprocessing.preprocessor import validate_batch

_ANON_COUNTER = 6001


def _next_anon_uid() -> int:
    global _ANON_COUNTER
    uid = _ANON_COUNTER
    _ANON_COUNTER += 1
    return uid


def _risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "high"
    if prob >= 0.4:
        return "medium"
    return "low"


def _confidence(prob: float) -> str:
    if prob > 0.8 or prob < 0.2:
        return "high"
    if prob > 0.6 or prob < 0.4:
        return "medium"
    return "low"


def _run_pipeline(df_raw: pd.DataFrame) -> tuple:
    t0 = time.time()
    art          = get_artifacts()
    preprocessor = art["preprocessor"]
    embedder     = art["embedder"]
    best_model   = art["best_model"]

    X_tab_scaled, df_entity, df_feat_raw, imputed_cols = preprocessor.transform(df_raw)
    X_emb_64d = embedder.get_embeddings(df_feat_raw, df_entity)

    X_sc3 = np.concatenate([X_tab_scaled, X_emb_64d], axis=1)
    probs = best_model.predict_proba(X_sc3)[:, 1]

    return probs, imputed_cols, int((time.time() - t0) * 1000)


def run_batch(df_raw: pd.DataFrame) -> dict:
    art       = get_artifacts()
    best_name = art["best_name"]
    thr       = art["threshold"]

    err = validate_batch(df_raw)
    if err:
        raise ValueError(err)

    df_raw = df_raw.copy()
    if "transaction_id" not in df_raw.columns:
        df_raw["transaction_id"] = [str(uuid.uuid4()) for _ in range(len(df_raw))]
    else:
        mask = df_raw["transaction_id"].isna()
        df_raw.loc[mask, "transaction_id"] = [str(uuid.uuid4()) for _ in range(mask.sum())]

    required_mask = df_raw["amount"].notna() & df_raw["merchant_category"].notna()
    df_skip  = df_raw[~required_mask]
    df_valid = df_raw[required_mask].copy()

    skipped = [
        {
            "transaction_id": str(row.get("transaction_id", "?")),
            "status":         "skipped",
            "reason":         "missing required field (amount or merchant_category)",
        }
        for _, row in df_skip.iterrows()
    ]

    probs, imputed_cols, elapsed_ms = _run_pipeline(df_valid)

    predictions = []
    for i, (_, row) in enumerate(df_valid.iterrows()):
        prob_f = float(probs[i])
        predictions.append({
            "transaction_id":    str(row["transaction_id"]),
            "user_id":           int(row["user_id"]) if "user_id" in row else None,
            "fraud_probability": round(prob_f, 6),
            "is_fraud":          bool(prob_f >= thr),
            "risk_level":        _risk_level(prob_f),
            "confidence":        _confidence(prob_f),
        })

    flagged = sum(p["is_fraud"] for p in predictions)
    n       = len(predictions)

    return {
        "predictions": predictions,
        "skipped":     skipped,
        "batch_summary": {
            "total":      n + len(skipped),
            "processed":  n,
            "skipped":    len(skipped),
            "flagged":    flagged,
            "fraud_rate": round(flagged / n, 4) if n else 0,
        },
        "imputed_fields":     imputed_cols,
        "warning":            (
            "Hasil didasarkan pada data tidak lengkap. "
            "Beberapa fitur menggunakan nilai default."
            if imputed_cols else None
        ),
        "model_used":         best_name,
        "threshold":          thr,
        "processing_time_ms": elapsed_ms,
        "timestamp":          datetime.now().isoformat(),
    }


def run_single(body: dict) -> dict:
    art       = get_artifacts()
    best_name = art["best_name"]
    thr       = art["threshold"]

    if not body.get("amount") and body.get("amount") != 0:
        raise ValueError("Jumlah transaksi wajib diisi")
    if not body.get("merchant_category"):
        raise ValueError("Kategori merchant wajib diisi")

    row = {k: v for k, v in body.items() if k != "is_fraud"}

    assigned = False
    if not row.get("user_id") and row.get("user_id") != 0:
        row["user_id"] = _next_anon_uid()
        assigned = True
    else:
        row["user_id"] = int(row["user_id"])

    row.setdefault("transaction_id", str(uuid.uuid4()))

    probs, imputed_cols, elapsed_ms = _run_pipeline(pd.DataFrame([row]))
    prob_f = float(probs[0])

    return {
        "transaction_id":    row["transaction_id"],
        "user_id":           row["user_id"],
        "assigned_user_id":  assigned,
        "fraud_probability": round(prob_f, 6),
        "is_fraud":          bool(prob_f >= thr),
        "risk_level":        _risk_level(prob_f),
        "confidence":        _confidence(prob_f),
        "imputed_fields":    imputed_cols,
        "warning":           (
            "Hasil didasarkan pada data tidak lengkap. "
            "Beberapa fitur menggunakan nilai default."
            if imputed_cols else None
        ),
        "model_used":        best_name,
        "threshold":         thr,
        "processing_time_ms":elapsed_ms,
        "timestamp":         datetime.now().isoformat(),
    }

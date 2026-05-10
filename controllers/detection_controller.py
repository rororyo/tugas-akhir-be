import io
import pandas as pd
from flask import request, jsonify
from services.detection_service import run_batch, run_single
from models.model_loader import get_artifacts


def predict_batch():
    """POST /api/predict/batch — multipart file upload (CSV or XLSX)."""
    if "file" not in request.files:
        return jsonify({"error": "multipart/form-data with 'file' field required"}), 400

    f     = request.files["file"]
    fname = (f.filename or "").lower()

    try:
        if fname.endswith(".csv"):
            df_raw = pd.read_csv(io.BytesIO(f.read()))
        elif fname.endswith((".xlsx", ".xls")):
            df_raw = pd.read_excel(io.BytesIO(f.read()))
        else:
            return jsonify({"error": "Only CSV and XLSX files are supported"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to parse file: {e}"}), 400

    if "user_id" not in df_raw.columns:
        return jsonify({"error": "File must contain a 'user_id' column"}), 400

    try:
        df_raw["user_id"] = df_raw["user_id"].astype(int)
    except (ValueError, TypeError):
        return jsonify({"error": "'user_id' column must be integer"}), 400

    try:
        result = run_batch(df_raw)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(result)


def predict_single():
    """POST /api/predict/single — JSON body for the manual form."""
    body = request.get_json()
    if not body:
        return jsonify({"error": "JSON body required"}), 400

    try:
        result = run_single(body)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(result)


def predict_json():
    """POST /predict — JSON batch used by the frontend file-upload mode."""
    data = request.get_json()
    if not data or "transactions" not in data:
        return jsonify({"error": "transactions required"}), 400

    transactions = data["transactions"]
    for tx in transactions:
        tx.pop("is_fraud", None)
        if not tx.get("user_id") and tx.get("user_id") != 0:
            from services.detection_service import _next_anon_uid
            tx["user_id"] = _next_anon_uid()
        else:
            tx["user_id"] = int(tx["user_id"])

    try:
        result = run_batch(pd.DataFrame(transactions))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(result)


def health():
    """GET /api/health."""
    art    = get_artifacts()
    loaded = bool(art)
    return jsonify({
        "status":           "ok" if loaded else "loading",
        "model":            art.get("best_name", "unknown") if loaded else "unknown",
        "artifacts_loaded": loaded,
    })


def model_info():
    """GET /api/model-info."""
    art = get_artifacts()
    if not art:
        return jsonify({"error": "model not loaded"}), 503

    entity_counts = {
        etype: int(mat.shape[0])
        for etype, mat in art["embedder"].entity_feat_matrices.items()
    }

    return jsonify({
        "best_model":          art["best_name"],
        "threshold":           art["threshold"],
        "feature_count":       len(art["preprocessor"].feature_cols_gnn),
        "known_entity_counts": entity_counts,
    })

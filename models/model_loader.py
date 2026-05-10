import json
import os
import pickle

# Custom classes must be importable before any pickle.load of the Sc3 models
import model_classes  # noqa: F401

from preprocessing.preprocessor import TabularPreprocessor
from preprocessing.inductive_gnn import HGTEmbedder

ARTIFACTS_DIR = "fraud_detection_artifacts"

_artifacts = {}


def init_model(artifacts_dir: str = ARTIFACTS_DIR):
    global _artifacts

    # ── Metadata ──────────────────────────────────────────────────────────────
    with open(os.path.join(artifacts_dir, "metadata_v28_hp.json")) as f:
        meta = json.load(f)

    best_name = meta["best_s3"]
    threshold = meta["all_s3"][best_name]["threshold"]

    # ── Best Sc3 classifier ───────────────────────────────────────────────────
    sc3_file_map = {
        "ANN":             "model_s3_ANN_residual.pkl",
        "GradientBoosting":"model_s3_GradientBoosting_emb.pkl",
        "RandomForest":    "model_s3_RandomForest_emb.pkl",
    }
    model_file = sc3_file_map[best_name]
    with open(os.path.join(artifacts_dir, model_file), "rb") as f:
        best_model = pickle.load(f)

    # ── Stage 1 preprocessor ─────────────────────────────────────────────────
    preprocessor = TabularPreprocessor(artifacts_dir=artifacts_dir)

    # ── Stage 2 HGT embedder ─────────────────────────────────────────────────
    embedder = HGTEmbedder(artifacts_dir=artifacts_dir)

    _artifacts = {
        "meta":        meta,
        "best_name":   best_name,
        "threshold":   threshold,
        "best_model":  best_model,
        "preprocessor":preprocessor,
        "embedder":    embedder,
    }

    print(f"Model ready — best_s3={best_name}  threshold={threshold:.4f}")


def get_artifacts() -> dict:
    return _artifacts

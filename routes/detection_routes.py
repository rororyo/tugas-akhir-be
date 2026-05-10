from flask import Blueprint
from controllers.detection_controller import (
    predict_json, predict_batch, predict_single, health, model_info
)

fraud_bp = Blueprint("fraud", __name__)

fraud_bp.route("/predict",            methods=["POST"])(predict_json)
fraud_bp.route("/api/predict/batch",  methods=["POST"])(predict_batch)
fraud_bp.route("/api/predict/single", methods=["POST"])(predict_single)
fraud_bp.route("/api/health",         methods=["GET"])(health)
fraud_bp.route("/api/model-info",     methods=["GET"])(model_info)

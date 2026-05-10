from flask import Flask, jsonify
from flask_cors import CORS
from routes.detection_routes import fraud_bp
from models.model_loader import init_model
from datetime import datetime

def create_app():
    app = Flask(__name__)

    CORS(
        app,
        resources={
            r"/*": {
                "origins": [
                    "http://localhost:3000",
                    "https://sigap-two.vercel.app"
                ]
            }
        },
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
        supports_credentials=True
    )

    app.register_blueprint(fraud_bp)

    @app.route("/")
    def health():
        return jsonify({
            "status": "online",
            "service": "Fraud Detection API",
            "timestamp": datetime.now().isoformat()
        })

    return app


if __name__ == "__main__":
    print("Initializing model...")
    init_model()

    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)
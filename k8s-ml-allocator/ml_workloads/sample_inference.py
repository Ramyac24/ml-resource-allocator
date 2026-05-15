"""
sample_inference.py
Flask inference server for the trained Iris classifier.
Runs as a long-lived Kubernetes Deployment.

Endpoints
---------
GET  /health        — liveness/readiness probe
POST /predict       — classify iris features
GET  /model/info    — model metadata
"""

import os
import logging
import joblib
import numpy as np
from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger("inference")

app    = Flask(__name__)
MODEL_PATH = os.getenv("MODEL_PATH", "/tmp/iris_model.pkl")

# ── Load or train a fallback model ────────────────────────────────────────────
pipeline = None

def _load_model():
    global pipeline
    if os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
        logger.info(f"Loaded model from {MODEL_PATH}")
    else:
        logger.warning(f"Model not found at {MODEL_PATH} — training fallback model…")
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        data = load_iris()
        X, y = data.data, data.target
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=50, random_state=42)),
        ])
        pipeline.fit(X, y)
        joblib.dump(pipeline, MODEL_PATH)
        logger.info(f"Fallback model trained and saved to {MODEL_PATH}")


_load_model()

IRIS_CLASSES = load_iris().target_names.tolist()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": pipeline is not None})


@app.post("/predict")
def predict():
    """
    Expects JSON: {"features": [sepal_len, sepal_wid, petal_len, petal_wid]}
    or batch:     {"features": [[...], [...]]}
    """
    data = request.get_json(force=True)
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' field"}), 400

    features = data["features"]
    # Allow single sample or batch
    if isinstance(features[0], (int, float)):
        features = [features]

    try:
        X = np.array(features, dtype=float)
        preds  = pipeline.predict(X)
        probas = pipeline.predict_proba(X)

        results = []
        for pred, proba in zip(preds, probas):
            results.append({
                "class":       IRIS_CLASSES[pred],
                "class_id":    int(pred),
                "confidence":  round(float(proba.max()), 4),
                "probabilities": {
                    cls: round(float(p), 4)
                    for cls, p in zip(IRIS_CLASSES, proba)
                },
            })

        return jsonify({
            "predictions": results,
            "count": len(results),
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.get("/model/info")
def model_info():
    if pipeline is None:
        return jsonify({"error": "No model loaded"}), 503
    clf = pipeline.named_steps.get("clf")
    return jsonify({
        "model_path":    MODEL_PATH,
        "model_type":    type(clf).__name__ if clf else "unknown",
        "n_classes":     len(IRIS_CLASSES),
        "classes":       IRIS_CLASSES,
        "n_features":    4,
        "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    logger.info(f"Inference server starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)

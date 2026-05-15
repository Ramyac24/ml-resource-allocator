"""
sample_training.py
Sample ML training workload — trains a RandomForest on the Iris dataset.
Designed to run as a Kubernetes Job container.

Environment variables
---------------------
MODEL_NAME      : output filename stem (default: iris-rf)
N_ESTIMATORS    : number of trees (default: 100)
MODEL_PATH      : where to save the model (default: /tmp/<MODEL_NAME>.pkl)
"""

import os
import time
import logging
import joblib
import numpy as np

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger("training")


def train(model_name: str = "iris-rf",
          n_estimators: int = 100,
          model_path: str = None) -> dict:

    model_path = model_path or f"/tmp/{model_name}.pkl"
    logger.info(f"Starting training job: {model_name}")
    logger.info(f"Config — n_estimators={n_estimators}, output={model_path}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    logger.info("Loading Iris dataset…")
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train={len(X_train)}, Test={len(X_test)}")

    # ── Build pipeline ────────────────────────────────────────────────────────
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    logger.info("Fitting RandomForest pipeline…")
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    cv     = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")

    logger.info(f"Test accuracy:      {acc:.4f}")
    logger.info(f"CV mean ± std:      {cv.mean():.4f} ± {cv.std():.4f}")
    logger.info(f"Training time:      {train_time:.2f}s")
    logger.info("\n" + classification_report(y_test, y_pred,
                                             target_names=data.target_names))

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info(f"Model saved → {model_path}")

    return {
        "model_name":  model_name,
        "model_path":  model_path,
        "accuracy":    round(acc, 4),
        "cv_mean":     round(cv.mean(), 4),
        "cv_std":      round(cv.std(), 4),
        "train_time_s": round(train_time, 2),
    }


if __name__ == "__main__":
    result = train(
        model_name   = os.getenv("MODEL_NAME",    "iris-rf"),
        n_estimators = int(os.getenv("N_ESTIMATORS", "100")),
        model_path   = os.getenv("MODEL_PATH",    "/tmp/iris_model.pkl"),
    )
    logger.info(f"Training complete: {result}")

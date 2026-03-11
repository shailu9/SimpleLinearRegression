import joblib
from pathlib import Path
from typing import Any

from src.model_def import SimpleLinearRegressionWithGradientDescent


def save_model(model: Any, path: Path) -> None:
    """Serialize model parameters and metadata to disk.

    Right now we only store the coefficients and a small info dict, but this
    could be extended with preprocessing steps, training hyperparameters, etc.
    """
    payload = {
        "w": model.w,
        "b": model.b,
    }
    joblib.dump(payload, path)


def load_model(path: Path) -> SimpleLinearRegressionWithGradientDescent:
    """Load parameters and return a ready-to-use model instance."""
    data = joblib.load(path)
    m = SimpleLinearRegressionWithGradientDescent()
    m.w = data["w"]
    m.b = data["b"]
    return m

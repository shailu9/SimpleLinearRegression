import os
import numpy as np
from pathlib import Path

from src.model_io import load_model


def predict_from_saved(x_values: np.ndarray, model_path: Path | str) -> np.ndarray:
    """Load a stored model and run predictions on input array."""
    model = load_model(Path(model_path))
    return model.predict(x_values)


def main():
    # example usage when running as a script
    project_root = os.path.dirname(os.path.dirname(__file__))
    model_file = os.path.join(project_root, "models", "linear_model.joblib")
    # sample points to predict
    xs = np.array([1.0, 5.0, 10.0])
    preds = predict_from_saved(xs, model_file)
    print(f"predictions for {xs}: {preds}")


if __name__ == "__main__":
    main()

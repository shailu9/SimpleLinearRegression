import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.model_io import save_model
from pathlib import Path

from src.model_def import SimpleLinearRegressionWithGradientDescent


def load_dataset(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a CSV file and return feature/target arrays.

    This helper assumes the dataset has exactly two columns: the first is the
    independent variable (feature) and the second the dependent variable
    (target).
    """
    df = pd.read_csv(csv_path)
    # In exploration we found an extra unnamed column that we don't need, so we drop it if it exists
    df=df.drop('Unnamed: 0',axis='columns')
    if df.shape[1] < 2:
        raise ValueError("expected at least two columns in the dataset")

    # grab the first column as X and the second column as y
    X = df.iloc[:, 0].to_numpy(dtype=float)
    y = df.iloc[:, 1].to_numpy(dtype=float)
    return X, y


def main() -> None:
    # build an absolute path to the CSV no matter where the script is run
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_file = os.path.join(project_root, "data", "Salary_dataset.csv")

    print(f"loading data from {data_file}")
    X, y = load_dataset(data_file)
    print(f"loaded {X.shape[0]} examples")

    # split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")

    model = SimpleLinearRegressionWithGradientDescent(
        learning_rate=0.01, n_iterations=1000
    )

    print("starting training on training set...")
    model.fit(X_train, y_train)

    print("training finished")
    print(f"model parameters: w={model.w}, b={model.b}")

    # evaluate on test set
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    print(f"test set MSE: {mse}")

    # save the trained model for later use

    out_path = Path(project_root) / "models" / "linear_model.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, out_path)
    print(f"saved model to {out_path}")


if __name__ == "__main__":
    main()

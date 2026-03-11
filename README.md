# Simple Linear Regression

This project implements a simple linear regression model from scratch in
Python. It includes a `SimpleLinearRegressionWithGradientDescent` class that
demonstrates training using gradient descent on a single feature dataset.

## Project layout

- `data/Salary_dataset.csv` – sample dataset (years of experience vs salary).
- `src/model_def.py` – model implementation.
- `src/train.py` – script to load the data, train the model (with an
  80/20 split), evaluate on the hold-out set and save the trained model.
- `src/model_io.py` – helpers to serialize/deserialize model parameters using
  `joblib`.
- `src/predict.py` – command-line example showing how to load a saved model and
  make predictions from a NumPy array.
- `models/` – output folder for saved model parameters.

## Training

The repository includes a `src/train.py` script (and a wrapper CLI in `main.py`) that:

1. Loads the salary dataset from `data/Salary_dataset.csv` using pandas.
2. Extracts the first column as features (X) and the second as targets (y).
3. Splits the data into training and test sets using scikit-learn's
   `train_test_split` (default 80/20).
4. Instantiates the `SimpleLinearRegressionWithGradientDescent` model from
   `src/model_def.py` and calls its `fit` method on the training partition.
5. Prints cost every 100 iterations, evaluates the model on the held-out test
   set (mean squared error), and saves the trained model using `joblib` to
   `models/linear_model.joblib`.

### Using the command‑line interface

The project exposes two subcommands through `main.py` which is designed to be
invoked either via `python main.py` or through `uv` as defined in
`pyproject.toml`.

```powershell
# train the model (same behavior as running src/train directly)
python main.py train

# make predictions with an existing model file
python main.py predict 1 4 6
# use --model to point at a different joblib file if needed
```

If you prefer to run the training script directly:

```powershell
cd SimpleLinearRegression
.\.venv\Scripts\activate
python -m src.train
```

(The wrapper CLI simply routes to the same functionality.)

## Notes

- The current learning rate may cause the cost to diverge; adjust `learning_rate`
  or scale/normalize the data if necessary. A value like `1e-4` tends to be
  more stable for the salary dataset.
- `src/predict.py` already provides an example CLI call; you can import
  `predict_from_saved` to use in larger applications.
- Dependencies are listed in `requirements.txt` and in `pyproject.toml` if you
  use `uv` or another PEP 517/518 build tool.
- Feel free to extend the CLI (`main.py`) with additional subcommands such as
  `evaluate`, `export`, or hyperparameter options.
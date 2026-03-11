import argparse
import os
import sys
from pathlib import Path


def train(args: argparse.Namespace) -> None:
    """Run the training script with optional hyperparameters."""
    # import here to keep dependencies lazy
    from src.train import main as train_main

    # forward command-line arguments if needed in the future
    train_main()


def predict(args: argparse.Namespace) -> None:
    """Run prediction on a list of values supplied via CLI."""
    from src.predict import predict_from_saved
    import numpy as np

    vals = np.array(args.values, dtype=float)
    model_path = Path(args.model) if args.model else Path("models") / "linear_model.joblib"
    preds = predict_from_saved(vals, model_path)
    for x, y in zip(vals, preds):
        print(f"{x} -> {y}")


def main():
    parser = argparse.ArgumentParser(
        description="Utility wrapper for the simple linear regression project"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="train the model")
    # possible future options: --lr, --epochs, etc.

    pred_parser = sub.add_parser("predict", help="predict using a saved model")
    pred_parser.add_argument(
        "values",
        nargs="+",
        help="numeric feature values to predict",
    )
    pred_parser.add_argument(
        "--model",
        help="path to saved joblib file (defaults to models/linear_model.joblib)",
    )

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

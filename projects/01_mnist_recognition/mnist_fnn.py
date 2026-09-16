"""Train a beginner-friendly fully connected network on local MNIST data."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


RANDOM_STATE = 42
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "MNIST_Data"


def load_mnist(data_dir: Path) -> tuple[np.ndarray, ...]:
    """Load, normalize, and flatten the four local MNIST arrays."""
    x_train = np.load(data_dir / "x_train.npy").astype(np.float32) / 255.0
    y_train = np.load(data_dir / "y_train.npy").reshape(-1).astype(np.int64)
    x_test = np.load(data_dir / "x_test.npy").astype(np.float32) / 255.0
    y_test = np.load(data_dir / "y_test.npy").reshape(-1).astype(np.int64)

    # MLPClassifier expects one row per sample and one column per feature.
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    return x_train, y_train, x_test, y_test


def stratified_subset(
    x: np.ndarray, y: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Take a reproducible subset while keeping class proportions similar."""
    if size >= len(y):
        return x, y
    x_small, _, y_small, _ = train_test_split(
        x,
        y,
        train_size=size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return x_small, y_small


def build_model(quick: bool) -> MLPClassifier:
    """Create the same model family as the notebook."""
    return MLPClassifier(
        hidden_layer_sizes=(64,) if quick else (128,),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=10 if quick else 30,
        shuffle=True,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=5,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing x_train.npy, y_train.npy, x_test.npy, y_test.npy.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller subsets and fewer iterations for a fast smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(RANDOM_STATE)

    if args.quick:
        # Quick mode intentionally stops early; this warning would only repeat that fact.
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        print("Quick mode: results are only for checking the workflow.\n")

    x_train_full, y_train_full, x_test, y_test = load_mnist(args.data_dir)
    if args.quick:
        x_train_full, y_train_full = stratified_subset(
            x_train_full, y_train_full, 12_000
        )
        x_test, y_test = stratified_subset(x_test, y_test, 2_000)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    print(f"Training shape  : {x_train.shape}")
    print(f"Validation shape: {x_val.shape}")
    print(f"Test shape      : {x_test.shape}")

    model = build_model(args.quick)
    model.fit(x_train, y_train)

    y_val_pred = model.predict(x_val)
    y_val_prob = model.predict_proba(x_val)
    y_test_pred = model.predict(x_test)
    y_test_prob = model.predict_proba(x_test)

    print(f"\nCompleted epochs   : {model.n_iter_}")
    print(f"Validation accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"Validation log loss: {log_loss(y_val, y_val_prob):.4f}")
    print(f"Test accuracy      : {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Test log loss      : {log_loss(y_test, y_test_prob):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred, digits=4))

    artifacts_dir = PROJECT_DIR / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_curve_)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("MNIST MLP training loss")
    plt.tight_layout()
    output_path = artifacts_dir / "mnist_loss.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Loss curve saved to: {output_path}")


if __name__ == "__main__":
    main()

"""Compare regression models on the bundled California Housing dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


RANDOM_STATE = 42
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = PROJECT_DIR / "data" / "california_housing.csv"
TARGET_COLUMN = "MedHouseVal"


def load_data(data_file: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the local CSV and separate features from the prediction target."""
    frame = pd.read_csv(data_file)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")
    return frame.drop(columns=TARGET_COLUMN), frame[TARGET_COLUMN]


def build_models(quick: bool) -> dict[str, object]:
    """Build four models with scaling only where it is needed."""
    return {
        "KNN (k=5)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=5)),
            ]
        ),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=10, random_state=RANDOM_STATE
        ),
        "Neural Network": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(50,),
                        activation="relu",
                        max_iter=150 if quick else 500,
                        early_stopping=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate_models(
    models: dict[str, object],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Fit each model and return comparable train/test metrics."""
    rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        train_prediction = model.predict(x_train)
        test_prediction = model.predict(x_test)
        train_mse = mean_squared_error(y_train, train_prediction)
        test_mse = mean_squared_error(y_test, test_prediction)
        rows.append(
            {
                "Model": name,
                "Training MSE": train_mse,
                "Test MSE": test_mse,
                "Test RMSE": np.sqrt(test_mse),
                "Test R^2": r2_score(y_test, test_prediction),
            }
        )
    return pd.DataFrame(rows).sort_values("Test MSE").reset_index(drop=True)


def tune_neural_network(
    x_train: pd.DataFrame, y_train: pd.Series, quick: bool
) -> GridSearchCV:
    """Choose network shape and activation using training data only."""
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    max_iter=150 if quick else 500,
                    early_stopping=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    parameter_grid = {
        "model__hidden_layer_sizes": [(50,), (50, 50), (50, 50, 50)],
        "model__activation": ["relu", "tanh", "logistic"],
    }
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        cv=kfold,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        return_train_score=True,
    )
    search.fit(x_train, y_train)
    return search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 3,000 samples and shorter neural-network training.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run 5-fold grid search for the neural network.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x, y = load_data(args.data_file)
    if args.quick:
        sample_indices = x.sample(
            n=min(3_000, len(x)), random_state=RANDOM_STATE
        ).index
        x = x.loc[sample_indices]
        y = y.loc[sample_indices]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=RANDOM_STATE
    )
    print(f"Samples: {len(x):,}; features: {x.shape[1]}")
    print(f"Training samples: {len(x_train):,}; test samples: {len(x_test):,}")

    results = evaluate_models(
        build_models(args.quick), x_train, x_test, y_train, y_test
    )
    print("\nModel comparison:")
    print(results.to_string(index=False))

    artifacts_dir = PROJECT_DIR / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    results.to_csv(artifacts_dir / "model_comparison.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(results["Model"], results["Test MSE"])
    plt.ylabel("Test MSE")
    plt.title("California Housing: test MSE comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(artifacts_dir / "test_mse_comparison.png", dpi=150)
    plt.close()

    if args.tune:
        print("\nRunning 5-fold grid search (9 combinations, 45 fits)...")
        search = tune_neural_network(x_train, y_train, args.quick)
        best_prediction = search.best_estimator_.predict(x_test)
        best_mse = mean_squared_error(y_test, best_prediction)
        print(f"Best parameters: {search.best_params_}")
        print(f"Best mean validation MSE: {-search.best_score_:.4f}")
        print(f"Final test MSE : {best_mse:.4f}")
        print(f"Final test RMSE: {np.sqrt(best_mse):.4f}")
        print(f"Final test R^2 : {r2_score(y_test, best_prediction):.4f}")

    print(f"\nArtifacts saved in: {artifacts_dir}")


if __name__ == "__main__":
    main()

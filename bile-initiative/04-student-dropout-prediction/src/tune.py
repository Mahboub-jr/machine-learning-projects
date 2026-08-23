import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.preprocess import split_features_target, build_preprocessor


DATA_PATH = "data/raw/student_dropout_dataset_v3.csv"


def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    X, y = split_features_target(df)

    # Preserve final test set
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    preprocessor, _, _ = build_preprocessor(X_train)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # Hyperparameters to test
    param_grid = {
        "classifier__C": [
            0.01,
            0.05,
            0.1,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
        ],
        "classifier__solver": [
            "liblinear",
            "lbfgs",
        ],
    }

    # Stratified CV preserves class proportions
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring={
            "roc_auc": "roc_auc",
            "f1": "f1",
            "recall": "recall",
        },
        refit="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    print("Starting hyperparameter tuning...\n")

    search.fit(X_train, y_train)

    print("\n=== TUNING COMPLETE ===")

    print("\nBest Parameters:")
    print(search.best_params_)

    print(
        f"\nBest Cross-Validated ROC-AUC: "
        f"{search.best_score_:.4f}"
    )

    best_index = search.best_index_

    results = search.cv_results_

    print(
        f"CV F1 at Best Parameters: "
        f"{results['mean_test_f1'][best_index]:.4f}"
    )

    print(
        f"CV Recall at Best Parameters: "
        f"{results['mean_test_recall'][best_index]:.4f}"
    )

    print(
        f"Training ROC-AUC at Best Parameters: "
        f"{results['mean_train_roc_auc'][best_index]:.4f}"
    )

    print("\nFinal test set remains untouched.")


if __name__ == "__main__":
    main()
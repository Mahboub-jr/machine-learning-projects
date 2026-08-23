import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.preprocess import split_features_target, build_preprocessor


DATA_PATH = "data/raw/student_dropout_dataset_v3.csv"


def evaluate_model(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
):
    """
    Train and evaluate one classification model.
    """

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1": f1_score(y_test, predictions),
        "ROC_AUC": roc_auc_score(y_test, probabilities),
    }

    matrix = confusion_matrix(y_test, predictions)

    return results, matrix


def evaluate_thresholds(y_test, probabilities):
    """
    Evaluate different probability thresholds for binary classification.
    """

    thresholds = np.arange(0.30, 0.65, 0.05)

    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
        ).ravel()

        results.append(
            {
                "Threshold": threshold,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "F1": f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "True_Positive": tp,
                "False_Positive": fp,
                "True_Negative": tn,
                "False_Negative": fn,
            }
        )

    return pd.DataFrame(results)


def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Separate features and target
    X, y = split_features_target(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Models to compare
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Balanced Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42,
        ),
    }

    all_results = []

    # Evaluate all models
    for name, model in models.items():
        preprocessor, _, _ = build_preprocessor(X_train)

        results, matrix = evaluate_model(
            name=name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            preprocessor=preprocessor,
        )

        all_results.append(results)

        print(f"\n=== {name} ===")
        print("Confusion Matrix:")
        print(matrix)

    # Model comparison table
    results_df = pd.DataFrame(all_results)

    print("\n=== MODEL COMPARISON ===")

    print(
        results_df
        .sort_values("F1", ascending=False)
        .round(4)
        .to_string(index=False)
    )

    # Threshold analysis for balanced logistic regression
    print(
        "\n=== BALANCED LOGISTIC REGRESSION "
        "THRESHOLD ANALYSIS ==="
    )

    preprocessor, _, _ = build_preprocessor(X_train)

    balanced_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    balanced_model.fit(X_train, y_train)

    probabilities = balanced_model.predict_proba(X_test)[:, 1]

    threshold_results = evaluate_thresholds(
        y_test,
        probabilities,
    )

    print(
        threshold_results
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
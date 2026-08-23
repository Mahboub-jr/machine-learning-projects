"""
Model loading and inference for the Student Dropout Prediction app.

This module is a thin, read-only wrapper around the already-trained artifact at
``models/student_dropout_pipeline.joblib``.  It never trains, refits or mutates
the pipeline, and it never re-implements preprocessing: the saved sklearn
``Pipeline`` owns imputation, scaling and encoding.

The artifact is a dict with two keys:
    "model"     -> fitted sklearn Pipeline (preprocessor + LogisticRegression)
    "threshold" -> decision threshold selected during evaluation (0.55)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "student_dropout_pipeline.joblib"

# Fallback ordering, used only if the fitted pipeline does not expose
# ``feature_names_in_``.  Student_ID is deliberately absent: it was dropped
# before training and is not a model input.
FALLBACK_FEATURE_ORDER: tuple[str, ...] = (
    "Age",
    "Gender",
    "Family_Income",
    "Internet_Access",
    "Study_Hours_per_Day",
    "Attendance_Rate",
    "Assignment_Delay_Days",
    "Travel_Time_Minutes",
    "Part_Time_Job",
    "Scholarship",
    "Stress_Index",
    "GPA",
    "Semester_GPA",
    "CGPA",
    "Semester",
    "Department",
    "Parental_Education",
)

LOWER_RISK = "Lower Risk"
ELEVATED_RISK = "Elevated Risk"


class ModelLoadError(RuntimeError):
    """Raised when the saved artifact is missing, unreadable or malformed."""


class FeatureMismatchError(ValueError):
    """Raised when an input record does not cover the pipeline's features."""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelBundle:
    """Everything the app needs about the loaded artifact."""

    model: Any
    threshold: float
    feature_names: tuple[str, ...]
    path: Path
    classifier_name: str
    classifier_params: dict[str, Any]
    n_encoded_features: int | None


@dataclass(frozen=True)
class PredictionResult:
    probability: float
    threshold: float
    is_elevated: bool

    @property
    def label(self) -> str:
        return ELEVATED_RISK if self.is_elevated else LOWER_RISK

    @property
    def margin(self) -> float:
        """Signed distance from the decision threshold, in probability points."""
        return self.probability - self.threshold

    @property
    def retention_probability(self) -> float:
        return 1.0 - self.probability


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    value: Any
    contribution: float  # log-odds, relative to the reference profile

    @property
    def direction(self) -> str:
        return "increases" if self.contribution > 0 else "decreases"


@dataclass(frozen=True)
class Explanation:
    """Exact additive decomposition of the model's log-odds for one student.

    For logistic regression the decision function is linear in the *encoded*
    feature space, so it can be split without approximation:

        log_odds = baseline + sum(contribution_i)

    Numeric contributions are measured relative to the training-set mean
    (StandardScaler centres them, so an average value contributes exactly 0).
    Categorical contributions are measured relative to the unweighted mean
    coefficient across that feature's own categories, and the remainder is
    folded into ``baseline`` so the identity above stays exact.
    """

    baseline: float
    contributions: tuple[FeatureContribution, ...]
    log_odds: float

    @property
    def total_contribution(self) -> float:
        return float(sum(c.contribution for c in self.contributions))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_bundle(path: Path | str = DEFAULT_MODEL_PATH) -> ModelBundle:
    """Load and validate the saved artifact. Raises ``ModelLoadError``."""
    path = Path(path)

    if not path.exists():
        raise ModelLoadError(
            f"Model artifact not found at '{path}'.\n\n"
            "Train the pipeline first (from the project root):\n"
            "    python -m src.train"
        )

    try:
        artifact = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
        raise ModelLoadError(
            f"The artifact at '{path}' could not be read ({type(exc).__name__}: {exc}).\n\n"
            "This usually means the file is corrupt or was written with an "
            "incompatible scikit-learn version. Re-run `python -m src.train`."
        ) from exc

    if not isinstance(artifact, Mapping) or "model" not in artifact or "threshold" not in artifact:
        raise ModelLoadError(
            f"The artifact at '{path}' does not have the expected structure. "
            "Expected a mapping with 'model' and 'threshold' keys."
        )

    model = artifact["model"]
    if not hasattr(model, "predict_proba"):
        raise ModelLoadError(
            "The loaded 'model' object does not expose predict_proba(); "
            "it is not a usable classification pipeline."
        )

    try:
        threshold = float(artifact["threshold"])
    except (TypeError, ValueError) as exc:
        raise ModelLoadError(
            f"The saved threshold ({artifact['threshold']!r}) is not a number."
        ) from exc

    if not 0.0 < threshold < 1.0:
        raise ModelLoadError(
            f"The saved threshold ({threshold}) is outside the open interval (0, 1)."
        )

    raw_names = getattr(model, "feature_names_in_", None)
    feature_names = tuple(raw_names) if raw_names is not None else FALLBACK_FEATURE_ORDER

    classifier = _get_classifier(model)
    classifier_name = type(classifier).__name__ if classifier is not None else "Unknown"
    classifier_params: dict[str, Any] = {}
    if classifier is not None:
        for key in ("C", "solver", "class_weight", "max_iter", "random_state"):
            if hasattr(classifier, key):
                classifier_params[key] = getattr(classifier, key)

    n_encoded = None
    coef = getattr(classifier, "coef_", None)
    if coef is not None:
        n_encoded = int(np.asarray(coef).shape[-1])

    return ModelBundle(
        model=model,
        threshold=threshold,
        feature_names=feature_names,
        path=path,
        classifier_name=classifier_name,
        classifier_params=classifier_params,
        n_encoded_features=n_encoded,
    )


def _get_classifier(model: Any) -> Any | None:
    named = getattr(model, "named_steps", None)
    if named:
        if "classifier" in named:
            return named["classifier"]
        return list(named.values())[-1]
    return None


def _get_preprocessor(model: Any) -> Any | None:
    named = getattr(model, "named_steps", None)
    if named and "preprocessor" in named:
        return named["preprocessor"]
    return None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def build_frame(record: Mapping[str, Any], feature_names: Sequence[str]) -> pd.DataFrame:
    """Build a one-row DataFrame with exactly the pipeline's feature columns.

    Column order matches ``feature_names`` so that the fitted ColumnTransformer
    receives the schema it was trained on.
    """
    missing = [name for name in feature_names if name not in record]
    if missing:
        raise FeatureMismatchError(
            "Missing required feature(s): " + ", ".join(missing)
        )

    return pd.DataFrame([{name: record[name] for name in feature_names}], columns=list(feature_names))


def predict(bundle: ModelBundle, record: Mapping[str, Any]) -> PredictionResult:
    """Dropout probability + classification using the saved threshold."""
    frame = build_frame(record, bundle.feature_names)
    probability = float(bundle.model.predict_proba(frame)[0, 1])

    return PredictionResult(
        probability=probability,
        threshold=bundle.threshold,
        is_elevated=bool(probability >= bundle.threshold),
    )


# ---------------------------------------------------------------------------
# Interpretability
# ---------------------------------------------------------------------------


def _encoded_slices(preprocessor: Any) -> list[tuple[str, slice]] | None:
    """Map each original feature to its slice of encoded columns.

    Returns ``None`` if the structure cannot be resolved unambiguously, in which
    case the caller must omit the explanation rather than guess.
    """
    transformers = getattr(preprocessor, "transformers_", None)
    if not transformers:
        return None

    mapping: list[tuple[str, slice]] = []
    cursor = 0

    for name, transformer, columns in transformers:
        if transformer in ("drop", None) or name == "remainder":
            continue
        if transformer == "passthrough":
            for column in columns:
                mapping.append((str(column), slice(cursor, cursor + 1)))
                cursor += 1
            continue

        try:
            out_names = list(transformer.get_feature_names_out(columns))
        except Exception:  # noqa: BLE001
            return None

        columns = list(columns)

        if len(out_names) == len(columns):
            # One encoded column per source column (numeric branch).
            for column in columns:
                mapping.append((str(column), slice(cursor, cursor + 1)))
                cursor += 1
            continue

        # Expanding branch: resolve widths from the one-hot encoder itself.
        categories = _find_categories(transformer)
        if categories is None or len(categories) != len(columns):
            return None

        for column, levels in zip(columns, categories):
            width = len(levels)
            mapping.append((str(column), slice(cursor, cursor + width)))
            cursor += width

    return mapping


def _find_categories(transformer: Any) -> list[Any] | None:
    if hasattr(transformer, "categories_"):
        return list(transformer.categories_)
    named = getattr(transformer, "named_steps", None)
    if named:
        for step in named.values():
            if hasattr(step, "categories_"):
                return list(step.categories_)
    return None


def explain(bundle: ModelBundle, record: Mapping[str, Any]) -> Explanation | None:
    """Exact per-feature log-odds decomposition, or ``None`` if not derivable.

    Returning ``None`` is deliberate: a misleading explanation is worse than no
    explanation, so the UI omits the section rather than approximating.
    """
    preprocessor = _get_preprocessor(bundle.model)
    classifier = _get_classifier(bundle.model)

    if preprocessor is None or classifier is None:
        return None

    coef = getattr(classifier, "coef_", None)
    intercept = getattr(classifier, "intercept_", None)
    if coef is None or intercept is None:
        return None

    coef = np.asarray(coef)
    if coef.ndim != 2 or coef.shape[0] != 1:
        return None  # multiclass / unexpected shape

    weights = coef[0]
    bias = float(np.asarray(intercept).ravel()[0])

    mapping = _encoded_slices(preprocessor)
    if mapping is None:
        return None

    frame = build_frame(record, bundle.feature_names)

    try:
        encoded = np.asarray(preprocessor.transform(frame), dtype=float).ravel()
    except Exception:  # noqa: BLE001
        return None

    if encoded.shape[0] != weights.shape[0]:
        return None
    if mapping and mapping[-1][1].stop != weights.shape[0]:
        return None

    terms = weights * encoded

    baseline = bias
    contributions: list[FeatureContribution] = []

    for feature, span in mapping:
        block = weights[span]
        if block.size > 1:
            # Categorical: measure against the mean weight of this feature's own
            # categories, folding that reference into the baseline.
            reference = float(block.mean())
        else:
            # Numeric: StandardScaler centres the column, so the training mean
            # already contributes exactly zero.
            reference = 0.0

        baseline += reference
        contributions.append(
            FeatureContribution(
                feature=feature,
                value=record.get(feature),
                contribution=float(terms[span].sum() - reference),
            )
        )

    log_odds = float(bias + terms.sum())

    # Sanity gate: the decomposition must reproduce the model's own decision
    # function. If it does not, we have mis-resolved the structure - omit.
    try:
        expected = float(np.asarray(bundle.model.decision_function(frame)).ravel()[0])
        if not np.isclose(log_odds, expected, atol=1e-6):
            return None
    except Exception:  # noqa: BLE001
        pass

    contributions.sort(key=lambda item: abs(item.contribution), reverse=True)

    return Explanation(
        baseline=baseline,
        contributions=tuple(contributions),
        log_odds=log_odds,
    )

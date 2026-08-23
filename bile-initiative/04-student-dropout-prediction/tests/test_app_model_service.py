"""
Integration checks for the Streamlit app's model layer.

These tests assert that the app is a faithful front end for the saved artifact:
the same inputs must produce the same probability and the same threshold
decision as `src/predict.py`, and the interpretability decomposition must
reproduce the pipeline's own decision function exactly.

Run from the project root:
    python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from model_service import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    ELEVATED_RISK,
    LOWER_RISK,
    FeatureMismatchError,
    ModelLoadError,
    build_frame,
    explain,
    load_bundle,
    predict,
)
from schema import (  # noqa: E402
    EXAMPLE_PROFILES,
    FIELDS,
    STABLE_PROFILE,
    STRUGGLING_PROFILE,
    TYPICAL_PROFILE,
    validate,
)

# The exact record hard-coded in src/predict.py.
PREDICT_PY_SAMPLE = {
    "Age": 21,
    "Gender": "Male",
    "Family_Income": 30000,
    "Internet_Access": "Yes",
    "Study_Hours_per_Day": 2.5,
    "Attendance_Rate": 65,
    "Assignment_Delay_Days": 4,
    "Travel_Time_Minutes": 45,
    "Part_Time_Job": "Yes",
    "Scholarship": "No",
    "Stress_Index": 7,
    "GPA": 2.4,
    "Semester_GPA": 2.3,
    "CGPA": 2.5,
    "Semester": "Year 2",
    "Department": "CS",
    "Parental_Education": "High School",
}


@pytest.fixture(scope="module")
def bundle():
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("models/student_dropout_pipeline.joblib is not present")
    return load_bundle(DEFAULT_MODEL_PATH)


# ---------------------------------------------------------------------------
# Artifact contract
# ---------------------------------------------------------------------------


def test_threshold_is_the_saved_one(bundle):
    raw = joblib.load(DEFAULT_MODEL_PATH)
    assert bundle.threshold == float(raw["threshold"]) == 0.55


def test_schema_matches_pipeline_exactly(bundle):
    assert len(bundle.feature_names) == 17
    assert "Student_ID" not in bundle.feature_names
    assert set(bundle.feature_names) == set(FIELDS)


def test_missing_model_file_raises_model_load_error(tmp_path):
    with pytest.raises(ModelLoadError):
        load_bundle(tmp_path / "does_not_exist.joblib")


def test_corrupt_artifact_raises_model_load_error(tmp_path):
    bad = tmp_path / "corrupt.joblib"
    bad.write_bytes(b"not a joblib file")
    with pytest.raises(ModelLoadError):
        load_bundle(bad)


def test_artifact_without_expected_keys_raises(tmp_path):
    bad = tmp_path / "wrong_shape.joblib"
    joblib.dump({"pipeline": None}, bad)
    with pytest.raises(ModelLoadError):
        load_bundle(bad)


# ---------------------------------------------------------------------------
# Parity with src/predict.py
# ---------------------------------------------------------------------------


def _reference_probability(record: dict) -> float:
    """Reproduce src/predict.py exactly, without importing Streamlit code."""
    artifact = joblib.load(DEFAULT_MODEL_PATH)
    frame = pd.DataFrame([record])
    return float(artifact["model"].predict_proba(frame)[0, 1])


def test_matches_predict_py_on_its_own_sample(bundle):
    result = predict(bundle, PREDICT_PY_SAMPLE)
    assert result.probability == pytest.approx(_reference_probability(PREDICT_PY_SAMPLE), abs=1e-12)


@pytest.mark.parametrize("profile_name", list(EXAMPLE_PROFILES))
def test_matches_predict_py_on_every_example_profile(bundle, profile_name):
    record = EXAMPLE_PROFILES[profile_name]
    result = predict(bundle, record)
    assert result.probability == pytest.approx(_reference_probability(record), abs=1e-12)


def test_column_order_does_not_change_the_probability(bundle):
    shuffled = dict(reversed(list(TYPICAL_PROFILE.items())))
    assert predict(bundle, shuffled).probability == pytest.approx(
        predict(bundle, TYPICAL_PROFILE).probability, abs=1e-12
    )


def test_build_frame_uses_pipeline_column_order(bundle):
    frame = build_frame(TYPICAL_PROFILE, bundle.feature_names)
    assert list(frame.columns) == list(bundle.feature_names)
    assert frame.shape == (1, 17)


def test_missing_feature_is_rejected(bundle):
    incomplete = dict(TYPICAL_PROFILE)
    del incomplete["GPA"]
    with pytest.raises(FeatureMismatchError):
        predict(bundle, incomplete)


# ---------------------------------------------------------------------------
# Threshold logic
# ---------------------------------------------------------------------------


def test_labels_follow_the_saved_threshold(bundle):
    for record in EXAMPLE_PROFILES.values():
        result = predict(bundle, record)
        expected = ELEVATED_RISK if result.probability >= bundle.threshold else LOWER_RISK
        assert result.label == expected
        assert result.is_elevated == (result.probability >= bundle.threshold)


def test_stable_profile_is_lower_risk(bundle):
    assert predict(bundle, STABLE_PROFILE).label == LOWER_RISK


def test_struggling_profile_is_elevated_risk(bundle):
    assert predict(bundle, STRUGGLING_PROFILE).label == ELEVATED_RISK


def test_probability_ordering_is_sane(bundle):
    stable = predict(bundle, STABLE_PROFILE).probability
    typical = predict(bundle, TYPICAL_PROFILE).probability
    struggling = predict(bundle, STRUGGLING_PROFILE).probability
    assert stable < typical < struggling


def test_threshold_boundary_is_inclusive(bundle):
    """A probability exactly at the threshold must count as Elevated Risk."""
    from model_service import PredictionResult

    at = PredictionResult(probability=bundle.threshold, threshold=bundle.threshold,
                          is_elevated=bundle.threshold >= bundle.threshold)
    just_below = PredictionResult(
        probability=bundle.threshold - 1e-9,
        threshold=bundle.threshold,
        is_elevated=(bundle.threshold - 1e-9) >= bundle.threshold,
    )
    assert at.label == ELEVATED_RISK
    assert just_below.label == LOWER_RISK


def test_extreme_inputs_stay_in_range(bundle):
    low = dict(TYPICAL_PROFILE, GPA=0.0, Semester_GPA=0.0, CGPA=0.0,
               Attendance_Rate=0.0, Stress_Index=10.0, Study_Hours_per_Day=0.0,
               Assignment_Delay_Days=14, Travel_Time_Minutes=150.0)
    high = dict(TYPICAL_PROFILE, GPA=4.0, Semester_GPA=4.0, CGPA=4.0,
                Attendance_Rate=100.0, Stress_Index=1.0, Study_Hours_per_Day=8.0,
                Assignment_Delay_Days=0, Travel_Time_Minutes=0.0)
    for record in (low, high):
        probability = predict(bundle, record).probability
        assert 0.0 <= probability <= 1.0
    assert predict(bundle, low).probability > predict(bundle, high).probability


# ---------------------------------------------------------------------------
# Interpretability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile_name", list(EXAMPLE_PROFILES))
def test_contributions_reconstruct_the_decision_function(bundle, profile_name):
    record = EXAMPLE_PROFILES[profile_name]
    explanation = explain(bundle, record)
    assert explanation is not None

    frame = build_frame(record, bundle.feature_names)
    expected = float(np.asarray(bundle.model.decision_function(frame)).ravel()[0])

    assert explanation.baseline + explanation.total_contribution == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("profile_name", list(EXAMPLE_PROFILES))
def test_log_odds_match_the_reported_probability(bundle, profile_name):
    record = EXAMPLE_PROFILES[profile_name]
    explanation = explain(bundle, record)
    probability = predict(bundle, record).probability
    recovered = 1.0 / (1.0 + np.exp(-explanation.log_odds))
    assert recovered == pytest.approx(probability, abs=1e-9)


def test_every_feature_appears_exactly_once(bundle):
    explanation = explain(bundle, TYPICAL_PROFILE)
    names = [c.feature for c in explanation.contributions]
    assert sorted(names) == sorted(bundle.feature_names)


def test_contributions_are_sorted_by_magnitude(bundle):
    explanation = explain(bundle, STRUGGLING_PROFILE)
    magnitudes = [abs(c.contribution) for c in explanation.contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_valid_record_passes_cleanly():
    report = validate(dict(TYPICAL_PROFILE))
    assert not report.is_blocking
    assert report.notices == ()


def test_out_of_bounds_value_is_blocking():
    report = validate(dict(TYPICAL_PROFILE, GPA=9.0))
    assert report.is_blocking


def test_unknown_category_is_blocking():
    report = validate(dict(TYPICAL_PROFILE, Department="Medicine"))
    assert report.is_blocking


def test_missing_field_is_blocking():
    record = dict(TYPICAL_PROFILE)
    del record["Stress_Index"]
    assert validate(record).is_blocking


def test_out_of_training_range_is_a_notice_not_an_error():
    report = validate(dict(TYPICAL_PROFILE, Age=38.0))
    assert not report.is_blocking
    assert any("outside the range seen during training" in n for n in report.notices)


def test_inconsistent_gpas_produce_a_notice():
    report = validate(dict(TYPICAL_PROFILE, GPA=4.0, CGPA=1.0, Semester_GPA=4.0))
    assert not report.is_blocking
    assert report.notices


def test_widget_bounds_cover_every_example_profile():
    for record in EXAMPLE_PROFILES.values():
        assert not validate(dict(record)).is_blocking

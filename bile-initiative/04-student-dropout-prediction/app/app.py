"""
EduRisk AI - Student Dropout Early-Warning System.

Presentation layer only. The trained pipeline in
models/student_dropout_pipeline.joblib is loaded as-is and its saved threshold is
applied verbatim; nothing here trains, refits, preprocesses or rescores.

Run from the project root:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:  # supports `streamlit run app/app.py` and direct import
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import streamlit as st

import ui
from model_service import (
    DEFAULT_MODEL_PATH,
    ModelBundle,
    ModelLoadError,
    explain,
    predict,
)
from schema import (
    DEFAULT_PROFILE_NAME,
    EXAMPLE_PROFILES,
    FIELDS,
    FIELD_ORDER,
    GROUPS,
    default_record,
    validate,
)

# Validated on the held-out test set at threshold 0.55 (see src/train.py output).
# These are reported figures, not recomputed here.
PERFORMANCE = {
    "Accuracy": 0.7675,
    "Precision": 0.5045,
    "Recall": 0.7219,
    "F1 Score": 0.5939,
    "ROC-AUC": 0.8204,
}

METRIC_NOTES = {
    "Accuracy": "Share of all test profiles labelled correctly.",
    "Precision": "Of the profiles flagged, the share who did leave.",
    "Recall": "Of the students who left, the share the model caught. Optimised for.",
    "F1 Score": "Harmonic mean of precision and recall.",
    "ROC-AUC": "Ranking quality, independent of any threshold.",
}

NUMBER_FORMAT = {
    "Age": "%.1f",
    "Family_Income": "%.0f",
    "GPA": "%.2f",
    "Semester_GPA": "%.2f",
    "CGPA": "%.2f",
    "Attendance_Rate": "%.1f",
    "Study_Hours_per_Day": "%.1f",
    "Travel_Time_Minutes": "%.0f",
    "Stress_Index": "%.1f",
}

INPUT_KEY_PREFIX = "in_"

# ---------------------------------------------------------------------------
# Product identity - single source of truth for every branding surface
# ---------------------------------------------------------------------------

PRODUCT_NAME = "EduRisk AI"
PRODUCT_SUBTITLE = "Student Dropout Early-Warning System"
PRODUCT_DESCRIPTION = (
    "AI-assisted early-warning support for identifying students who may need "
    "academic intervention."
)
DEVELOPER = "Nabil Mahboub"
PROGRAMME = "BILE Initiative AI & Machine Learning Track"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading the trained pipeline...")
def get_bundle(path_str: str) -> ModelBundle:
    from model_service import load_bundle

    return load_bundle(Path(path_str))


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def init_state() -> None:
    for name, value in default_record().items():
        st.session_state.setdefault(INPUT_KEY_PREFIX + name, value)
    st.session_state.setdefault("assessment", None)
    st.session_state.setdefault("errors", ())


def apply_profile(profile_name: str) -> None:
    """Callback: load an example profile into the form and clear any result."""
    profile = EXAMPLE_PROFILES.get(profile_name)
    if not profile:
        return
    for name, value in profile.items():
        st.session_state[INPUT_KEY_PREFIX + name] = value
    st.session_state["assessment"] = None
    st.session_state["errors"] = ()


def reset_form() -> None:
    apply_profile(DEFAULT_PROFILE_NAME)


def current_record() -> dict[str, Any]:
    return {name: st.session_state[INPUT_KEY_PREFIX + name] for name in FIELD_ORDER}


def run_assessment(bundle: ModelBundle) -> None:
    """Button callback: validate, score with the saved pipeline, store the result.

    Runs as a callback rather than inline so the result exists before the script
    body executes - that is what lets the sidebar show it on the same rerun.
    """
    st.session_state["errors"] = ()
    record = current_record()
    report = validate(record)

    if report.is_blocking:
        st.session_state["assessment"] = None
        st.session_state["errors"] = report.errors
        return

    try:
        result = predict(bundle, record)
    except Exception as exc:  # noqa: BLE001
        st.session_state["assessment"] = None
        st.session_state["errors"] = (
            f"The pipeline could not score this profile ({type(exc).__name__}: {exc}).",
        )
        return

    st.session_state["assessment"] = {
        "record": record,
        "result": result,
        "notices": report.notices,
    }


def is_stale() -> bool:
    assessment = st.session_state.get("assessment")
    return bool(assessment) and assessment["record"] != current_record()


# ---------------------------------------------------------------------------
# Form
# ---------------------------------------------------------------------------


def render_field(name: str) -> None:
    spec = FIELDS[name]
    key = INPUT_KEY_PREFIX + name
    help_text = spec.help or None

    if spec.widget == "select":
        st.selectbox(spec.label, options=list(spec.options), key=key, help=help_text)
        return

    if spec.is_integer:
        st.session_state[key] = int(st.session_state[key])
        st.slider(
            spec.label,
            min_value=int(spec.min_value),
            max_value=int(spec.max_value),
            step=int(spec.step or 1),
            key=key,
            help=help_text,
        )
        return

    st.session_state[key] = float(st.session_state[key])
    common = dict(
        min_value=float(spec.min_value),
        max_value=float(spec.max_value),
        step=float(spec.step or 1.0),
        key=key,
        help=help_text,
        format=NUMBER_FORMAT.get(name),
    )
    if spec.widget == "slider":
        st.slider(spec.label, **common)
    else:
        st.number_input(spec.label, **common)


def render_form() -> None:
    """Four numbered cards, two per row on desktop, stacked on narrow screens."""
    for row_start in range(0, len(GROUPS), 2):
        columns = st.columns(2, gap="medium")
        for offset, (column, group) in enumerate(
            zip(columns, GROUPS[row_start : row_start + 2])
        ):
            index = row_start + offset
            with column:
                with st.container(border=True, key=f"sdcard_group_{index}"):
                    ui.card_head(index + 1, group.title, group.caption)
                    for spec in group.fields:
                        render_field(spec.name)


def input_summary_frame(record: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Section": group.title,
                "Input": spec.label,
                "Value": spec.format_value(record[spec.name]),
            }
            for group in GROUPS
            for spec in group.fields
        ]
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


def interpretation(result) -> str:
    gap = abs(result.margin) * 100
    direction = "above" if result.is_elevated else "below"
    return (
        f"The model estimates a **{result.probability:.1%}** probability that a student with "
        f"this profile does not continue - **{gap:.1f} percentage points {direction}** the "
        f"{result.threshold:.2f} threshold fixed during evaluation. This is a probability, "
        "not a forecast about any individual."
    )


def next_step(result) -> None:
    if result.is_elevated:
        ui.callout(
            "Recommended next step",
            "<strong>Treat this as a request for attention, not a verdict.</strong> Route the "
            "student toward academic advising, a financial-aid review or wellbeing support, "
            "and weigh the flag alongside what staff already know. At this threshold roughly "
            "half of flagged profiles would have continued anyway, so acting on a flag means "
            "<em>checking</em>, never assuming. It must not be used to restrict enrolment, "
            "withdraw funding, or otherwise penalise a student.",
            accent=ui.RISK_HIGH,
        )
    else:
        ui.callout(
            "Recommended next step",
            "<strong>No escalation indicated by this profile.</strong> The threshold is "
            "deliberately permissive so that fewer at-risk students are missed, and the model "
            "still misses about a quarter of those who leave. A Lower Risk result is not a "
            "clearance - routine monitoring and staff judgement continue to apply, and "
            "concerns this data does not capture are still worth acting on.",
            accent=ui.RISK_LOW,
        )


def render_factors(bundle: ModelBundle, record: dict[str, Any]) -> None:
    explanation = explain(bundle, record)
    if explanation is None:
        return  # omit rather than approximate

    ui.section_intro(
        "Factors influencing this assessment",
        "Each input's exact contribution to this student's log-odds, largest first.",
        label="Insights",
    )
    ui.contribution_chart(
        explanation.contributions[:8],
        label_lookup=lambda name: FIELDS[name].label,
        value_lookup=lambda name: FIELDS[name].format_value(record[name]),
    )
    ui.note(
        "These are the model's own weights, <strong>not causes</strong>. Logistic regression "
        "is additive in its encoded inputs, so the bars above plus a baseline term reproduce "
        "the estimate exactly. Numeric inputs are measured against the training-set average, "
        "categorical inputs against the average of their own categories. Associations learned "
        "from synthetic data do not establish that changing an input would change a student's "
        "outcome."
    )


def render_result(bundle: ModelBundle) -> None:
    assessment = st.session_state.get("assessment")
    if not assessment:
        return

    record = assessment["record"]
    result = assessment["result"]

    ui.rule()
    ui.eyebrow("Assessment result")

    if is_stale():
        st.info(
            "Inputs have changed since this assessment was generated. "
            "Select **Assess Dropout Risk** to refresh it.",
            icon=":material/update:",
        )

    ui.result_panel(
        label=result.label,
        is_elevated=result.is_elevated,
        probability=result.probability,
        threshold=result.threshold,
        stats=(
            ("Decision threshold", f"{result.threshold:.2f}"),
            ("Distance from threshold", f"{result.margin * 100:+.1f} pp"),
            ("Estimated to continue", f"{result.retention_probability:.1%}"),
            ("Model recall at 0.55", f"{PERFORMANCE['Recall']:.1%}"),
        ),
    )

    st.markdown(interpretation(result))
    next_step(result)

    for message in assessment["notices"]:
        st.warning(message, icon=":material/info:")

    ui.rule()
    render_factors(bundle, record)

    detail_left, detail_right = st.columns(2, gap="medium")
    with detail_left:
        with st.expander("Input summary used for this assessment"):
            st.dataframe(
                input_summary_frame(record),
                hide_index=True,
                width="stretch",
                height=36 * (len(FIELD_ORDER) + 1) + 3,  # all 17 rows, no inner scroll
            )
    with detail_right:
        with st.expander("How this prediction works"):
            st.markdown(
                f"""
1. The 17 inputs are assembled into a single-row table using exactly the column
   names and order the fitted pipeline expects. `Student_ID` is not included - it
   was dropped before training and is not a model input.
2. That row is passed to the saved scikit-learn `Pipeline`, which performs its own
   imputation, scaling and one-hot encoding. The app never preprocesses values
   itself, so training and serving cannot drift apart.
3. `predict_proba()` returns the probability of the positive class (dropout).
4. The profile is labelled **Elevated Risk** when that probability is greater than
   or equal to **{result.threshold:.2f}**, the threshold stored inside the artifact.
   No other rule is applied, and the threshold is never overridden in the app.

The same three steps run in `src/predict.py`, so the number shown here matches what
the command-line script reports for identical inputs.
                """
            )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


def tab_assessment(bundle: ModelBundle) -> None:
    ui.section_intro(
        "Student risk assessment",
        "Enter the student's current details. All 17 fields are model inputs and none are "
        "optional - load an example profile from the sidebar to see a complete assessment.",
        label="Step 1 · Inputs",
    )
    st.write("")
    render_form()
    st.write("")

    with st.container(border=True, key="sdaction_bar"):
        action, hint = st.columns([1, 2.1], gap="medium", vertical_alignment="center")
        with action:
            st.button(
                "Assess Dropout Risk",
                type="primary",
                width="stretch",
                icon=":material/analytics:",
                on_click=run_assessment,
                args=(bundle,),
            )
        with hint:
            st.markdown(
                '<div class="sd-action-hint">Scores the profile with the saved pipeline and '
                "applies its <strong>fixed 0.55 decision threshold</strong>. The result "
                "appears below.</div>",
                unsafe_allow_html=True,
            )

    for message in st.session_state.get("errors", ()):
        st.error(message, icon=":material/error:")

    render_result(bundle)


def tab_performance(bundle: ModelBundle) -> None:
    ui.section_intro(
        "Held-out test set, threshold 0.55",
        "Measured on the 20% stratified split that was never used for fitting or tuning.",
        label="Validated metrics",
    )
    st.write("")

    ui.metric_tiles(
        [(name, f"{value:.2%}", METRIC_NOTES[name]) for name, value in PERFORMANCE.items()],
        accent_label="Recall",
        accent=ui.LOWERS,  # focus colour, not an alarm colour - recall is what we optimised for
    )

    ui.rule()

    left, right = st.columns([1.4, 1], gap="large")

    with left:
        ui.section_intro("Why recall is the metric that matters here")
        st.markdown(
            """
An early-warning system fails in two different ways, and they do not cost the same.

- A **missed at-risk student** (false negative) is a student who needed support and
  never got offered any. The cost is borne by that student, and the system had one
  chance to catch it.
- A **false alarm** (false positive) is an advising conversation with a student who
  was going to be fine. The cost is staff time.

Because the first failure is far more expensive than the second, the model was tuned
toward **recall**: at the 0.55 threshold it identifies about **72%** of the students
who actually leave. The price is precision near **50%** - roughly half of the flagged
students would have continued anyway. That trade is deliberate, and it is why the
output is framed as *Elevated Risk* rather than a prediction that a student will
drop out.

**ROC-AUC of 82%** describes ranking quality independently of any threshold: given one
student who leaves and one who stays, the model gives the leaver the higher score about
82% of the time. It is the fairest single summary of how much signal the features carry.
            """
        )

    with right:
        with st.container(border=True, key="sdcard_config"):
            ui.card_head(None, "Model configuration", "Read from the loaded artifact.")
            params = bundle.classifier_params
            ui.definition_list(
                [
                    ("Estimator", bundle.classifier_name),
                    ("Regularisation (C)", str(params.get("C", "-"))),
                    ("Solver", str(params.get("solver", "-"))),
                    ("Class weight", str(params.get("class_weight", "-"))),
                    ("Decision threshold", f"{bundle.threshold:.2f}"),
                    ("Model inputs", f"{len(bundle.feature_names)} features"),
                    ("Encoded features", str(bundle.n_encoded_features or "-")),
                    ("Artifact", bundle.path.name),
                ]
            )
        st.write("")
        with st.container(border=True, key="sdcard_prep"):
            ui.card_head(None, "Preprocessing", "Owned entirely by the saved pipeline.")
            st.markdown(
                """
- Numeric columns: median imputation, then standardisation.
- Categorical columns: most-frequent imputation, then one-hot encoding with
  `handle_unknown='ignore'`.
- Fitted on the training split only, so the metrics above are free of leakage.
                """
            )


def tab_responsible(bundle: ModelBundle) -> None:
    ui.section_intro(
        "Responsible use",
        "Stated plainly, because a risk score that travels without its limits does harm.",
        label="Transparency",
    )
    st.write("")

    ui.callout(
        "Read this first",
        "<strong>This is an educational early-warning prototype.</strong> It was built for "
        "the BILE Initiative AI &amp; Machine Learning Track and trained on a "
        "<strong>synthetic dataset</strong>, so its numbers describe patterns in generated "
        "data, not any real student body. Every output is <strong>probabilistic</strong>. "
        "Predictions are intended to <strong>support human decision-making, never to "
        "determine a student's outcome automatically</strong>. A high-risk result should "
        "trigger support or further assessment - never punishment, and never a change to a "
        "student's standing on its own.",
    )
    st.write("")

    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(border=True, key="sdcard_does"):
            ui.card_head(None, "What the model does", "Scope of the prediction.")
            st.markdown(
                """
Estimates the probability that a student with a given profile does not continue,
then compares that probability with a fixed decision threshold to produce a
**Lower Risk** or **Elevated Risk** label.

It is a balanced logistic regression: a linear, fully inspectable model chosen so
that every number the app shows can be traced back to the pipeline's own weights.
                """
            )
    with right:
        with st.container(border=True, key="sdcard_doesnot"):
            ui.card_head(None, "What the model does not do", "Boundaries worth stating.")
            st.markdown(
                """
- It does not diagnose a student, or explain *why* anyone is struggling.
- It does not establish cause. The factors panel shows learned association only.
- It does not account for anything outside its 17 inputs - illness, caregiving,
  a change at home, a good mentor.
- It does not improve with use; it is a fixed artifact loaded from disk.
                """
            )

    st.write("")
    ui.callout(
        "Fairness review required before real-world use",
        "<strong>Gender is currently one of the 17 model features.</strong> It was retained "
        "because it is present in the dataset, and the app does not hide that. Before this "
        "model is used on real students it needs a fairness review - subgroup error rates at "
        "minimum - and an explicit decision about whether the feature should be used at all. "
        "A protected attribute earning predictive weight is a finding to investigate, not a "
        "result to ship.",
        accent=ui.RISK_HIGH,
    )
    st.write("")

    with st.container(border=True, key="sdcard_limits"):
        ui.card_head(None, "Known limitations", "Read these before quoting any number.")
        st.markdown(
            """
- **Synthetic data.** Relationships in the training set were generated, so performance
  on real enrolment records is unknown and probably lower.
- **Precision near 50%.** Around half of the flagged profiles would have continued.
  Acting on a flag must always mean *checking*, not *assuming*.
- **Recall of 72%.** Roughly a quarter of the students who leave are not flagged at all.
  A Lower Risk label is not evidence that a student is fine.
- **Extrapolation.** Values outside the ranges seen in training are flagged in the
  results, but the model will still return a number for them.
- **A threshold is a policy choice.** 0.55 encodes a particular tolerance for false
  alarms. A different institution with different support capacity should revisit it.
            """
        )

    st.write("")
    with st.container(border=True, key="sdcard_about"):
        ui.card_head(None, "About this project", "Provenance of the system you are using.")
        ui.definition_list(
            [
                ("Product", f"{PRODUCT_NAME} — {PRODUCT_SUBTITLE}"),
                ("Developed by", DEVELOPER),
                ("Built for", PROGRAMME),
                ("Model", "Balanced logistic regression"),
                ("Data", "Synthetic dataset (educational use)"),
                ("Status", "Prototype — not for operational decisions"),
            ]
        )

    st.write("")
    with st.expander("The 17 model inputs"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Feature": name,
                        "Shown as": FIELDS[name].label if name in FIELDS else "-",
                        "Section": next(
                            (g.title for g in GROUPS for f in g.fields if f.name == name),
                            "-",
                        ),
                    }
                    for name in bundle.feature_names
                ]
            ),
            hide_index=True,
            width="stretch",
            height=36 * 18 + 3,
        )
        st.caption(
            "`Student_ID` is deliberately excluded: it is an identifier only and was "
            "dropped before training."
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar(bundle: ModelBundle) -> None:
    with st.sidebar:
        ui.brand("ER", PRODUCT_NAME, "Early-warning system")
        st.divider()

        assessment = st.session_state.get("assessment")
        if assessment:
            result = assessment["result"]
            ui.readout(
                label=result.label,
                is_elevated=result.is_elevated,
                probability=result.probability,
                stale=is_stale(),
            )
            st.divider()

        ui.eyebrow("Example profiles")
        choice = st.selectbox(
            "Load an example",
            options=list(EXAMPLE_PROFILES),
            key="profile_choice",
            label_visibility="collapsed",
        )
        st.button(
            "Load profile",
            width="stretch",
            on_click=apply_profile,
            args=(choice,),
            icon=":material/download:",
        )
        st.button(
            "Reset form",
            width="stretch",
            on_click=reset_form,
            icon=":material/restart_alt:",
        )
        st.caption(
            "Illustrative starting points, not real students. They are not guaranteed to "
            "land on a particular risk label."
        )

        st.divider()

        ui.eyebrow("Model")
        params = bundle.classifier_params
        ui.definition_list(
            [
                ("Estimator", "Balanced logistic regression"),
                ("C", str(params.get("C", "-"))),
                ("Solver", str(params.get("solver", "-"))),
                ("Threshold", f"{bundle.threshold:.2f}"),
                ("Inputs", str(len(bundle.feature_names))),
                ("ROC-AUC", f"{PERFORMANCE['ROC-AUC']:.2%}"),
            ]
        )

        st.divider()
        st.caption(
            "Educational prototype trained on synthetic data. Supports human judgement; "
            "does not replace it."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title=f"{PRODUCT_NAME} | {PRODUCT_SUBTITLE}",
        page_icon=":material/school:",
        layout="wide",
        initial_sidebar_state="auto",  # expanded on desktop, collapsed on mobile
    )
    ui.inject_styles()

    try:
        bundle = get_bundle(str(DEFAULT_MODEL_PATH))
    except ModelLoadError as exc:
        st.error(str(exc), icon=":material/error:")
        st.caption(
            "The app intentionally stops here rather than falling back to an untrained "
            "model, so no prediction is ever produced from an unverified artifact."
        )
        st.stop()
        return

    init_state()
    render_sidebar(bundle)

    ui.hero(
        brand=PRODUCT_NAME,
        subtitle=PRODUCT_SUBTITLE,
        lede=PRODUCT_DESCRIPTION,
        stats=(
            ("Model", "Balanced logistic regression", None),
            ("ROC-AUC", f"{PERFORMANCE['ROC-AUC']:.2%}", None),
            ("Recall", f"{PERFORMANCE['Recall']:.2%}", None),
            ("Decision threshold", f"{bundle.threshold:.2f}", None),
            ("Status", "Educational ML prototype", ui.RISK_HIGH),
        ),
    )

    assessment_tab, performance_tab, responsible_tab = st.tabs(
        ["Risk Assessment", "Model Performance", "Responsible AI"]
    )
    with assessment_tab:
        tab_assessment(bundle)
    with performance_tab:
        tab_performance(bundle)
    with responsible_tab:
        tab_responsible(bundle)

    ui.footer(
        brand_line=f"{PRODUCT_NAME} · {PRODUCT_SUBTITLE}",
        meta_lines=(
            f"Developed by {DEVELOPER}",
            f"Built for the {PROGRAMME}",
        ),
    )


if __name__ == "__main__":
    main()

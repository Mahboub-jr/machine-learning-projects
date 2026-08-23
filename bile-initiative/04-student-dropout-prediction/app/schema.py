"""
Input schema, form grouping, example profiles and validation rules.

The ranges and category levels recorded here mirror what was observed in
``data/raw/student_dropout_dataset_v3.csv``. They drive widget bounds and
"outside observed range" notices; they are presentation metadata only and are
never used to transform values before the model sees them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

WidgetKind = Literal["slider", "number", "select"]


@dataclass(frozen=True)
class Field:
    name: str
    label: str
    widget: WidgetKind
    help: str = ""
    unit: str = ""
    # numeric
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    is_integer: bool = False
    observed_min: float | None = None
    observed_max: float | None = None
    fmt: str = "{:g}"
    # categorical
    options: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_numeric(self) -> bool:
        return self.widget in ("slider", "number")

    def format_value(self, value: Any) -> str:
        if self.is_numeric:
            try:
                text = self.fmt.format(float(value))
            except (TypeError, ValueError):
                text = str(value)
        else:
            text = str(value)
        return f"{text}{self.unit}" if self.unit else text


@dataclass(frozen=True)
class FieldGroup:
    title: str
    caption: str
    fields: tuple[Field, ...]


# ---------------------------------------------------------------------------
# The 17 model inputs, grouped for the form. Student_ID is intentionally absent.
# ---------------------------------------------------------------------------

GROUPS: tuple[FieldGroup, ...] = (
    FieldGroup(
        title="Student Profile",
        caption="Enrolment context and demographics.",
        fields=(
            Field(
                name="Age",
                label="Age",
                widget="number",
                unit=" yrs",
                min_value=15.0,
                max_value=40.0,
                step=0.5,
                observed_min=17.0,
                observed_max=29.6,
                fmt="{:.1f}",
                help="Age in years. Training data covers roughly 17-30.",
            ),
            Field(
                name="Gender",
                label="Gender",
                widget="select",
                options=("Female", "Male"),
                help="Recorded in the dataset; retained as a model input.",
            ),
            Field(
                name="Semester",
                label="Year of study",
                widget="select",
                options=("Year 1", "Year 2", "Year 3", "Year 4"),
            ),
            Field(
                name="Department",
                label="Department",
                widget="select",
                options=("Arts", "Business", "CS", "Engineering", "Science"),
            ),
        ),
    ),
    FieldGroup(
        title="Academic Performance",
        caption="Grades and coursework timeliness.",
        fields=(
            Field(
                name="GPA",
                label="Current GPA",
                widget="slider",
                min_value=0.0,
                max_value=4.0,
                step=0.01,
                observed_min=0.0,
                observed_max=4.0,
                fmt="{:.2f}",
                help="Grade point average for the current period, on a 0-4 scale.",
            ),
            Field(
                name="Semester_GPA",
                label="Semester GPA",
                widget="slider",
                min_value=0.0,
                max_value=4.0,
                step=0.01,
                observed_min=0.0,
                observed_max=4.0,
                fmt="{:.2f}",
            ),
            Field(
                name="CGPA",
                label="Cumulative GPA",
                widget="slider",
                min_value=0.0,
                max_value=4.0,
                step=0.01,
                observed_min=0.0,
                observed_max=4.0,
                fmt="{:.2f}",
            ),
            Field(
                name="Assignment_Delay_Days",
                label="Average assignment delay",
                widget="slider",
                unit=" days",
                min_value=0.0,
                max_value=14.0,
                step=1.0,
                is_integer=True,
                observed_min=0.0,
                observed_max=8.0,
                fmt="{:.0f}",
                help="Mean number of days assignments are submitted late.",
            ),
        ),
    ),
    FieldGroup(
        title="Attendance & Study Behaviour",
        caption="Engagement, workload and daily friction.",
        fields=(
            Field(
                name="Attendance_Rate",
                label="Attendance rate",
                widget="slider",
                unit="%",
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                observed_min=38.2,
                observed_max=100.0,
                fmt="{:.1f}",
            ),
            Field(
                name="Study_Hours_per_Day",
                label="Study hours per day",
                widget="slider",
                unit=" h",
                min_value=0.0,
                max_value=14.0,
                step=0.1,
                observed_min=0.5,
                observed_max=9.0,
                fmt="{:.1f}",
            ),
            Field(
                name="Travel_Time_Minutes",
                label="Travel time to campus",
                widget="slider",
                unit=" min",
                min_value=0.0,
                max_value=150.0,
                step=1.0,
                observed_min=5.0,
                observed_max=74.9,
                fmt="{:.0f}",
                help="One-way commute time.",
            ),
            Field(
                name="Stress_Index",
                label="Stress index",
                widget="slider",
                min_value=1.0,
                max_value=10.0,
                step=0.1,
                observed_min=1.0,
                observed_max=10.0,
                fmt="{:.1f}",
                help="Self-reported stress on a 1-10 scale; 10 is highest.",
            ),
        ),
    ),
    FieldGroup(
        title="Socioeconomic & Support Factors",
        caption="Household resources and access to support.",
        fields=(
            Field(
                name="Family_Income",
                label="Annual family income",
                widget="number",
                min_value=0.0,
                max_value=1_000_000.0,
                step=1000.0,
                observed_min=25_000.0,
                observed_max=316_601.0,
                fmt="{:,.0f}",
                help="Dataset currency units. Observed range is roughly 25,000-317,000.",
            ),
            Field(
                name="Scholarship",
                label="Receives scholarship",
                widget="select",
                options=("No", "Yes"),
            ),
            Field(
                name="Part_Time_Job",
                label="Holds a part-time job",
                widget="select",
                options=("No", "Yes"),
            ),
            Field(
                name="Internet_Access",
                label="Reliable internet access",
                widget="select",
                options=("No", "Yes"),
            ),
            Field(
                name="Parental_Education",
                label="Highest parental education",
                widget="select",
                options=("High School", "Bachelor", "Master", "PhD"),
            ),
        ),
    ),
)

FIELDS: dict[str, Field] = {f.name: f for group in GROUPS for f in group.fields}
FIELD_ORDER: tuple[str, ...] = tuple(FIELDS)


# ---------------------------------------------------------------------------
# Example profiles
# ---------------------------------------------------------------------------

# "Typical" values are the dataset medians (numeric) and modes (categorical).
TYPICAL_PROFILE: dict[str, Any] = {
    "Age": 21.0,
    "Gender": "Female",
    "Semester": "Year 2",
    "Department": "Science",
    "GPA": 2.35,
    "Semester_GPA": 2.35,
    "CGPA": 2.35,
    "Assignment_Delay_Days": 2,
    "Attendance_Rate": 81.8,
    "Study_Hours_per_Day": 4.0,
    "Travel_Time_Minutes": 30.0,
    "Stress_Index": 5.5,
    "Family_Income": 29_741.0,
    "Scholarship": "No",
    "Part_Time_Job": "No",
    "Internet_Access": "Yes",
    "Parental_Education": "Bachelor",
}

STABLE_PROFILE: dict[str, Any] = {
    "Age": 20.0,
    "Gender": "Female",
    "Semester": "Year 3",
    "Department": "CS",
    "GPA": 3.6,
    "Semester_GPA": 3.5,
    "CGPA": 3.55,
    "Assignment_Delay_Days": 0,
    "Attendance_Rate": 94.0,
    "Study_Hours_per_Day": 5.5,
    "Travel_Time_Minutes": 15.0,
    "Stress_Index": 3.0,
    "Family_Income": 75_000.0,
    "Scholarship": "Yes",
    "Part_Time_Job": "No",
    "Internet_Access": "Yes",
    "Parental_Education": "Master",
}

STRUGGLING_PROFILE: dict[str, Any] = {
    "Age": 22.5,
    "Gender": "Male",
    "Semester": "Year 2",
    "Department": "Arts",
    "GPA": 1.3,
    "Semester_GPA": 1.2,
    "CGPA": 1.4,
    "Assignment_Delay_Days": 5,
    "Attendance_Rate": 62.0,
    "Study_Hours_per_Day": 2.0,
    "Travel_Time_Minutes": 55.0,
    "Stress_Index": 8.5,
    "Family_Income": 25_000.0,
    "Scholarship": "No",
    "Part_Time_Job": "Yes",
    "Internet_Access": "No",
    "Parental_Education": "High School",
}

EXAMPLE_PROFILES: dict[str, dict[str, Any]] = {
    "Typical student (dataset medians)": TYPICAL_PROFILE,
    "Academically stable profile": STABLE_PROFILE,
    "Academically struggling profile": STRUGGLING_PROFILE,
}

DEFAULT_PROFILE_NAME = "Typical student (dataset medians)"


def default_record() -> dict[str, Any]:
    return dict(TYPICAL_PROFILE)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...] = ()
    notices: tuple[str, ...] = ()

    @property
    def is_blocking(self) -> bool:
        return bool(self.errors)


def validate(record: dict[str, Any]) -> ValidationReport:
    """Check a submitted record for blocking errors and soft notices."""
    errors: list[str] = []
    notices: list[str] = []

    for name, spec in FIELDS.items():
        if name not in record or record[name] is None:
            errors.append(f"{spec.label} is required.")
            continue

        value = record[name]

        if spec.is_numeric:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                errors.append(f"{spec.label} must be a number.")
                continue

            if spec.min_value is not None and numeric < spec.min_value:
                errors.append(
                    f"{spec.label} must be at least {spec.format_value(spec.min_value)}."
                )
            if spec.max_value is not None and numeric > spec.max_value:
                errors.append(
                    f"{spec.label} must be at most {spec.format_value(spec.max_value)}."
                )
            if not errors and spec.observed_min is not None and spec.observed_max is not None:
                if numeric < spec.observed_min or numeric > spec.observed_max:
                    notices.append(
                        f"{spec.label} ({spec.format_value(numeric)}) falls outside the range "
                        f"seen during training ({spec.format_value(spec.observed_min)} to "
                        f"{spec.format_value(spec.observed_max)}). The model is extrapolating here."
                    )
        else:
            if str(value) not in spec.options:
                errors.append(
                    f"{spec.label} must be one of: {', '.join(spec.options)}."
                )

    if not errors:
        gpa = float(record["GPA"])
        semester_gpa = float(record["Semester_GPA"])
        cgpa = float(record["CGPA"])

        if abs(gpa - cgpa) > 1.5:
            notices.append(
                "Current GPA and cumulative GPA differ by more than 1.5 points. "
                "Double-check the entry if that was not intended."
            )
        if abs(semester_gpa - gpa) > 1.5:
            notices.append(
                "Semester GPA and current GPA differ by more than 1.5 points. "
                "Double-check the entry if that was not intended."
            )
        if float(record["Study_Hours_per_Day"]) > 10:
            notices.append(
                "Study hours above 10 per day are unusual and were not represented "
                "in the training data."
            )
        if record["Semester"] == "Year 1" and float(record["Age"]) >= 26:
            notices.append(
                "A first-year student aged 26 or above is uncommon in the training data."
            )

    return ValidationReport(errors=tuple(errors), notices=tuple(notices))

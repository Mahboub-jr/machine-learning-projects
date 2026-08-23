import joblib
import pandas as pd


MODEL_PATH = "models/student_dropout_pipeline.joblib"


def main():
    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    threshold = artifact["threshold"]

    sample = pd.DataFrame([
        {
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
    ])

    probability = model.predict_proba(sample)[0, 1]

    prediction = int(probability >= threshold)

    print(f"Dropout probability: {probability:.4f}")
    print(f"Threshold: {threshold}")
    print(f"Prediction: {prediction}")

    if prediction == 1:
        print("Result: Student may be at risk of dropout.")
    else:
        print("Result: Student is predicted to remain enrolled.")


if __name__ == "__main__":
    main()
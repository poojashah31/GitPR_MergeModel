import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained RFE model and the StandardScaler
try:
    # The 'model.pkl' file contains the RFE object, which in turn holds the trained Ridge model.
    # The RFE object is what we'll use for prediction, as it handles feature selection internally.
    rfe_model = joblib.load('model.pkl')
    
    # Load the StandardScaler object. This is crucial for correctly preprocessing new data.
    scaler = joblib.load('scaler.pkl')

except FileNotFoundError:
    st.error("One or more model files (`model.pkl`, `scaler.pkl`) could not be found. Please ensure they are in the same directory as this script.")
    st.stop()

# ---

st.title("GitHub Pull Request Merge Time Predictor")
st.write("This app uses a machine learning model to estimate the time it will take for a pull request to be merged.")
st.write("---")

# User Input Section
st.subheader("Pull Request Details")
st.write("Please provide the following information about your pull request:")

# User inputs for the six selected features
lines_added = st.number_input("Lines Added", min_value=0, value=150)
number_of_commits = st.number_input("Number of Commits", min_value=1, value=10)
files_changed = st.number_input("Files Changed", min_value=1, value=5)
weekday_opened = st.selectbox("Weekday Opened", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

# We still need `lines_deleted` to calculate `total_lines_changed`, even though it's not a final feature.
lines_deleted = st.number_input("Lines Deleted", min_value=0, value=50, help="This is used to calculate total lines changed.")

# The app also needs inputs for all features from the original dataset to correctly pass to the scaler,
# but we can hide these from the user as they are not part of the final model features.
with st.expander("Show additional inputs (required for preprocessing)"):
    st.info("These values are needed for the data scaling step, but are not used by the final model for prediction.")
    review_comments = st.number_input("Review Comments", min_value=0, value=3)
    approvals = st.number_input("Approvals", min_value=0, value=2)
    is_urgent = st.selectbox("Is Urgent?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    labels_count = st.number_input("Labels Count", min_value=0, value=2)
    reviewer_count = st.number_input("Reviewer Count", min_value=1, value=2)
    ci_cd_passed = st.selectbox("CI/CD Passed?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    draft_status = st.selectbox("Draft Status", options=[0, 1], format_func=lambda x: "Draft" if x == 1 else "Not a Draft")
    author_experience_level = st.selectbox("Author Experience Level", options=["Beginner", "Intermediate", "Advanced"])
    time_of_day = st.selectbox("Time of Day", options=["Morning", "Afternoon", "Evening", "Night"])

# ---

# Prediction Logic
if st.button("Predict Merge Time"):
    # Create the complete input DataFrame with all original features
    user_input = {
        'lines_added': lines_added,
        'lines_deleted': lines_deleted,
        'number_of_commits': number_of_commits,
        'files_changed': files_changed,
        'review_comments': review_comments,
        'approvals': approvals,
        'is_urgent': is_urgent,
        'labels_count': labels_count,
        'reviewer_count': reviewer_count,
        'ci_cd_passed': ci_cd_passed,
        'draft_status': draft_status,
        'author_experience_level': author_experience_level,
        'weekday_opened': weekday_opened,
        'time_of_day': time_of_day,
        'total_lines_changed': lines_added + lines_deleted
    }
    input_df = pd.DataFrame([user_input])

    # Preprocessing pipeline
    # 1. Define feature sets
    num_cols = ['lines_added', 'lines_deleted', 'number_of_commits', 'files_changed',
                'review_comments', 'approvals', 'is_urgent', 'labels_count',
                'reviewer_count', 'ci_cd_passed', 'draft_status', 'total_lines_changed']
    cat_cols = ['author_experience_level', 'weekday_opened', 'time_of_day']

    # 2. Scale numeric features
    input_num_scaled = pd.DataFrame(scaler.transform(input_df[num_cols]), columns=num_cols)

    # 3. One-hot encode categorical features and align columns
    # We must include ALL possible categories that the model was trained on
    cat_mapping = {
        'author_experience_level': ["Beginner", "Intermediate", "Advanced"],
        'weekday_opened': ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        'time_of_day': ["Morning", "Afternoon", "Evening", "Night"]
    }
    
    encoded_df = pd.DataFrame()
    for col, categories in cat_mapping.items():
        dummies = pd.get_dummies(input_df[col], prefix=col, dtype=int)
        for cat in categories[1:]: # Drop the first category, as per `drop_first=True`
            dummy_col_name = f"{col}_{cat}"
            if dummy_col_name not in dummies.columns:
                dummies[dummy_col_name] = 0
        encoded_df = pd.concat([encoded_df, dummies.drop(f"{col}_{categories[0]}", axis=1, errors='ignore')], axis=1)

    # 4. Concatenate and predict
    # The RFE model expects the same column names and order as the preprocessed data during training
    full_input_prepared = pd.concat([input_num_scaled, encoded_df], axis=1)

    try:
        # The RFE object handles feature selection automatically when you call `predict` on the full dataset.
        prediction_hours = rfe_model.predict(full_input_prepared)[0]

        st.success(f"Predicted Merge Time: **{prediction_hours:.2f} hours**")
        st.info("This is an estimation based on the trained model. Actual times may vary.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please check your input values and the model files.")

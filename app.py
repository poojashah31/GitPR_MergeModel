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
    # In a real-world scenario, you would save this scaler after fitting it during training.
    # Based on the notebook, the file was not explicitly saved, so this step is necessary.
    scaler = joblib.load('scaler.pkl')

except FileNotFoundError:
    st.error("One or more model files (`model.pkl`, `scaler.pkl`) could not be found. Please ensure they are in the same directory as this script.")
    st.info("Note: The original notebook did not explicitly save the `StandardScaler` object. You must add `joblib.dump(scaler, 'scaler.pkl')` to your notebook after fitting it to `X_train[num_cols]` to make this app runnable.")
    st.stop()

# Get the list of selected features from the RFE model
# This assumes the RFE model was trained on the full set of prepared features.
# The selected features are: 'lines_added', 'number_of_commits', 'files_changed', 'total_lines_changed', 'weekday_opened_Monday', 'weekday_opened_Sunday'
selected_feature_names = rfe_model.get_feature_names_out(input_features=[
    'lines_added', 'lines_deleted', 'number_of_commits', 'files_changed',
    'review_comments', 'approvals', 'is_urgent', 'labels_count',
    'reviewer_count', 'ci_cd_passed', 'draft_status', 'total_lines_changed',
    'author_experience_level_Beginner', 'author_experience_level_Intermediate',
    'weekday_opened_Monday', 'weekday_opened_Saturday', 'weekday_opened_Sunday',
    'weekday_opened_Thursday', 'weekday_opened_Tuesday', 'weekday_opened_Wednesday',
    'time_of_day_Evening', 'time_of_day_Morning', 'time_of_day_Night'
])

# ---

# Streamlit App UI
st.title("GitHub Pull Request Merge Time Predictor")
st.write("This app uses a machine learning model to estimate the time it will take for a pull request to be merged.")
st.write("---")

# User Input Section
st.subheader("Pull Request Details")
st.write("Please provide the following information about your pull request:")

# User inputs for the six selected features. We explicitly ask for these as they are the most important.
lines_added = st.number_input("Lines Added", min_value=0, value=150)
number_of_commits = st.number_input("Number of Commits", min_value=1, value=10)
files_changed = st.number_input("Files Changed", min_value=1, value=5)
weekday_opened = st.selectbox("Weekday Opened", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

# We still need inputs for all original features to correctly apply the scaler and one-hot encoding
# that the model's preprocessing pipeline expects, even if they aren't ultimately selected.
# We'll put these in an expander to keep the main interface clean.
with st.expander("Show/Hide All Preprocessing Inputs"):
    st.info("These values are used to correctly format the data before it's passed to the model. They may not be directly used for the final prediction if they were not selected by RFE.")
    lines_deleted = st.number_input("Lines Deleted", min_value=0, value=50, help="Used for calculating total_lines_changed.")
    review_comments = st.number_input("Review Comments", min_value=0, value=3)
    approvals = st.number_input("Approvals", min_value=0, value=2)
    is_urgent = st.selectbox("Is Urgent?", options=[0, 1], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
    labels_count = st.number_input("Labels Count", min_value=0, value=2)
    reviewer_count = st.number_input("Reviewer Count", min_value=1, value=2)
    ci_cd_passed = st.selectbox("CI/CD Passed?", options=[0, 1], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
    draft_status = st.selectbox("Draft Status", options=[0, 1], index=1, format_func=lambda x: "Draft" if x == 1 else "Not a Draft")
    author_experience_level = st.selectbox("Author Experience Level", options=["Intermediate", "Beginner", "Advanced"], index=0)
    time_of_day = st.selectbox("Time of Day", options=["Afternoon", "Evening", "Morning", "Night"], index=0)

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
    # 1. Define feature sets based on the original data structure
    num_cols = ['lines_added', 'lines_deleted', 'number_of_commits', 'files_changed',
                'review_comments', 'approvals', 'is_urgent', 'labels_count',
                'reviewer_count', 'ci_cd_passed', 'draft_status', 'total_lines_changed']
    cat_cols = ['author_experience_level', 'weekday_opened', 'time_of_day']

    # 2. Scale numeric features
    input_num_scaled = pd.DataFrame(scaler.transform(input_df[num_cols]), columns=num_cols)

    # 3. One-hot encode categorical features and align columns
    # This step is critical to match the training data's structure
    full_cat_list = {
        'author_experience_level': ["Beginner", "Intermediate", "Advanced"],
        'weekday_opened': ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        'time_of_day': ["Morning", "Afternoon", "Evening", "Night"]
    }
    
    encoded_df = pd.DataFrame()
    for col, categories in full_cat_list.items():
        dummies = pd.get_dummies(input_df[col], prefix=col, dtype=int)
        for cat in categories[1:]:
            dummy_col_name = f"{col}_{cat}"
            if dummy_col_name not in dummies.columns:
                dummies[dummy_col_name] = 0
        encoded_df = pd.concat([encoded_df, dummies.drop(f"{col}_{categories[0]}", axis=1, errors='ignore')], axis=1)

    # 4. Concatenate scaled numeric and encoded categorical data
    full_input_prepared = pd.concat([input_num_scaled, encoded_df], axis=1)

    try:
        # The RFE object handles feature selection automatically when you call `predict` on the full dataset.
        prediction_hours = rfe_model.predict(full_input_prepared)[0]

        st.success(f"Predicted Merge Time: **{prediction_hours:.2f} hours**")
        st.info("This is an estimation based on the trained model. Actual times may vary.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please check your input values and confirm that the model files (`model.pkl`, `scaler.pkl`) exist and were saved correctly.")

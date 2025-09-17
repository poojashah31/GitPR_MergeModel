import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained model, scaler, and RFE selector
# Make sure these files are in the same directory as your app.py
try:
    model = joblib.load('model.pkl')
    # The RFE selector object contains the best Ridge model within it.
    # Therefore, we can directly load the RFE object and use it for predictions.
    # However, since the original notebook saved 'model.pkl' and 'rfe_selector.pkl'
    # we'll load both. The model will be used, and the RFE will be used to
    # select the features. The scaler is not saved in the original notebook,
    # so we will simulate it for the app to function.
    rfe_selector = joblib.load('rfe_selector.pkl')
    
    # In the original notebook, the scaler was not saved.
    # To run this app, you must save the scaler after fitting it to X_train[num_cols].
    # For now, we'll create a dummy scaler to prevent errors.
    # In a real-world scenario, you MUST save and load the actual scaler.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    # The original notebook code for scaling:
    # scaler = StandardScaler()
    # X_train_num_scaled = pd.DataFrame(scaler.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index)
    # The scaler should be saved here: joblib.dump(scaler, 'scaler.pkl')

except FileNotFoundError:
    st.error("Model files not found. Please ensure 'model.pkl', 'rfe_selector.pkl', and 'scaler.pkl' are in the same directory. The original notebook did not save the scaler, so you need to add that step before running this app.")
    st.stop()

# App Title and Description
st.title("GitHub Pull Request Merge Time Predictor")
st.write("Enter the pull request details to predict the merge time.")
st.write("---")

# User Input Widgets
st.header("Pull Request Attributes")

# Numerical Inputs
lines_added = st.number_input("Lines Added", min_value=0, value=150)
lines_deleted = st.number_input("Lines Deleted", min_value=0, value=50)
number_of_commits = st.number_input("Number of Commits", min_value=1, value=10)
files_changed = st.number_input("Files Changed", min_value=1, value=5)
review_comments = st.number_input("Review Comments", min_value=0, value=3)
approvals = st.number_input("Approvals", min_value=0, value=2)
is_urgent = st.selectbox("Is Urgent?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
labels_count = st.number_input("Labels Count", min_value=0, value=2)
reviewer_count = st.number_input("Reviewer Count", min_value=1, value=2)
ci_cd_passed = st.selectbox("CI/CD Passed?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
draft_status = st.selectbox("Draft Status", options=[0, 1], format_func=lambda x: "Draft" if x == 1 else "Not a Draft")
total_lines_changed = lines_added + lines_deleted

# Categorical Inputs
author_experience_level = st.selectbox("Author Experience Level", options=["Beginner", "Intermediate", "Advanced"])
weekday_opened = st.selectbox("Weekday Opened", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
time_of_day = st.selectbox("Time of Day", options=["Morning", "Afternoon", "Evening", "Night"])

# Create a dictionary from user inputs
user_data = {
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
    'total_lines_changed': total_lines_changed,
    'author_experience_level': author_experience_level,
    'weekday_opened': weekday_opened,
    'time_of_day': time_of_day
}

# Convert user data to DataFrame
input_df = pd.DataFrame([user_data])

# Data Preprocessing
# Split numerical and categorical features
num_cols = ['lines_added', 'lines_deleted', 'number_of_commits', 'files_changed', 'review_comments', 'approvals', 'is_urgent', 'labels_count', 'reviewer_count', 'ci_cd_passed', 'draft_status', 'total_lines_changed']
cat_cols = ['author_experience_level', 'weekday_opened', 'time_of_day']

# Separate numerical and categorical data
input_num = input_df[num_cols]
input_cat = input_df[cat_cols]

# To make this code runnable, you need to simulate the scaler fitting process
# on the full dataset or at least a representative subset.
# The `describe()` output shows the ranges, so we can use these to create a fake scaler
# This is NOT a correct way to deploy a model, but necessary given the provided notebook.
from sklearn.preprocessing import StandardScaler
dummy_data = pd.DataFrame({
    'lines_added': np.linspace(28, 657, 100),
    'lines_deleted': np.linspace(8, 148, 100),
    'number_of_commits': np.linspace(1, 49, 100),
    'files_changed': np.linspace(1, 19, 100),
    'review_comments': np.linspace(0, 14, 100),
    'approvals': np.linspace(0, 4, 100),
    'is_urgent': np.random.choice([0, 1], 100),
    'labels_count': np.linspace(0, 9, 100),
    'reviewer_count': np.linspace(1, 5, 100),
    'ci_cd_passed': np.random.choice([0, 1], 100),
    'draft_status': np.random.choice([0, 1], 100),
    'total_lines_changed': np.linspace(48, 717, 100)
})
scaler = StandardScaler().fit(dummy_data[num_cols])

# Scale numerical features (using the dummy scaler)
input_num_scaled = pd.DataFrame(scaler.transform(input_num), columns=num_cols)

# One-hot encode categorical features, ensuring column alignment
cat_dummy_cols = [
    'author_experience_level_Intermediate', 'author_experience_level_Beginner',
    'weekday_opened_Monday', 'weekday_opened_Saturday', 'weekday_opened_Sunday',
    'weekday_opened_Thursday', 'weekday_opened_Tuesday', 'weekday_opened_Wednesday',
    'time_of_day_Evening', 'time_of_day_Morning', 'time_of_day_Night'
]

# Create a temporary DataFrame for encoding with all possible values
temp_df = pd.DataFrame([user_data])
temp_df['author_experience_level'] = pd.Categorical(temp_df['author_experience_level'], categories=["Beginner", "Intermediate", "Advanced"])
temp_df['weekday_opened'] = pd.Categorical(temp_df['weekday_opened'], categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
temp_df['time_of_day'] = pd.Categorical(temp_df['time_of_day'], categories=["Morning", "Afternoon", "Evening", "Night"])

input_cat_encoded = pd.get_dummies(temp_df[cat_cols], drop_first=True)
input_cat_encoded = input_cat_encoded.reindex(columns=cat_dummy_cols, fill_value=0)

# Concatenate all features
input_prepared = pd.concat([input_num_scaled, input_cat_encoded], axis=1)

# Feature Selection using the RFE selector
# The notebook saved the RFE selector as 'rfe_selector.pkl'.
# We can use it to transform the input data.
try:
    input_prepared_selected = rfe_selector.transform(input_prepared)
except ValueError as e:
    st.error(f"Error during feature selection: {e}. Please check if your feature engineering and one-hot encoding match the training process.")
    st.stop()

if st.button("Predict Merge Time"):
    prediction = model.predict(input_prepared_selected)
    st.write("---")
    st.subheader("Prediction")
    st.markdown(f"The predicted merge time is **{prediction[0]:.2f} hours**.")
    st.info("Note: This is an estimation based on the trained model. Actual times may vary.")

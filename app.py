import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration for a better layout
st.set_page_config(
    page_title="Pull Request Merge Time Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use st.cache_resource to load the model and RFE selector only once
@st.cache_resource
def load_model_and_selector():
    """Loads the pre-trained model and RFE selector from disk."""
    try:
        model = joblib.load('model.pkl')
        rfe_selector = joblib.load('rfe_selector.pkl')
        return model, rfe_selector
    except FileNotFoundError:
        st.error("Model or RFE selector files not found. Please upload 'model.pkl' and 'rfe_selector.pkl'.")
        return None, None

model, rfe_selector = load_model_and_selector()

# List of all features used in the original model
# This list is based on the one-hot encoding applied in the provided notebook.
categorical_features = {
    'author_experience_level': ['Beginner', 'Intermediate', 'Advanced'],
    'weekday_opened': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    'time_of_day': ['Morning', 'Afternoon', 'Evening', 'Night']
}

# --- Main Page UI ---
st.title('Pull Request Merge Time Predictor')
st.markdown("""
    This app predicts the time it will take for a pull request to be merged,
    based on a machine learning model.
""")

if model and rfe_selector:
    st.sidebar.header('Input Features')

    # --- User Input Widgets ---
    # Numerical features
    lines_added = st.sidebar.number_input('Lines Added', min_value=0, value=100)
    lines_deleted = st.sidebar.number_input('Lines Deleted', min_value=0, value=50)
    number_of_commits = st.sidebar.number_input('Number of Commits', min_value=1, value=5)
    files_changed = st.sidebar.number_input('Files Changed', min_value=1, value=3)
    review_comments = st.sidebar.number_input('Review Comments', min_value=0, value=2)
    approvals = st.sidebar.number_input('Approvals', min_value=0, value=1)
    reviewer_count = st.sidebar.number_input('Reviewer Count', min_value=1, value=2)

    # Binary features
    is_urgent = st.sidebar.selectbox('Is Urgent?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    ci_cd_passed = st.sidebar.selectbox('CI/CD Passed?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    draft_status = st.sidebar.selectbox('Draft Status', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

    # Categorical features
    author_experience = st.sidebar.selectbox(
        'Author Experience Level',
        categorical_features['author_experience_level']
    )
    weekday_opened = st.sidebar.selectbox(
        'Weekday Opened',
        categorical_features['weekday_opened']
    )
    time_of_day = st.sidebar.selectbox(
        'Time of Day',
        categorical_features['time_of_day']
    )

    # Calculate the `total_lines_changed` feature from user input
    total_lines_changed = lines_added + lines_deleted

    # --- Prediction Logic ---
    # Create a DataFrame from user input
    user_input = pd.DataFrame({
        'lines_added': [lines_added],
        'lines_deleted': [lines_deleted],
        'number_of_commits': [number_of_commits],
        'files_changed': [files_changed],
        'review_comments': [review_comments],
        'approvals': [approvals],
        'is_urgent': [is_urgent],
        'reviewer_count': [reviewer_count],
        'ci_cd_passed': [ci_cd_passed],
        'draft_status': [draft_status],
        'total_lines_changed': [total_lines_changed]
    })

    # One-hot encode categorical features and append to the DataFrame
    def one_hot_encode_user_input(df):
        df_encoded = df.copy()
        
        # Experience Level
        for level in categorical_features['author_experience_level']:
            df_encoded[f'author_experience_level_{level}'] = 1 if level == author_experience else 0

        # Weekday Opened
        for weekday in categorical_features['weekday_opened']:
            df_encoded[f'weekday_opened_{weekday}'] = 1 if weekday == weekday_opened else 0
        
        # Time of Day
        for tod in categorical_features['time_of_day']:
            df_encoded[f'time_of_day_{tod}'] = 1 if tod == time_of_day else 0
        
        return df_encoded

    input_df_encoded = one_hot_encode_user_input(user_input)

    # Apply the RFE selector to get the correct features for the model
    try:
        selected_features_mask = rfe_selector.support_
        selected_features_names = np.array(rfe_selector.feature_names_in_)[selected_features_mask]
        final_input_df = input_df_encoded[selected_features_names]

    except Exception as e:
        st.error(f"Error applying feature selector. Please ensure the features match the trained model. Details: {e}")
        final_input_df = None

    # Display the final input dataframe
    st.subheader('Final Processed Input')
    if final_input_df is not None:
        st.dataframe(final_input_df)

        if st.button('Predict Merge Time'):
            # Make a prediction
            with st.spinner('Predicting...'):
                prediction = model.predict(final_input_df)[0]
                prediction_in_hours = max(0, prediction) # Ensure no negative predictions

            st.subheader('Predicted Merge Time')
            st.success(f"The predicted merge time is approximately **{prediction_in_hours:.2f} hours**.")

else:
    st.warning("Please upload the required 'model.pkl' and 'rfe_selector.pkl' files.")

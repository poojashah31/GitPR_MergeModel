# GitPR Merge Time Predictor

A Machine Learning application that predicts the estimated time required to merge a GitHub Pull Request (PR) based on its characteristics, review activity, and development workflow.

## Overview

Pull requests are an essential part of collaborative software development, but the time required to review and merge them can vary significantly. **GitPR Merge Time Predictor** uses a trained machine learning model to estimate PR merge time and help developers better understand the factors associated with the merging process.

The application provides an interactive Streamlit interface where users can enter pull request details and receive an estimated merge time in hours.

## Features

- **Merge Time Prediction:** Estimates the time required to merge a pull request.
- **Interactive Interface:** Enter PR details through a simple Streamlit dashboard.
- **PR Metrics Analysis:** Uses development and review-related features, including lines added and deleted, number of commits, files changed, review comments, approvals, and reviewer count.
- **Workflow Indicators:** Considers urgency, CI/CD status, draft status, author experience, opening weekday, and time of day.
- **Feature Selection:** Uses Recursive Feature Elimination (RFE) to select the features expected by the trained model.
- **Pre-trained Model:** Loads the saved model and feature selector for predictions without retraining during application use.

## Tech Stack

- **Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Model Persistence:** Joblib
- **Web Application:** Streamlit

## Project Structure

```text
GitPR_MergeModel/
├── app.py                # Streamlit application
├── model.pkl             # Pre-trained ML model
├── rfe_selector.pkl      # Saved RFE feature selector
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Input Features

The application uses the following information to generate a prediction:

| Category | Features |
|---|---|
| Code Changes | Lines added, lines deleted, total lines changed, files changed |
| Development Activity | Number of commits |
| Code Review | Review comments, approvals, reviewer count |
| Workflow Status | Urgency, CI/CD status, draft status |
| Author Information | Author experience level |
| Timing | Weekday opened, time of day |

## How It Works

1. **Input PR Details:** Enter the pull request metrics and workflow information using the sidebar.
2. **Prepare Features:** The application processes numerical, binary, and categorical inputs into a model-compatible format.
3. **Select Features:** The saved RFE selector identifies the features required by the trained model.
4. **Generate Prediction:** The trained model estimates the time required to merge the pull request.
5. **Display Results:** The predicted merge time is displayed in hours.

## Installation and Setup

### Prerequisites

- Python 3.9 or a compatible Python version
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/poojashah31/GitPR_MergeModel.git
cd GitPR_MergeModel
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the environment:

**Windows**
```bash
venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to access the application.

## Example Workflow

1. Enter the number of lines added and deleted.
2. Specify commits, files changed, review comments, approvals, and reviewers.
3. Select workflow indicators and timing information.
4. Click **Predict Merge Time**.
5. View the estimated merge time in hours.

## Limitations

- Predictions depend on the quality and representativeness of the model's training data.
- Estimated merge time is not a guarantee of actual pull request completion time.
- Prediction quality may vary across repositories, teams, review practices, and development workflows.
- The application requires the saved model and feature selector files to be available locally.

## Future Improvements

- Integrate with the GitHub API to retrieve real pull request data.
- Compare predicted merge time with actual merge time.
- Add model evaluation metrics and prediction confidence information where appropriate.
- Visualize important features and their influence on predictions.
- Deploy the application for public access.

## Author

**Pooja Shah**

GitHub: [@poojashah31](https://github.com/poojashah31)

## License

A license has not yet been specified for this project.

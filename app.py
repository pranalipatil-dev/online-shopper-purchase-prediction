
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Configure the application
st.set_page_config(
    page_title="Shopper Insight Lab",
    page_icon="🛒",
    layout="wide"
)


st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}

/* Style headings */
h1, h2, h3 {
    color: #20BFA9;
}

/* Style metric cards */
[data-testid="stMetric"] {
    background-color: #17332F;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #28665B;
}

/* Make metric labels visible */
[data-testid="stMetricLabel"] {
    color: #D4F5EB;
}

/* Make metric values visible */
[data-testid="stMetricValue"] {
    color: #FFFFFF;
}

</style>
""", unsafe_allow_html=True)


# Define the saved model paths
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

# Get the folder containing this application
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_model(model_name):
    # Get the selected model file
    file_name = MODEL_FILES[model_name]

    # Build the complete model path
    model_path = os.path.join(
        BASE_DIR, "model", file_name
    )

    # Load the saved preprocessing and model pipeline
    return joblib.load(model_path)


def load_test_data(uploaded_file):
    # Read the uploaded test dataset
    return pd.read_csv(uploaded_file)


def validate_test_data(data, model):
    # Get the features expected by the trained model
    required_cols = list(model.feature_names_in_)

    # Check for missing input columns
    missing_cols = [
        col for col in required_cols
        if col not in data.columns
    ]

    # Check that the target column exists
    if "Revenue" not in data.columns:
        st.error("Target column 'Revenue' is missing.")
        return None, None

    if missing_cols:
        st.error(f"Missing input features: {missing_cols}")
        return None, None

    # Reject missing values in required columns
    if data[required_cols + ["Revenue"]].isnull().any().any():
        st.error("Test data contains missing values.")
        return None, None

    # Separate input features and target
    X_test = data[required_cols].copy()

    # Convert target values into 0 and 1
    target_values = data["Revenue"].astype(str).str.lower()

    if not target_values.isin(
        ["true", "false", "1", "0"]
    ).all():
        st.error("Revenue must contain True/False or 1/0.")
        return None, None

    y_test = target_values.map({
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0
    }).astype(int)

    return X_test, y_test



def evaluate_model(model, X_test, y_test):

    # Predict the target classes
    y_pred = model.predict(X_test)

    # Get scores for AUC calculation
    if hasattr(model, "predict_proba"):
        # Use probabilities for supported models
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        # Use decision scores for SVM
        y_score = model.decision_function(X_test)

    # Calculate all six evaluation metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": (
            roc_auc_score(y_test, y_score)
            if y_test.nunique() == 2
            else np.nan
        ),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    return metrics, y_pred



def display_metrics(metrics):
    # Display metrics in two rows
    cols = st.columns(3)

    for i, name in enumerate(
        ["Accuracy", "AUC", "Precision"]
    ):
        cols[i].metric(name, f"{metrics[name]:.4f}")

    cols = st.columns(3)

    for i, name in enumerate(
        ["Recall", "F1", "MCC"]
    ):
        cols[i].metric(name, f"{metrics[name]:.4f}")


def plot_confusion_matrix(y_test, y_pred):

    # Calculate the confusion matrix
    cm = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    )

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="BuGn",
        xticklabels=["No Purchase", "Purchase"],
        yticklabels=["No Purchase", "Purchase"],
        ax=ax
    )

    # Add labels and title
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title("Confusion Matrix")

    # Adjust the plot layout
    fig.tight_layout()

    # Display the plot
    st.pyplot(fig, clear_figure=False)

    # Close the figure
    plt.close(fig)



def display_classification_report(y_test, y_pred):
    # Generate the classification report
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No Purchase", "Purchase"],
        output_dict=True,
        zero_division=0
    )

    # Convert the report into a table
    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )


def display_classification_report(y_test, y_pred):
    # Generate the classification report
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No Purchase", "Purchase"],
        output_dict=True,
        zero_division=0
    )

    # Convert the report into a table
    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )


def compare_all_models(X_test, y_test):

    # Store evaluation results for all models
    results = []

    for model_name in MODEL_FILES:

        # Load the saved trained model
        model = load_model(model_name)

        # Calculate metrics on uploaded test data
        metrics, _ = evaluate_model(
            model, X_test, y_test
        )

        # Add model name to the results
        metrics["ML Model Name"] = model_name
        results.append(metrics)

    # Convert results into a comparison table
    comparison = pd.DataFrame(results)

    # Arrange columns
    comparison = comparison[
        [
            "ML Model Name",
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1",
            "MCC"
        ]
    ]

    return comparison



def main():
    # Application heading
    st.title("🛒 Shopper Insight Lab")
    st.write(
        "Evaluate machine learning models for "
        "online purchase intention prediction."
    )

    st.caption(
        "Dataset: UCI Online Shoppers Purchasing Intention"
    )

    st.divider()

    # Display application information
    with st.sidebar:
        st.header("About This Project")
        st.write("Classification Type: Binary")
        st.write("Input Features: 17")
        st.write("Target: Revenue")
        st.write("Models Available: 5")
        st.info(
            "Upload test data containing all input "
            "features and the Revenue target column."
        )

    # Upload test dataset
    st.subheader("1. Upload Test Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is None:
        st.info("Upload test_data.csv to begin evaluation.")
        return

    try:
        test_data = load_test_data(uploaded_file)
    except Exception as error:
        st.error(f"Unable to read CSV: {error}")
        return

    st.success("Test dataset uploaded successfully!")

    # Show basic dataset information
    st.write("Number of test records:", len(test_data))

    with st.expander("Preview Uploaded Dataset"):
        st.dataframe(
            test_data.head(10),
            use_container_width=True
        )

    # Select the classification model
    st.subheader("2. Select Classification Model")

    model_name = st.selectbox(
        "Choose a trained model",
        list(MODEL_FILES.keys())
    )

    # Load the selected model
    try:
        model = load_model(model_name)
    except Exception as error:
        st.error(f"Unable to load model: {error}")
        return

    # Validate test dataset
    X_test, y_test = validate_test_data(
        test_data, model
    )

    if X_test is None:
        return

    # Evaluate the selected model
    try:
        metrics, y_pred = evaluate_model(
            model, X_test, y_test
        )
    except Exception as error:
        st.error(f"Evaluation failed: {error}")
        return

    # Display evaluation metrics
    st.subheader("3. Model Evaluation Results")
    st.write("Selected Model:", model_name)

    display_metrics(metrics)

    if np.isnan(metrics["AUC"]):
        st.warning(
            "AUC is unavailable because the uploaded "
            "test data contains only one target class."
        )

    st.divider()

    # Display confusion matrix and classification report
    st.subheader("4. Classification Performance")

    left, right = st.columns(2)

    with left:
        st.write("**Confusion Matrix**")
        plot_confusion_matrix(y_test, y_pred)

    with right:
        st.write("**Classification Report**")
        display_classification_report(
            y_test, y_pred
        )

    # Show actual and predicted values
    with st.expander("View Actual vs Predicted Results"):
        results = pd.DataFrame({
            "Actual Revenue": y_test.to_numpy(),
            "Predicted Revenue": y_pred
        })

        st.dataframe(
            results.head(100),
            use_container_width=True
        )


    # Compare all five models on the uploaded test data
    st.divider()
    st.subheader("5. Comparison of All Models")

    comparison = compare_all_models(X_test, y_test)

    # Display the comparison table
    st.dataframe(
        comparison.round(4),
        use_container_width=True
    )

    # Identify the model with the highest F1 score
    best_index = comparison["F1"].idxmax()
    best_name = comparison.loc[
        best_index, "ML Model Name"
    ]

    st.success(
        f"Highest F1 Score: {best_name}"
    )


    st.caption(
        "Predictions are generated using saved "
        "trained pipelines. No model retraining "
        "is performed on uploaded test data."
    )


if __name__ == "__main__":
    main()

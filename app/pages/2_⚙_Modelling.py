import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model.classification.classification_models import LogisticRegressionModel, DecisionTreeClassifierModel, SVMClassifierModel
from autoop.core.ml.model.regression.regression_models import LinearRegressionModel, DecisionTreeRegressorModel, SVMRegressorModel
from autoop.core.ml.metric import MeanSquaredError, RootMeanSquaredError, RSquared, Accuracy, Precision, Recall
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.feature import detect_feature_types


st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

# datasets = automl.registry.list(type="dataset")
# Load existing datasets
st.subheader("Select Dataset")
datasets = [dataset.name for dataset in automl.registry.list(type="dataset")]
dataset_name = st.selectbox("Choose a Dataset", datasets)
for dataset in automl.registry.list(type="dataset"):
    if dataset.name == dataset_name:
        selected_dataset = dataset.id
        break
dataset = automl.registry.get(selected_dataset)

# Detect and select features
st.subheader("Select Features")
features = detect_feature_types(dataset)
chosen_input_features = st.multiselect("Select Input Features", [f.name for f in features if f.type == "numerical"])
chosen_target_feature = st.selectbox("Select Target Feature", [f.name for f in features])
input_features = [f for f in features if f.name in chosen_input_features]
target_feature = next((f for f in features if f.name == chosen_target_feature), None)

# Determine task type (classification or regression)
task_type = "classification" if any(f.name == target_feature and f.type == "categorical" for f in features) else "regression"
st.write(f"Detected task type: {task_type}")

# Select model based on task type
st.subheader("Select Model")
model = None
if task_type == "classification":
    model_choice = st.selectbox("Choose a Model", ["Logistic Regression", "Decision Tree Classifier", "SVM Classifier"])
    if model_choice == "Logistic Regression":
        model = LogisticRegressionModel()
    elif model_choice == "Decision Tree Classifier":
        model = DecisionTreeClassifierModel()
    elif model_choice == "SVM Classifier":
        model = SVMClassifierModel()
elif task_type == "regression":
    model_choice = st.selectbox("Choose a Model", ["Linear Regression", "Decision Tree Regressor", "SVM Regressor"])
    if model_choice == "Linear Regression":
        model = LinearRegressionModel()
    elif model_choice == "Decision Tree Regressor":
        model = DecisionTreeRegressorModel()
    elif model_choice == "SVM Regressor":
        model = SVMRegressorModel()

# Select dataset split
st.subheader("Dataset Split")
split_ratio = st.slider("Train/Test Split Ratio", min_value=0.1, max_value=0.9, value=0.8)

# Select metrics
st.subheader("Select Metrics")
metrics = []
if task_type == "classification":
    metric_choices = st.multiselect("Choose Metrics", ["Accuracy", "Precision", "Recall"])
    if "Accuracy" in metric_choices:
        metrics.append(Accuracy())
    if "Precision" in metric_choices:
        metrics.append(Precision())
    if "Recall" in metric_choices:
        metrics.append(Recall())
elif task_type == "regression":
    metric_choices = st.multiselect("Choose Metrics", ["Mean Squared Error", "Root Mean Squared Error", "R Squared"])
    if "Mean Squared Error" in metric_choices:
        metrics.append(MeanSquaredError())
    if "Root Mean Squared Error" in metric_choices:
        metrics.append(RootMeanSquaredError())
    if "R Squared" in metric_choices:
        metrics.append(RSquared())

# Pipeline summary
st.subheader("Pipeline Summary")
st.write(f"Model: {model_choice}")
st.write(f"Input Features: {chosen_input_features}")
st.write(f"Target Feature: {chosen_target_feature}")
st.write(f"Dataset Split Ratio: {split_ratio}")
st.write("Selected Metrics:")
for m in metrics:
    st.write(f"{m.__class__.__name__}")

# Train pipeline and show results
st.subheader("Train and Evaluate Pipeline")
if st.button("Run Pipeline"):
    pipeline = Pipeline(metrics=metrics, dataset=dataset, model=model, input_features=input_features, target_feature=target_feature, split=split_ratio)
    result = pipeline.execute()
    st.write("Training Metrics:")
    for index in range(len(metrics)):
        st.write(f"{result['training metrics'][index][0]}: {result['training metrics'][index][1]:.4f}")
    st.write("Testing Metrics:")
    for index in range(len(metrics)):
        st.write(f"{result['testing metrics'][index][0]}: {result['testing metrics'][index][1]:.4f}")
    st.write("Predictions:", result["predictions"])

import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

# datasets = automl.registry.list(type="dataset")

st.title("Dataset Management")

# Upload CSV and create Dataset
st.subheader("Upload and Create Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if "dataset" not in st.session_state:
    st.session_state["dataset"] = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    dataset_name = st.text_input("Dataset Name", "MyDataset")
    asset_path = st.text_input("Asset Path", "mydataset")

    if st.button("Create Dataset"):
        st.session_state["dataset"] = Dataset.from_dataframe(df, name=dataset_name, asset_path=asset_path)
        st.write("Dataset created:", dataset_name)

# Save Dataset
st.subheader("Save Dataset")
if st.button("Save Dataset"):
    automl.registry.register(st.session_state["dataset"])
    st.write("Dataset saved to registry.")

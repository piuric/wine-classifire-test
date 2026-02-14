import streamlit as st
import pandas as pd
import numpy as np
import pickle, json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix)

st.set_page_config(page_title="Wine Quality Classifier", layout="wide")
st.title("Wine Quality Classifier")
st.write("Upload wine test data and select a model to evaluate.")

MODELS = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl",
}

@st.cache_resource
def load_pkl(path):
    return pickle.load(open(path, "rb"))

scaler = load_pkl("model/scaler.pkl")
features = json.load(open("model/features.json"))

# sidebar
st.sidebar.header("Configuration")
model_name = st.sidebar.selectbox("Select Model", list(MODELS.keys()))

with open("data/test.csv", "rb") as f:
    st.sidebar.download_button("Download sample test data", f, "test.csv", "text/csv")

# file upload
uploaded = st.file_uploader("Upload test CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    if "good" not in df.columns:
        st.error("CSV needs a 'good' column with true labels (0 = not good, 1 = good)")
        st.stop()

    missing = [f for f in features if f not in df.columns]
    if missing:
        st.error(f"Missing feature columns: {missing}")
        st.stop()

    X = df[features]
    y = df["good"]

    model = load_pkl(MODELS[model_name])
    X_sc = scaler.transform(X)
    preds = model.predict(X_sc)
    probs = model.predict_proba(X_sc)[:, 1]

    # metrics
    st.subheader(f"Evaluation Metrics - {model_name}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy_score(y, preds):.4f}")
    c1.metric("AUC Score", f"{roc_auc_score(y, probs):.4f}")
    c2.metric("Precision", f"{precision_score(y, preds):.4f}")
    c2.metric("Recall", f"{recall_score(y, preds):.4f}")
    c3.metric("F1 Score", f"{f1_score(y, preds):.4f}")
    c3.metric("MCC", f"{matthews_corrcoef(y, preds):.4f}")

    # confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Good", "Good"],
                yticklabels=["Not Good", "Good"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
else:
    st.info("Upload a CSV file to get started. Download sample test data from the sidebar.")

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_data
from src.model import LearningCurveTool

st.set_page_config(page_title="Learning Curve", page_icon="📉", layout="wide")
st.title("📉 Learning Curve Analyzer")
st.markdown("Diagnose bias/variance by plotting training and validation scores vs dataset size.")

dataset_name = st.selectbox("Dataset", ["wine", "iris", "breast_cancer"])
X, y, fn = load_data(dataset_name)

if st.button("Generate Learning Curve", type="primary"):
    tool = LearningCurveTool()
    results = tool.compute(X, y)
    st.metric("Train-Val Gap", f"{results['gap']:.3f}")
    import plotly.graph_objects as go
    fig = go.Figure()
    x = results["train_sizes"]
    fig.add_trace(go.Scatter(x=x, y=results["train_mean"], mode="lines+markers", name="Training", line=dict(color="steelblue")))
    fig.add_trace(go.Scatter(x=x, y=results["val_mean"], mode="lines+markers", name="Validation", line=dict(color="crimson")))
    fig.update_layout(xaxis_title="Training Examples", yaxis_title="Accuracy", height=500)
    st.plotly_chart(fig, use_container_width=True)
    df = pd.DataFrame({
        "Train Size": results["train_sizes"],
        "Train Score": results["train_mean"],
        "Val Score": results["val_mean"],
    })
    st.dataframe(df.set_index("Train Size"), use_container_width=True)

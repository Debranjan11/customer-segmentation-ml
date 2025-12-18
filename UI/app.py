# ui/app.py
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from backend.controller import (
    run_rule_based_segmentation,
    run_ml_segmentation
)

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Segmentation System",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>📊 Customer Segmentation System</h1>
    <p style='text-align: center; font-size:18px;'>
    Rule-Based & Machine Learning Based Customer Analysis
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload Customer Dataset (CSV)",
    type=["csv"]
)

segmentation_type = st.sidebar.radio(
    "Select Segmentation Method",
    ["Rule-Based Segmentation", "ML-Based Segmentation"]
)

k_value = None
if segmentation_type == "ML-Based Segmentation":
    k_value = st.sidebar.slider(
        "Number of Clusters (K)",
        min_value=2,
        max_value=6,
        value=3
    )

run_button = st.sidebar.button("🚀 Run Segmentation")

# -----------------------------
# Main Content
# -----------------------------
if uploaded_file is None:
    st.info("👈 Upload a CSV file to begin analysis.")
    st.stop()

# Save uploaded file temporarily
file_path = "data/uploaded.csv"
with open(file_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

# Preview data
st.subheader("📄 Dataset Preview")
df_preview = pd.read_csv(file_path)
st.dataframe(df_preview.head(), use_container_width=True)

# -----------------------------
# Run Segmentation
# -----------------------------
if run_button:

    with st.spinner("Running segmentation..."):
        if segmentation_type == "Rule-Based Segmentation":
            result = run_rule_based_segmentation(file_path)
            segmented_df = result["data"]
            summary = result["summary"]
            segment_column = "Rule_Segment"

        else:
            result = run_ml_segmentation(file_path, k=k_value)
            segmented_df = result["data"]
            summary = result["summary"]
            segment_column = "ML_Segment"

    st.success("✅ Segmentation completed successfully!")

    # -----------------------------
    # Summary Section
    # -----------------------------
    st.subheader("📊 Segment Summary")

    cols = st.columns(len(summary))
    for col, (segment, count) in zip(cols, summary.items()):
        col.metric(label=str(segment), value=count)

    # -----------------------------
    # Visualization
    # -----------------------------
    
    st.subheader("📈 Segment Visualization")

    fig, ax = plt.subplots(figsize=(3, 2))

    pd.Series(summary).plot(
        kind="bar",
        ax=ax,
        width=0.6
    )

    ax.set_xlabel("Segment", fontsize=8)
    ax.set_ylabel("Number of Customers", fontsize=8)
    ax.set_title("Customer Distribution by Segment", fontsize=10)

    ax.tick_params(axis='both', labelsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()

    st.pyplot(fig)

    # -----------------------------
    # Segmented Data
    # -----------------------------
    st.subheader("🧾 Segmented Customer Data")
    st.dataframe(segmented_df, use_container_width=True)

    st.subheader("🧠 PCA Cluster Visualization")

    if segment_column == "ML_Segment":

    # Smaller, presentation-friendly figure
        fig, ax = plt.subplots(figsize=(4, 3))

        scatter = ax.scatter(
            segmented_df["PC1"],
            segmented_df["PC2"],
            c=segmented_df["ML_Segment"],
            cmap="viridis",
            s=40,          # marker size (important)
            alpha=0.8
    )

    ax.set_xlabel("Principal Component 1", fontsize=8)
    ax.set_ylabel("Principal Component 2", fontsize=8)
    ax.set_title("Customer Clusters (PCA Projection)", fontsize=12)

    ax.tick_params(axis='both', labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()

    st.pyplot(fig)


    # -----------------------------
    # Download Option
    # -----------------------------
    csv = segmented_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Segmented Data",
        data=csv,
        file_name="segmented_customers.csv",
        mime="text/csv"
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; color: gray;'>
    Final Year Project | Customer Segmentation using ML
    </p>
    """,
    unsafe_allow_html=True
)

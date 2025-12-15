# backend/controller.py

import pandas as pd

from backend.services.preprocessing import preprocess_data
from backend.services.feature_engineering import rule_based_segmentation
from backend.services.ml_train import train_kmeans_model
from backend.services.ml_predict import predict_ml_segments
from backend.services.pca_visualization import apply_pca


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load customer data from CSV file.
    """
    return pd.read_csv(file_path)


def run_rule_based_segmentation(file_path: str) -> dict:
    """
    Execute rule-based customer segmentation.
    """

    # Load raw data
    df = load_data(file_path)

    # Rule-based segmentation uses original scale
    segmented_df = rule_based_segmentation(df)

    # Summary for UI
    summary = segmented_df["Rule_Segment"].value_counts().to_dict()

    return {
        "data": segmented_df,
        "summary": summary
    }


def run_ml_segmentation(file_path: str, k: int = 3) -> dict:
    """
    Execute ML-based customer segmentation using K-Means.
    """

    # Load raw data
    original_df = load_data(file_path)

    # Preprocess data for ML
    processed_df = preprocess_data(original_df)

    # Train ML model
    model_path = train_kmeans_model(processed_df, k)

    # Predict segments
    segmented_df = predict_ml_segments(
        original_df=original_df,
        processed_df=processed_df,
        model_path=model_path
    )
    #pca for visualization
    pca_df = apply_pca(processed_df)
    segmented_df["PC1"] = pca_df["PC1"]
    segmented_df["PC2"] = pca_df["PC2"]

    # Summary for UI
    summary = segmented_df["ML_Segment"].value_counts().to_dict()

    return {
        "data": segmented_df,
        "summary": summary,
        "model_path": model_path
    }

def generate_segment_explanations(df: pd.DataFrame, segment_col: str) -> dict:
    """
    Generate human-readable explanations for each segment
    based on feature averages.
    """

    explanations = {}

    grouped = df.groupby(segment_col)

    for segment, group in grouped:
        avg_income = group["Annual Income (k$)"].mean()
        avg_spending = group["Spending Score (1-100)"].mean()
        avg_age = group["Age"].mean()

        explanation = (
            f"Average Age: {avg_age:.1f} years\n"
            f"Average Income: ${avg_income:.1f}k\n"
            f"Average Spending Score: {avg_spending:.1f}\n"
        )

        # Simple interpretation logic
        if avg_spending > 60 and avg_income > 70:
            explanation += "💡 High-value, premium customers."
        elif avg_spending < 40 and avg_income < 40:
            explanation += "💡 Low engagement, price-sensitive customers."
        else:
            explanation += "💡 Medium-value, balanced customers."

        explanations[str(segment)] = explanation

    return explanations

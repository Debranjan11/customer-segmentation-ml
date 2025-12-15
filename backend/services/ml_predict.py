# backend/services/ml_predict.py

import joblib
import pandas as pd


def predict_ml_segments(
    original_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    model_path: str
) -> pd.DataFrame:
    """
    Predict customer segments using trained ML model.

    Args:
        original_df (pd.DataFrame): Original customer data
        processed_df (pd.DataFrame): Preprocessed numerical data
        model_path (str): Path to trained model

    Returns:
        pd.DataFrame: Original DataFrame with ML_Segment column
    """

    # Load trained model
    model = joblib.load(model_path)

    # Predict clusters
    cluster_labels = model.predict(processed_df)

    # Attach predictions to original data
    result_df = original_df.copy()
    result_df["ML_Segment"] = cluster_labels

    return result_df

# backend/services/feature_engineering.py

import pandas as pd


def rule_based_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign rule-based customer segments using business logic.

    Adds a new column: Rule_Segment
    """

    segmented_df = df.copy()

    segments = []

    for _, row in segmented_df.iterrows():
        income = row["Annual Income (k$)"]
        spending = row["Spending Score (1-100)"]

        if income >= 70 and spending >= 60:
            segments.append("High Value")
        elif income <= 40 and spending <= 40:
            segments.append("Low Value")
        else:
            segments.append("Medium Value")

    segmented_df["Rule_Segment"] = segments

    return segmented_df

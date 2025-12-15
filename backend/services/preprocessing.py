# backend/services/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler


NUMERIC_FEATURES = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess customer data:
    - Select relevant features
    - Handle missing values
    - Scale numerical columns

    Returns a new DataFrame
    """

    # Select required columns
    df = df[NUMERIC_FEATURES].copy()

    # Handle missing values (simple strategy)
    df.fillna(df.mean(), inplace=True)

    # Scale numerical features
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df)

    processed_df = pd.DataFrame(
        scaled_values,
        columns=NUMERIC_FEATURES
    )

    return processed_df

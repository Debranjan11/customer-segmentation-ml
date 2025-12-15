# backend/services/ml_train.py

import os
import joblib
import pandas as pd
from sklearn.cluster import KMeans

MODEL_DIR = "models"
MODEL_NAME = "kmeans.pkl"


def train_kmeans_model(df: pd.DataFrame, k: int = 3) -> str:
    """
    Train K-Means clustering model and save it to disk.

    Args:
        df (pd.DataFrame): Preprocessed numerical data
        k (int): Number of clusters

    Returns:
        str: Path to saved model file
    """

    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Initialize and train model
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(df)

    # Save trained model
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    joblib.dump(kmeans, model_path)

    return model_path

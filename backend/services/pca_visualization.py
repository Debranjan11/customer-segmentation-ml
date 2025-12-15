# backend/services/pca_visualization.py

import pandas as pd
from sklearn.decomposition import PCA


def apply_pca(processed_df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """
    Apply PCA to reduce data to n_components dimensions.

    Returns a DataFrame with PCA components.
    """

    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(processed_df)

    pca_df = pd.DataFrame(
        components,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    return pca_df
